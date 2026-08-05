# CacheEOPD 过程记录

C2C 融入 EOPD 学生 rollout（路线 A）的实验流水账：做了什么改动、跑了什么实验、
得到什么结果、踩了什么坑。按时间顺序记，便于回溯"当时为什么这么做"。

设计说明见 `README.md`，本文件只记过程与结论。

环境：apex-llm，6×48GB。GPU0 被 VLLM 占满，GPU1-5 各约 7GB 可用（`knhdu` 的任务占着）。
Python 必须用 `/home/knhdu/anaconda3/envs/rosetta/bin/python`（transformers 4.52.4；
Qwen3 需要 ≥4.51，另一个 4.45.2 的环境会报 `KeyError: 'qwen3'`）。
`from_pretrained` 的路径不能带 `~`，不会展开。

---

## 阶段一：搭骨架，验证融合机制正确性

**目标**：把 teacher 的 KV-Cache 投影融合进 student 的前缀 KV，让 student 带着
这个 fused cache 去生成。先不管效果好坏，只确认机制本身没写错。

**改动**
- `fused_kv.py` —— 核心模块。teacher/student 各做一次前缀前向拿 KV，
  逐层过 `C2CProjector` 融合。层映射按相对深度对齐（Qwen3-4B 36 层 → 1.7B 28 层）。
- `c2c_hf_rollout.py` —— `C2CHFRollout`，继承 verl 的 `HFRollout`，
  把 `generate()` 换成「构造 fused cache → 带 cache 自回归解码」。
- `prototype_generate.py` / `smoke_test_rollout.py` / `eval_fused_kv.py` —— 原型与冒烟。

融合公式直接用 C2C 原码（`rosetta/model/projector.py:1493`），未作修改：

```python
output_key   = target_key + key_gate * norm_key_scalar * projected_key
output_value = target_value + value_gate * norm_value_scalar * projected_value
```

**结果**：零初始化下 fused 生成与原生 `generate` 逐 token 一致，随机初始化下 KV
明显偏移（相对 L2 0.105/0.127）。机制正确，融合通路确实接通。

提交 `756951d`。

---

## 阶段二：训练 projector（连续三轮失败）

**目标**：冻结 teacher/student，只训 `C2CProjector`，让 student 在 fused KV 下的
输出分布逼近 teacher。损失是 response 全位置的 KL(student ‖ teacher)。

**改动**
- `fused_kv.py` 加 `build_trainable()` —— 只让 projector 进 autograd 图，
  teacher/student 的 KV 全部 detach 当常量。
- 新增 `train_projector.py`、`eval_projector_kl.py`。

### 失败记录

| 轮次 | 配置 | 结果 | 当时的判断 |
|---|---|---|---|
| v1 | 门控默认 | zero/random/pretrained 三者数值**完全相同** | 融合在推理时被关掉了 |
| v2 | 强制门控常开 | pretrained 42.4 > baseline 27.8 | 随机初始化起点太差？ |
| v3 | + `--zero-init` | pretrained 41.9 > baseline 26.4 | 训练在把 projector 推离恒等点 |

v1 的原因当时就找到了：`C2CProjector` 的 `gate_logit` 初始为 0，推理时硬门控
`(gate_logit > 0)` 为 False，融合被完全关闭，三个变体都退化成 student 自身。
修法是 `use_gumbel=False` + `gate_logit.fill_(3.0)`，训练和评测两侧都要设。

但 v2/v3 一直没解决。训练 loss 的 EMA 明明在降（17~22），held-out 却比 baseline 更差。

### 根因：三个 bug 层层套娃

回头系统查代码才发现，是三个互相掩盖的静默 bug，每修好一个下一个才暴露：

**Bug A — `--zero-init` 从未生效。**
`train_projector.py` 里 `FusedKVConfig(zero_init=False)` 硬编码，把命令行参数覆盖了。
所以 v3 的"零初始化"实际还是随机初始化，跟 v2 是同一个实验。

**Bug B — KL 的量纲错了（这才是训练失败的主因）。**
```python
F.kl_div(..., reduction="batchmean")   # 除数是 input.size(0)
```
传进去的是 `(1, R, V)`，`size(0)` = 1，所以除以 1 —— 得到的是整段 response 的
**KL 之和**，不是平均。R ≈ 71，梯度被放大约 71 倍，配 lr=2e-4 等效学习率约 1.5e-2，
projector 第一步就被轰离恒等点，之后再也没回来。

这也解释了历史数据那个诡异的量纲：`0.3713 nats/token × 71 ≈ 26.4`，
之前记录的"baseline = 26.4"其实就是同一个数的求和形式。

修法：抽出 `token_mean_kl()`，先 reshape 成 `(B*R, V)` 再算，训练与验收共用同一个
函数，避免口径漂移。

**Bug C — 训练成果从未落盘。**
`rosetta` 的 `save_projector` 只写 `__init__` 参数的 JSON（349 字节），
**不存 `state_dict`**；`load_projector` 是拿这些参数 `cls(**init_args)` 重新构造一个
新实例。训练时 `zero_init=True`，加载回来就是个全零 projector。

这是最阴的一个——它让 v4（Bug A/B 已修，训练其实成功了）的验收依然 FAIL。
修法：`fused_kv.py` 加 `save_projector_ckpt` / `load_projector_ckpt`，额外存权重。

### 顺带修的方法论问题

- 训练集与验收集重叠。改为按文件顺序切前 200 条作 holdout，训练不碰。
- 训练中每 N 步在 holdout 上评估，直接看是否优于 baseline，不再依赖 training loss。
- 超参对齐 C2C recipe：lr 1e-4、grad_accum 8、weight_decay 0.01
  （原来 lr 2e-4 + grad_accum 1，在 Bug B 的加持下更容易发散）。

### v5 结果 ✅

400 步，Qwen3-4B → Qwen3-1.7B，DAPO-Math-17k，max-len 160、prefix-ratio 0.5。

holdout 曲线（16 条，per-token KL）：
```
baseline 0.3713 → step100 0.1998 → step200 0.1922 → step300 0.1942 → step400 0.1905
```

最终验收（50 条 held-out，`eval_projector_kl.py`）：

| 设置 | per-token KL | 说明 |
|---|---|---|
| baseline | 0.4030 | student 自身，无 teacher 信息 |
| zero | 0.4070 | 融合恒等 → 印证零初始化确实等价于 baseline |
| random | 0.6888 | 未训练 projector → 印证融合通路真的接通 |
| **pretrained** | **0.1812** | **比 baseline 低 55.0%，比 random 低 73.7%** |

四个数字互相自洽，PASS。提交 `d11d15f`。

> `zero` 与 `baseline` 差 1% 而非严格相等，是 bf16 数值噪声：baseline 走单次全序列
> 前向，fused 走「前缀前向 + 带 cache 第二次前向」两段式，matmul 分块方式不同。

---

## 阶段三：下游验收 —— 数学正确率（进行中）

**为什么还要做这个**：阶段二的评测指标 *就是训练目标本身*，它只能证明优化成功了，
不能证明这件事有用。projector 完全可能只学到 teacher 的说话风格/置信度校准，
把 KL 拉低却没搬运任何解题知识。真正的硬证据是下游任务指标。

**改动**：新增 `eval_math_acc.py`。GSM8K-COT 300 题（带 label），greedy 解码，
答案抽取按 `\boxed{}` → `Answer:` → 最后一个数字 的优先级，数值归一化后比对。

### 两处会污染结论的问题（已修）

**harness 自检口径。** 最初只比「zero 融合 vs 原生 generate」，出现 1/3 分歧时
无法判断是融合有 bug 还是数值噪声。改成三路：

- A = 原生 `generate`
- B = 手写 decode loop + student 自身 KV（不过 projector）
- C = 手写 decode loop + zero-init 融合 KV（过 projector，但投影输出恒为 0）

`B == C` 必须严格成立（唯一差别是 projector，零初始化下是恒等），否则融合有 bug；
`A == B` 只反映 decode 路径的 bf16 差异，不影响结论。

实测：**B==C 5/5 ✅**，A==B 3/5。融合机制无问题，分歧确系数值噪声。

**两臂 decode 路径不一致。** 原本 baseline 走原生 `generate`、pretrained 走手写
decode loop —— 上面那个 bf16 噪声会直接混进 accuracy 对比。已改为 baseline 也走
手写 decode loop（用自身 KV），两臂唯一差别就只剩 projector。

### 结果 ❌ KL 降了 55%，正确率反而掉 6pp

300 题 GSM8K，harness 自检 B==C 全通过（融合机制无误）：

| 设置 | 正确率 |
|---|---|
| baseline | 86.0% |
| **pretrained (v5)** | **80.0%** |

**这是本项目到目前为止最重要的一个结果**：它直接证伪了"KL 低就等于知识迁移成功"。
翻看逐条生成，pretrained 的输出比 baseline 更流畅、更像 teacher 的语气，但**读错题**
的比例明显上升 —— projector 学到的是 teacher 的说话风格，不是解题能力。

阶段二把训练目标定成 KL、又拿 KL 当验收指标，属于自我印证。下游正确率是唯一可信的判据。

---

## 阶段四：推翻阶段二，改回 C2C 原版 SFT 口径

带着"到底哪里跟 C2C 不一样"这个问题逐行对读原码，找到 4 处偏离，
按影响从大到小：

### 差异 1（主因）—— 训练任务根本不对

C2C 的 `ChatDataset.__getitem__`（`dataset_adapters.py:1503-1546`）：

```python
labels = [-100] * len(instruction_tokens) + full_tokens[len(instruction_tokens):]
kv_cache_index = generate_kv_cache_index(len(instruction_tokens), len(full_tokens))
```

- `labels`：instruction 全 -100，**loss 只算在答案 token 上** → 学的是"把题做对"
- `kv_cache_index`：instruction 标 `[1,0]`（融合），response 标 `[-1,0]`（**不融合**）

而 v5 是把任意一段文本从中间劈开、后半段对齐 teacher logits。这个信号里既没有
题目/答案的边界，也没有"什么是正确答案"。**projector 只可能学到风格**，
正确率下降完全说得通。

### 差异 2 —— projector 数量

`SFT_train.py:618`：`num_projectors = slm_num_layers`，`projector_idx = target_layer_idx`。
**每个 student 层一个独立 projector**（28 个）。v5 是 28 层共用一个 MLP —— 第 0 层和
第 27 层的 KV 几何完全不同，共用只能学到"所有层的平均妥协"。

### 差异 3 —— 门控被焊死

v5 为了绕开"gate_logit=0 导致融合静默全关"这个坑，直接 `fill_(3.0)` + `use_gumbel=False`。
坑是绕过去了，但也**剥夺了 projector 拒绝融合的能力**，变成无差别灌入
（用户的判断"把不该融合的融合了"正指此处）。C2C 的 gate 是训练出来的，
配 per-layer 就是 28 组独立的层级开关。

### 差异 4 —— 训练/推理融合边界差一格

`evaluate.py:839-840`：`instruction_index = [1,0]*(L-1)`，`response_index = [[-1,0]]`
—— **最后一个 token 不融合**。评测侧 `crop_cache(cache, L-1)` 是对的，但训练侧
`build_trainable` 融合了全部 P 个位置，两边错位一格。

### 改动

**数据**（新增 `gen_teacher_traj.py`）：让 teacher 把 GSM8K train split 做一遍，
**只保留答对的轨迹** —— 这就是"teacher 关于解题的那部分知识"的文本载体。

- 数据源从 DAPO 换成 GSM8K train：DAPO 是竞赛级，teacher 只做对 18.8%，
  且题型与评测集差距大；GSM8K train 保留率约 90%，与评测的 300 题（全部 test split）**零重叠**
- 题面模板与评测**逐字节一致**。评测 jsonl 里的模板有两个数据生成期留下的 bug
  必须原样复刻：`{{}}` 双花括号（所以不能用 `.format()`，会折叠成 `{}`）、
  以及 `\boxed` 的 `\b` 在 JSON 里是**退格符转义**、`json.loads` 后真变成 0x08 字符。
  已验证 300/300 逐字节匹配

**`fused_kv.py`**：
- `per_layer_projector=True` → 28 个独立 projector；hidden 降到 512 控制总参数（176.6M）
- `keep_last_token_unfused=True` → 训练/推理都只融合前 L-1 位
- `build_trainable` 返回长度 P-1 的 cache，调用方从 x_{P-1} 开始喂，
  与 rollout 的 `crop_cache(cache, L-1)` + decode 首步逐位同构
- `need_teacher_logits=False`：SFT 口径下 teacher **只看前缀**，
  不让它提前读到答案（信息泄漏）
- `save/load_projector_ckpt` 支持 ModuleList

**`train_projector.py`** 整个重写为 SFT-CE 口径，训练/holdout 指标改成
response 段的 CE 与 token 准确率。

### 又两个静默坑（冒烟阶段抓到）

**门控在 bf16 下学不动。** `gate_logit` 是个**标量**，bf16 只有 8 位尾数，
1.0 附近 ulp ≈ 0.0078，而优化器每步给它的更新在 1e-3 量级 —— `param += update`
直接被舍入回原值。日志里 30 步后 logit 精确停在 `1.0000`，看着像梯度为零，
其实是精度吞了更新。修法：`gate_params_to_fp32()`，另外把门控单独分组、
lr ×20、不加 weight_decay（decay 会把它往 0 拽，正好是硬门控的翻转点）。

**`next(projectors.parameters())` 抓错 dtype。** 门控提到 fp32 后，
用它取 dtype 会把 KV 转成 fp32，撞 `mat1 and mat2 must have the same dtype`。
改为显式读 `key_in.weight`。

**gate 初值的取舍。** C2C 原版是 0.0，但那配的是 1929 步退火 + 50 万条数据。
本 recipe 下实测 30 步后 logit 只挪了 ±0.0003、符号纯属噪声 —— 而推理走硬门控
`logit>0`，等于**抛硬币决定每层要不要融合**，比全开更糟。故默认 +1.0：
起步全开、且仍然可学（与 v5 的 `fill_(3.0)`+`use_gumbel=False` 焊死不训有本质区别）。
holdout 同时报"推理口径（硬门控）"与"门控强制常开"两个数，把门控决策与投影质量拆开看。

### 冒烟（500 条数据 / 30 步）

```
baseline response CE 0.2087 / token acc 0.935
step30   推理口径 CE 0.1634 (+21.7%) / acc 0.938
gate     1.0000 → 1.0019，各层已分化（fp32 修复前恒为 1.0000）
```

600 步正式训练已跑完（1041 条轨迹，600 步，batch 8，lr 1e-4，per-layer 28 projector，
gate_init 1.0，anneal 600，attn sdpa）。结果见阶段五。

---

## 阶段五：v6 下游正确率 + 逐条归因

**为什么做这个**：阶段四对齐了 C2C 的 SFT-CE 口径，但 holdout CE 的曲线已经预警会过拟合
（见下）。真正的判据还是 300 题 GSM8K 下游正确率。本阶段把 step200 / step400 两个
checkpoint 接进 `eval_math_acc.py` 测，并对差异样本做了逐条归因。

### 1) holdout CE 曲线（来自 `logs/train_v6.log`，这是过拟合的最早信号）

| step | fused CE | token acc | base CE（不融合） | 相对 base |
|---|---|---|---|---|
| 100 | **0.1618** | 0.941 | 0.2017 | **−19.8%** ✅ 最优 |
| 200 | 0.1651 | 0.942 | 0.2017 | −18.1% |
| 300 | 0.1801 | 0.941 | 0.2017 | −10.7% |
| 400 | 0.1864 | 0.938 | 0.2017 | −7.6% |

「推理口径 CE」与「门控强制常开 CE」在各步完全相同（gate 早已常开），故只有一条曲线。
**step100 最低，之后单调上升 → 典型过拟合**：projector 在 1041 条上把训练 loss 越压越低，
但 holdout（没见过）上的融合质量开始退化。这直接预言了下面下游正确率的崩。

> 注意：holdout CE 仍是**训练目标在未见数据上的代理**，不是最终答案。它只能用来
> 选 checkpoint，不能代替 300 题评测下结论（与阶段三 v5「KL 降了正确率反而掉」同构）。

### 2) 下游正确率（300 题 GSM8K，harness 三路自检 B==C 全过）

- **step400**（最先评，后为了让出 GPU 给 step200 被主动杀掉，仅留 19 题部分结果）：
  baseline 89.5% / fused **73.7%** / **Δ −15.8pp**（0 赚 3 丢）。
- **step200**（运行中的实时值，截至 46/300，watcher 跑满会自动出最终数并重启 step400）：
  baseline 89.1% / fused **82.6%** / **Δ −6.5pp**（2 赚 5 丢，另有 3 题两边都错）。

**核心结论**：step200 远好于 step400，方向完全印证 holdout 曲线（越靠近最优 step100 越不伤）。
但即便 best-so-far 的 step200 仍是**净负**——融合在「学生已答错」时能纠错，但也在
「学生本就对」时把对的带歪，且代价目前大于收益。

### 3) 逐条归因（46 题里丢/赚的样本，这是本阶段最有价值的发现）

**增益侧——融合真的搬来了 teacher 的「读题/列式」能力（优势是真实的）**
- **idx30**（label 42）：baseline 把「糖:水=7:13、总共 120」误读成「120 茶匙糖」→ `7x=120`→120（错）；
  fused 正确建成 `x+y=120, x/y=7/13`→`x=42`（对）。**teacher 的方程列式能力被迁移过来。**
- **idx15**（label 5）：baseline 算成「去程3h+回程5h=总时间 8」（答非所问）；fused 给 5
  （回程时间，gold）。学生在子问题上跑偏时，融合把它拽向 teacher 答的那一侧。

**损失侧——分成两类，结论很关键**
- **A 类：推理链逐字相同，只被融合改坏了最后一步算术（最该警惕）**
  - idx1（label 694）：`204+160+330` baseline **694** ✓，fused 末尾写成 **700** ✗。
  - idx19（label 595）：`175+140+280` baseline **595** ✓，fused 末尾写成 **600** ✗。
  - 两套生成推理完全一致，仅末步求和翻车 → **证明融合的 KV 扰动强到能翻转一个近平局的
    最终加法**，它没改思路，只是把本来算对的结果搅错。
- **B 类：融合往本来对的题里注入了新的读题/列式错误**
  - idx18（label 36）：学生原算 `3h/天×3天/周×4周=36` ✓；fused 把「每周3次」丢了→`3h/周×4=12` ✗。
  - idx26（label 348）：学生原算 `兔=(狗+猫)−12=168` ✓；fused 写成自相矛盾的 `R=D+12` ✗。
  - idx42（label 15）：学生原算 `少赚105−花费90=多15` ✓；fused 把 `105+90=195` 当净损失、
    最后却说「请会计多 90」✗（比较逻辑整个搅乱）。

**对「当前策略优势」的判断**
1. 优势成立：teacher 的**读题 + 建立方程**能力确实能通过 KV 融合迁移给 student（idx15/30
   是铁证），且恰好补在 student 弱点上——这正是最初想要的「搬运解题经验」，不是搬语气。
2. 当前净负的根因是**转向太猛 + 过拟合**，不是方向错：融合像个强力带噪声的转向器，
   学生跑偏时能拉回（增益），学生本对时也强行扰动（A 类把正确算术带歪、B 类注入新错）。
   A 类证据直接说明**融合强度过大**。

---

## 阶段六：下一步——只保留纠偏效益、避免过拟合

目标（用户原话）：进一步改正 C2C 的融入，使得**只保留纠偏的效益**，并且**避免过拟合**
（如每 20 步记录一次，争取选到最适合的 checkpoint）。

### 已落地的代码改动（本阶段）
- **每 20 步记录 + 存盘**：`train_projector.py` 的 `--eval-every` / `--save-every` 默认值
  由 100 / 200 改为 **20**。这样 holdout 曲线粒度从 100 步细化到 20 步，且每 20 步都存一个
  projector 权重，能直接挑出最优步数（而不是只有 200/400 两个粗糙候选）。
- **融合强度系数 `fusion_scale`**（核心新旋钮）：`FusedKVConfig` 加 `fusion_scale`，
  `build()` / `build_trainable()` 改为
  `fused = student + fusion_scale * (proj_out − student)`，默认 1.0 与原版完全等价。
  `train_projector.py`、`eval_math_acc.py` 均暴露 `--fusion-scale`。
  **关键性质**：该系数在**评测侧直接扫即可，无需重训**——因为 `(proj_out − student)` 只依赖
  已训好的 projector 输出。可据此找「纠偏效益最大、把对的带歪最小」的 scale。

### 验证路线（建议先做、零重训成本）
1. **扫 `fusion_scale` 于现有 step200 / step400 权重**：预期低 scale（~0.3–0.5）能消除 A 类
   算术翻车（idx1/19），同时大概率保住 idx15/30 类增益。这一刀若见效，就证明「强度过大」
   是唯一主因，纠偏收益本身是可保留的。
2. 选定 scale 后，再扫 step200 附近每 20 步的 checkpoint，用 300 题评测挑 Δ 最优者。

### 抗过拟合（重训时做）
- **扩数据**：GSM8K train 有 7473 题，当前只用 ~1.1k；量级差一个数量级是过拟合主因。
- **早停 + 正则**：现在已有 20 步粒度记录，可据 holdout CE 曲线在最低点附近早停；
  必要时加 dropout / 权重衰减 / projector 容量下探。
- **（未来真正的解法）置信度门控融合**：只在 student 自身置信度低时开融合、置信度高时
  不扰动——这才是从机制上实现「只保留纠偏效益」。当前 `fusion_scale` 是全局近似，
  置信度门控是更精准的下一步。

---

## 关于与 C2C 原始配方的差异（阶段四后）

| 项 | C2C 原版 | 本项目 | 状态 |
|---|---|---|---|
| 损失 | SFT 交叉熵，只算 response | 同 | ✅ 阶段四对齐 |
| projector 数量 | 每层一个（28） | 同 | ✅ 阶段四对齐 |
| 门控 | 可学习 + Gumbel 退火 | 同，但初值 +1.0 而非 0.0 | ⚠️ 见上文取舍 |
| 融合边界 | 只融合 instruction 的前 L-1 位 | 同 | ✅ 阶段四对齐 |
| 数据 | 通用对话 500k | GSM8K teacher 正确轨迹 ~1.1k | ⚠️ 对齐数学任务，量级差很多 |
| 规模 | 1 epoch × 500k | 600 步 × batch 8 | ⚠️ 先验证机制 |

---

## 待办

- [x] v6（SFT 口径）600 步训完 → 300 题 GSM8K 评测（step200/step400 均净负，见阶段五）
- [x] 逐条 diff 归因：已确认融合能搬 teacher 的**题面理解/列式**（增益侧），
      但也会把本来对的题带歪（A 类算术翻车 + B 类注入新错）
- [ ] **（优先）扫 `fusion_scale` 于现有 step200/step400 权重**：评测侧直接扫，零重训，
      验证低 scale 能否消除 A 类翻车、保留增益（见阶段六验证路线）
- [ ] 选定 scale 后，扫 step200 附近每 20 步 ckpt，挑 Δ 最优者
- [ ] 扩数据：GSM8K train 有 7473 题，当前只用了 1.1k（过拟合主因）
- [ ] 重训：全量数据 + 20 步记录/存盘 + 据 holdout 曲线早停
- [ ] （未来机制解法）置信度门控融合：只在 student 不确定时融合，实现「只保留纠偏效益」
- [ ] Part II：teacher logits 接入 `core_algos.compute_policy_loss_on_policy_distill`
      的 token 重要性加权
- [ ] 端到端 EOPD 训练跑通（需完整 verl 环境 + ray）
- [ ] 路线 B：vLLM/SGLang server 内注入 fused KV（规模化）

---

## 复现命令

### 复现流程详解：projector 训练到底在做什么

【C2C 核心】一句话：**只训练 `C2CProjector`（28 个 per-layer 小 MLP + 56 个门控标量），
teacher 与 student 全程冻结，权重不动**。训好即插即用，与 student 不是端到端一起训的。
训练/评测代码：`cache_eopd/train_projector.py` + `cache_eopd/fused_kv.py` + `rosetta/model/projector.py`。

#### 1. 训练什么、用什么数据
- 可训参数只有 projector（`train_projector.py:243-258`：优化器只收 `body_params`+`gate_params`；
  `fused_kv.py:411-416`：`freeze_teacher_student()` 把 teacher/student 全 `requires_grad_(False)`）。
- 数据 = `gen_teacher_traj.py` 产出的 jsonl，每条 `{prompt: 题面, solution: teacher 答对的完整解题轨迹}`。
  按文件顺序切（`train_projector.py:209`）：
  ```
  hold_rows  = rows[:64]      # 前 64 条 = holdout，训练绝不使用
  train_rows = rows[64:]      # 后 ~1041 条 = 训练集
  ```
- 每条样本拼成 `(input_ids, prefix_len)`（`build_samples` `train_projector.py:133-163`）：
  instruction = chat-template（带 generation prompt）作前缀（**融合 + 不计 loss**），
  response = solution token + eos 作后缀（**不融合 + 计 loss**），严格对齐 C2C 的 `kv_cache_index`。

#### 2. 一次训练迭代的内部机制（核心数学）
单步流程：`response_ce`（`train_projector.py:270-294`）→ `build_trainable`（`fused_kv.py:310-400`）。
- **① 两段前向（都冻结、no_grad）**：teacher 只看 instruction 前缀 → teacher KV；
  student 只看 instruction 前缀 → student KV（`fused_kv.py:352-377`）。
- **② 逐层投影融合**（`projector.py:1277-1282`、`:1093-1098`）：
  ```
  对 student 第 s 层：
    projected = MLP( 把 teacher_KV 投影进 student 的 head/head_dim 空间 )   # 处理 GQA 头数/维度差
    w_head    = sigmoid(scalar_weight)            # 每头、输入相关的混合权重 ∈ [0,1]
    gate      = gumbel_sigmoid(gate_logit)        # 每层标量，退火后≈硬开/关
    proj_out  = student_KV + gate * w_head * projected
  ```
  再套评测期 `fusion_scale`（`fused_kv.py:394-397`）：
  ```
  fused = student + fusion_scale * (proj_out - student)
        = student + fusion_scale * gate * w_head * projected
  ```
  【C2C 对齐 2】只融合前 L-1 位，末位 `x_{P-1}` 不融合、留给 decode 首步用 student 自身 KV 重算
  （`keep_last_token_unfused=True`，`fused_kv.py:383/402-406`），与 rollout 推理逐位对齐。
- **③ 算 loss**：把 fused 前缀当 `past_key_values`，teacher-forcing 让 student 生成 response，
  只算 response 位的交叉熵，目标 = teacher 写出的 gold token（`train_projector.py:285-294`）：
  ```
  L = mean_{t∈response} CE( student_logits[t], gold_token[t] )
  ```
  这就是 v6 目标「前缀塞进 teacher 的 KV 后，student 能不能把这题做对」，推翻了 v1 的 KL 蒸馏
  （v1 学的是风格不是解题，GSM8K 掉 6pp，见 `train_projector.py:15-19`）。

#### 3. 反向传播路径（为什么叫"提前单训"）
```
loss → student_response_logits → fused_prefix_KV → projector
```
teacher/student 全程 `detach`，梯度**只流回 projector**（`fused_kv.py:388-392`）。student 的 1.7B
权重只是"被借用前向"的常量容器，永不更新 —— 即"提前、单独把 projector 训好，再即插即用"。

#### 4. 工程细节（几个关键坑）
| 项 | 设置 | 为什么（`train_projector.py` / `fused_kv.py`） |
|---|---|---|
| `zero_init=True` | 投影输出层置 0 | 起步 `proj_out=student`（恒等），训练只能**改善**不能搞坏 |
| `gate_init=+1.0` | 门控初值正数 | C2C 默认 0.0 → 推理硬门控 `(0>0)=False` 会**静默全关**；+1.0 起步融合开着且仍可学 |
| `gate_lr_mult=20` + fp32 标量 | 门控单独高 lr | bf16 标量大不过更新量，30 步纹丝不动，改 fp32 才学得动（`fused_kv.py:139-151`） |
| `grad_accum=8` | 每 8 样本 1 步 | 显存限制下的有效 batch |
| Gumbel 退火 | T: 1.0→0.001 | 训练末期门控趋近硬开/关，与推理一致（`fused_kv.py:165-168`） |
| `lr=1e-4`, warmup 30 | 对齐 C2C recipe | — |

#### 5. 训练中的监控（据此判断过拟合）
每 `eval_every=20` 步在 holdout（64 条未训练样本）上记两组数（`train_projector.py:387-399`）：
- **推理口径 CE**：走真实硬门控（= `eval_math_acc` 会看到的）
- **门控常开 CE**：临时把门控拨到 3.0，把"门控决策"和"投影质量"拆开看
- 基准 `base CE`（纯 student、无融合）≈ 0.2017 恒定

观察到的 holdout 曲线（阶段五的最早过拟合信号）：
```
step100  0.1618  ← 最佳
step200  0.1651
step300  0.1801
step400  0.1864   ← 明显过拟合 1041 条训练集
```
另：`gate_on` 行（`:374-379`）实时显示"推理时真正会融合的层数"——若一路停在 0，说明训完融合被全关、
pretrained 会退化成 baseline。

#### 6. 产出与下游
每 `save_every=20` 步落盘 `projector_step{N}.pt`（+ `.weights`）。下游评测（`eval_math_acc.py`）加载
**冻结 student + 冻结 teacher + 训好的 projector**，对 GSM8K 每题同时跑 `baseline`（无融合）和
`pretrained`（融合）两路比准确率 —— 即阶段五 step200 净负 6.5pp、归因出 GAIN/LOSS-A/LOSS-B 的来源。

#### 7. 数据流一图总览
```
teacher(冻) ──prefix──► teacher_KV ┐
                                   ├─► C2CProjector(唯独可训) ─► fused_prefix_KV
student(冻) ─prefix──► student_KV ┘                                        │
                                                                        ▼
                          student(冻, teacher-forcing) ──► response logits ──► CE(gold)
                                                                        │
                                                          ▓▓ 梯度只回 projector ▓▓
```

### 概念：为什么"融合 KV cache"能训出一个独立权重

【C2C 核心】先掰正一个认知：**融合出来的 KV cache 本身不是权重**；独立权重 = `C2CProjector`
（28 个 per-layer MLP + 56 门控标量），它是架构上就独立的一个小网络。KV cache 是每次前向
临时算出的张量（存进 `DynamicCache`，用完即弃，从不是可学习参数）；projector 才是被学出来的那坨。

#### 1. projector 从一开始就是独立的
`from_models`（`fused_kv.py:198-216`）给每个 student 层新建 `C2CProjector` 组成 `nn.ModuleList`，
**既不注册进 teacher 也不注册进 student**（`fused_kv.py:104-106` 只持引用）；优化器里**只有它**
（`train_projector.py:243-258`）。

#### 2. 融合 KV 是一条可微管道，梯度借它流回 projector
`build_trainable`（`fused_kv.py:310-400`）里 teacher/student 的 KV 都 `.detach()` 当常量，唯有
`projector(...)` 的输出留在 autograd 图里，再喂给冻结 student 算 logits、CE，最后
`(ce/accum).backward()`（`train_projector.py:350`）。梯度**穿过**冻结 student（前向可微）一路回传到
projector，但 student 不在优化器里 → 梯度到不了也不更新它。这便是"独立"的来源。

#### 3. 抽象成标准监督学习
| 要素 | 在这里 |
|---|---|
| 可学习 θ | projector 权重（独立） |
| 输入 x | (teacher_KV, student_KV) 冻结常量对 |
| 固定函数 f_θ | projector→融合→**冻结 student 前向** |
| 目标 y | teacher 的 gold answer token |
| loss | response 交叉熵 |

唯一特别：**固定函数里包着 1.7B 冻结大模型**，它当"评分器"，把 projector 的输出（一个 KV cache）
翻译成"student 能否答对"的信号再反传回 projector。

#### 4. 类比：KV-cache 版 prefix-tuning
学一组虚拟 token、冻结 LM、梯度从输出反传到那组 token（prefix-tuning）。这里同构——只是学的不是
输入 token，而是**注意力层的 KV cache**（由 teacher KV 经 projector 映射）。小模块 + 冻结大模型 +
梯度穿过大模型回到小模块，三者一致。

#### 5. 为什么偏偏选 KV cache 而不是改权重
KV cache 是模型解码时真正"读"的上下文/记忆；把 teacher 知识注入 student 的 KV cache = 在 student
解题时喂给它 teacher 的"思路上下文"，而完全不碰 student 权重。projector 学的是 **teacher 表征空间 →
student 表征空间的映射桥**：把 teacher 的 KV 投影成 student 能看懂的 KV。这比动 student 权重更便宜、
可逆、即插即用（也是 `fusion_scale` 能当推理期旋钮的根因）。

```bash
cd /home/kejiechen/CacheEOPD
PY=/home/knhdu/anaconda3/envs/rosetta/bin/python
M=/home/kejiechen/taopd-baseline/modelweights

# 1) 造 teacher 解题轨迹（GSM8K train split，只留答对的）
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1,2 $PY -m cache_eopd.gen_teacher_traj \
    --teacher $M/Qwen3-4B --source gsm8k \
    --num-problems 1500 --batch-size 16 --max-new-tokens 512 \
    --out ./data/teacher_traj_gsm8k1500.jsonl

# 2) projector 预训练（C2C SFT 口径，per-layer + 可学门控）
#    --teacher-gpus 必须显式指定空闲卡，否则 auto 会挑到被别人占满的卡而 OOM
#    --attn-impl sdpa：训练要反传过 student，eager 保留 L×L 注意力矩阵会爆显存
#    --eval-every / --save-every 20（阶段六）：每 20 步记录 holdout 并存盘，便于挑最优步数
#    --fusion-scale 1.0 与原版等价；调小可削弱融合（评测侧也能直接扫，无需重训）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. $PY \
  -m cache_eopd.train_projector \
    --teacher $M/Qwen3-4B --student $M/Qwen3-1.7B \
    --data-path ./data/train_v6.jsonl \
    --device cuda:3 --teacher-device auto --teacher-gpus 4,5 --attn-impl sdpa \
    --steps 600 --grad-accum 8 --lr 1e-4 --anneal-steps 600 --gate-init 1.0 \
    --eval-every 20 --save-every 20 --holdout 64 --fusion-scale 1.0 \
    --out-dir ./ckpt_projector_v6

# 3) 数学正确率验收（含 harness 三路自检；这才是唯一可信的判据）
#    --fusion-scale 可在不重训的前提下直接扫不同融合强度（阶段六验证路线）
PYTHONPATH=. $PY -m cache_eopd.eval_math_acc \
    --teacher $M/Qwen3-4B --student $M/Qwen3-1.7B \
    --data-path /home/kejiechen/taopd-baseline/data/GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl \
    --device cuda:3 --teacher-device auto \
    --num-samples 300 --max-new-tokens 512 --sanity 5 \
    --fusion-scale 1.0 \
    --projector-path ./ckpt_projector_v6/projector_step200.pt --out ./gsm8k_math_v6_step200.jsonl
```

checkpoint：apex-llm `~/CacheEOPD/ckpt_projector_v6/projector_step{N}.pt`（+ `.weights`，
每 20 步一个，N=20,40,...,600；另有 `projector_final.pt`）。
v5 的 ckpt 已作废（训练口径错误，见阶段四）。

---

## 阶段七：轻量蒸馏可行性（4B→0.6B 官方 projector）

【EOPD 核心】目标 = 验证「学生 rollout 注入 teacher KV（C2C）做蒸馏训练」是否让模型变好。
这是与阶段二/六**梯度目标相反**的实验：冻结 teacher + 官方 projector，**只训 student**。

- 配对刻意用官方对 **Qwen3-0.6B(student) + Qwen3-4B(teacher)** + 官方 fuser
  `nics-efc/C2C_Fuser/qwen3_0.6b+qwen3_4b_base_Fuser`（即 wrap4api.py 同款），
  **省去 projector 自训**，直接回答核心问题。
- 新增脚本 `cache_eopd/train_student_distill.py`：
  - `--mode fused`：训时前缀用官方 projector 融 teacher KV（w/ C2C）
  - `--mode plain`：训时前缀用学生自身 KV（w/o C2C 基线）
  - 官方 fuser 经 `rosetta.utils.evaluate.load_rosetta_model` 加载，权重 `load_state_dict`
    搬进 `FusedKVBuilder`（同属 vendored C2CProjector）。
  - 优化器只收 `student.parameters()`；teacher + projector 全冻结。
- 评测：两学生各自用 `eval_math_acc.py` 的 **baseline 臂**（不加载 projector）评 plain 准确率，
  比 (fused 学生 plain) vs (plain 学生 plain) = C2C 训练是否让学生本身变好。
- 数据前提：`gen_teacher_traj.py` 用 Qwen3-4B 在 GSM8K 产出 teacher 轨迹 jsonl（prompt/solution）。
- 待办：① 取官方 fuser + Qwen3-0.6B 到运行机；② 生成 4B teacher 轨迹；
  ③ 跑 fused / plain 两版训练；④ baseline 臂评测对比。

### 可行性测试已跑通（apex-llm，2026-07-31）

目标只是确认「官方 fuser + 学生自训」这套管道在 apex 上能跑、不 OOM、两臂都能训。结论：**跑通**。

**资源就位**（apex `/home/kejiechen/taopd-baseline/modelweights`）：
- 下载 `Qwen3-0.6B`、`qwen3_0.6b+qwen3_4b_base_Fuser/final`（28 个 projector，hidden=1024）。
- apex 直连 huggingface.co 超时，改走 `HF_ENDPOINT=https://hf-mirror.com` 的
  `huggingface_hub.snapshot_download` 下载。teacher 轨迹复用现成的
  `data/teacher_traj_gsm8k1500.jsonl`（4B 轨迹 1368 条，字段 prompt/solution）。

**三个必须修的 bug（否则管道跑不起来）**：
1. **`build_trainable` 的 value 行 `s_val` 未定义**（`fused_kv.py:403`）。`build()` 里变量叫
   `s_value`，`build_trainable` 里融合公式误写成 `s_val` → `NameError`。修为 `s_value`。
   （这是 阶段六 加 fusion_scale 时引入的未测笔误，之前 stage 4-6 走的是另一条路径没触发。）
2. **plain 臂前缀 cache 越界一格**：原 `x[:, :P]` 建 cache（长 P），却从 `c=P-1` 喂新 token，
   把 `x_{P-1}` 既存 cache 又当新 token → SDPA 报 `expanded size 227 vs 226`。
   对齐 fused 臂的 `keep_last_token_unfused`，plain 也只取 `x[:, :P-1]` 建 cache。
3. **官方 fuser 加载方式 OOM**：原 `load_rosetta_model` 会把 4B teacher 按
   `device_map={"":student.device}` 塞进 cuda:1（仅 ~7GB）→ OOM。改为**直接**从
   `final/projector_{idx}.pt+.json` 读权重，并**整体替换** `builder.projectors`
   （官方 projector 的 MLP hidden=1024 ≠ 本项目默认 512，复制权重会被 strict=False 静默丢弃）。
   顺带确认两模型 `head_dim=128`（config 显式字段），与 fuser 的 128/128/8/8 完全吻合。

**极简冒烟（steps=10, grad_accum=1, 8 holdout, fusion_scale=1.0, 单卡 cuda:1 + 4B 跨 4,5）**：

| 臂 | step10 训练 CE | holdout plain CE (step5→10) |
|---|---|---|
| **fused**（w/ C2C） | 0.3453 | 0.2367 → 0.2360 |
| **plain**（w/o C2C） | 0.1308 | 0.2351 → 0.2308 |

两臂都正常训、CE 有限、都存档 `ckpt_student_{fused,plain}_smoke`。fused 训练 CE 明显高于 plain
（fused 前缀是 teacher 分布，学生要适应 → 难而有用），印证 `fused ≠ plain`、管道语义正确。
> 注意：10 步太小，holdout plain CE 两臂几乎一致属正常；要分胜负得看下游 300 题准确率。

**待办（真正的实验，非可行性）**：
- [x] ① 取官方 fuser + Qwen3-0.6B 到运行机（hf-mirror 下载，完成）
- [x] ② 4B teacher 轨迹（`teacher_traj_gsm8k1500.jsonl` 已存在，无需重生成）
- [ ] ③ **完整两版训练**（steps≈300、lr 1e-5、batch 8）—— 两臂必须**串行**（各占 4B+0.6B，
      并发会抢 GPU 5 的 4B 权重导致 OOM）
- [ ] ④ `eval_math_acc.py` baseline 臂分别评两学生 plain 准确率，比 (fused 学生) vs (plain 学生)

### 完整训练 OOM 排查与修复（2026-07-31 晚，apex-llm）

极简冒烟过后，**完整 300 步 fused 训练在首个 backward 就 OOM**，而且现象很诡异：
`GPU1 Tried to allocate 210 MiB / only 167 MiB free`，进程内 PyTorch 已用 **6.64 GiB**。
无论 `--max-prompt-len/answer-len` 取 256 还是 384，报错数字**逐字节相同** → 说明不是
序列长度（激活）问题，而是某个**固定开销**在作祟。这次 OOM 发生在 `grad_accum` 首轮
backward，**优化器状态还没分配**，所以换 CPU-offload AdamW 也救不了它。

**根因（排查过程，不是猜出来的）**：
- 分阶段打点测量（`/tmp/diag_train.py`、`/tmp/diag_loop.py`）：
  - 0.6B student + AdamW 固定占用 **1.11 GiB**（torch 2.6 AdamW 状态默认 bf16，很省）
  - 28 个官方 projector（hidden=1024）搬上 GPU1 后 **+~1.0 GiB**（2.10 总计）
  - 一个完整 fused step（build_trainable + decode fwd+bwd），跨 12 个样本（ans 170~385）
    实测 GPU1 **峰值最高仅 4.01 GiB** → 理论上 7 GiB 预算绰绰有余。
- 所以 6.64 的多出来 ~2.6 GiB 只可能来自 **teacher**。teacher 用 `device_map="auto"` +
  `max_memory={2:"7GiB",3:"7GiB",其余"1MiB"}` 加载时，auto 规划器在单卡 7GiB 预算下过于
  保守，把 4B 的 **5.4 GiB offload 到磁盘**（日志 `Some parameters are on the meta device...`），
  只留 2.8 GiB 在 GPU2/3。`build_trainable` 前向时，被 offload 的层按**当前设备**
  （torch 默认 = 学生所在的 cuda:1）加载 → 4B 权重临时压到 GPU1 → 6.64 OOM。
- 这也解释了「256/256 与 384/384 报错完全一致」：OOM 的是被临时搬到 GPU1 的 teacher 权重，
  跟学生序列长度无关。

**修复迭代（共三版，全部落地进 `train_student_distill.py`，未提交）**：

1. **teacher 显式 device_map 消除磁盘 offload**：`AutoConfig` 取层数后手写
   `model.layers.{i}` 到 GPU2/3 的映射（embed_tokens/norm/lm_head 也显式指定），
   4B 完整落进 GPU2+GPU3（各 4.11 GiB），**不再有 5.4 GiB 挂在磁盘**、teacher 前向
   也不再临时把权重搬上 GPU1。
   - **坑 1**：`lm_head` 必须与 `embed_tokens` 同卡（Qwen3 两者 tied），否则 accelerate 报
     `Tied parameters are on different devices` 直接崩在 teacher forward。
     第一版 lm_head 放 cuda:3 → 崩 → 改回 cuda:2 通过。
   - **坑 2**：第一版把层**轮询**分到 2,3（layer0→cuda2, layer1→cuda3...），36 层 = 36 次
     PCIe 跨卡 → teacher forward 极慢。改为**连续均分**（0-17→cuda2, 18-35→cuda3），
     只 1 次跨卡边界。
2. **CPU-offload AdamW 尝试（脚本内 `CPUAdamW` 类）**：m/v 存 CPU、step 时梯度搬 CPU
   算完搬回。能跑、不 OOM，但**慢得离谱**：实测 ~45-50 s/step（22160% CPU），
   300 步要 ~18h，不可接受 → 弃用（类保留在脚本里作注释参考）。
3. **最终组合（当前在跑）**：**projector 搬到 GPU4**（新增 `--proj-device cuda:4`，
   释放 GPU1 ~1 GiB）+ **GPU AdamW fused**（torch 2.6 支持 bf16）+ teacher 连续均分在
   GPU2/3 + 学生 384/384 全长。
   - GPU1 只剩 student(1.11) + 激活 + m/v(2.4) ≈ 峰值 5.7-6.8 GiB < 7.08 预算，**不 OOM**。
   - 速度 ~3 s/micro-batch（≈9× 快于 CPUAdamW）。

**验证**：最终组合 10 步 smoke（grad_accum=2, 384/384）**60 秒跑完全程**：
`step5 CE 0.516 → step10 CE 0.402`、`holdout plain CE 0.253`、正常存档 `ckpt_smoke_fused3`，无 OOM。

**当前 GPU 拓扑（apex，2026-07-31 晚）**：GPU0 被 vLLM 占满 48G；GPU1-5 各被 `knhdu` 占 ~41 GiB，
单卡实际可用仅 **~7 GiB**。所以 student 固定放 cuda:1，teacher 放 cuda:2,3，串行跑两臂。

**任务状态（实时，2026-08-01 00:5x）**：
- [x] ① 资源就位（fuser + Qwen3-0.6B + 4B teacher 轨迹）
- [x] ② 修 3 个管道 bug + OOM 根因（teacher 磁盘 offload）+ lm_head tied 坑 + CPUAdamW 太慢
- [x] ③ **fused 完整训练 300 步完成**：CE 0.43→0.20，holdout plain CE 0.3075，`ckpt_student_fused`
- [x] ④ **plain 完整训练 300 步完成**：CE 0.15–0.25，holdout plain CE 0.2103，`ckpt_student_plain`
- [x] ⑤ **fused / plain 下游 300 题评测完成**（base 0.6B 参照仍在跑，~1h）
      **坑**：`--student` 必须指向 ckpt **根目录**（训练脚本只有根目录存了 tokenizer；
      `student_step300` 子目录只有模型，`AutoTokenizer` 会报 vocab_file=None）

### 阶段七结果（300 题 GSM8K，baseline 臂 = 学生自身生成，2026-08-01）

| 学生 | 训练方式 | plain 准确率 | 平均生成长度 |
|---|---|---|---|
| **plain 学生** | 学生自身前缀 SFT（w/o C2C） | **63.0%**（189/300） | 293 |
| 原始 Qwen3-0.6B | 未训练（参照） | 61.3%（184/300） | 264 |
| **fused 学生** | teacher KV 融合前缀训练（w/ C2C） | 60.3%（181/300） | 268 |

**结论**：
1. **C2C 融合训练未帮助、反而有害**：plain − fused = **+2.7pp**；fused 甚至**低于未训练 base −1.0pp**
   （融合前缀 SFT 把学生教得比原版还差）。
2. **普通 SFT 本身有效**（对照成立）：plain 比 base 高 **+1.7pp**，训练管道无信号问题。
3. **适用范围**：仅此配置（300 步、lr 1e-5、grad_accum 8、官方 fuser、off-policy 教师轨迹 SFT）。
   与真正要跑的 on-policy EOPD 蒸馏（w/ vs w/o C2C rollout）是不同实验，**不构成对蒸馏假设的否决**。
   （过拟合诊断见下节：取 step50/100/200 早停点评测，证实 fused 随步数**特异过拟合**、且全点被 plain 压制，
   主因 train/test 失配而非蒸馏无效。）

**后续可选方向**：更长步数（~1-2k）/ 更小 lr / `--fusion-scale<1` 扫描；或直接做 2-phase EOPD 蒸馏的 w/ vs w/o C2C 对比。

### 阶段七(过拟合诊断)：取更早 checkpoint 评测（2026-08-01，已完成）

用户提出：融合 teacher KV 容易过拟合，要不要把 step100 / step200 的 fused 学生拿出来测，
看是否比 step300(60.3%) 更好。

**关键澄清**：阶段七的评测走 `eval_math_acc.py` 的 **baseline 臂**（不带 `--projector-path`），
所以它测的是「该学生 checkpoint 自身的泛化能力」，与训练时是否融合 teacher KV 无关。
因此：直接把 `student_step100` / `student_step200` 子目录当 `--student` 评即可，无需重做融合。

**已做**：
- 4 个 step 子目录（fused/plain × step100/step200）原本只有模型权重、无 tokenizer；
  已从各自根目录复制 tokenizer 文件进去（否则 `AutoTokenizer` 报 vocab_file=None）。
- 启动 `eval_early_ckpts.sh`（apex pid 1186224）：对上述 4 个 ckpt 各跑 300 题 standalone 评测，
  复用阶段七卡分配（学生 cuda:1，teacher auto 于 GPU2,3 磁盘 offload），写各自 jsonl+log。
- 后台 watcher 轮询 master log，4 个全跑完即通知。
- **step50 改为并行加速**（用户更想先看 step50）：原 KoBu3l 排队脚本已取消，改为在 **GPU4/5**
  起 `eval_step50_parallel.sh`（apex pid 1193743）与主链(GPU1)并行跑 `fused_step50`/`plain_step50`
  各 300 题（student cuda:4，teacher 分片 4,5）。目的：step50 约 2h 内出数，而非原队列末尾的 ~5h。
  `student_step50` tokenizer 已补。⚠️ GPU5 余量仅 ~0.9G（teacher 分片偏重），但 baseline 臂不前向
  teacher，内存静态，应无 OOM；若炸则降 `--teacher-mem-per-gpu` 重试。

**待收**：fused / plain 在 **step50 / step100 / step200 / step300(=60.3% / 63.0%)** 的对比
→ 判过拟合是否成立（fused 早停点显著前移、且 plain 不明显 = fused 特异性过拟合；
若 step50>step100>step200>step300 则最优点在 <50，需再补更早 ckpt）。

**实时结果(进行中, 2026-08-01 ~12:30)**：

| 学生 | step50 | step100 | step200 | step300(已知) |
|---|---|---|---|---|
| **fused** | **61.0%**（183/300） | **62.0%**（186/300） | **60.3%**（181/300） | 60.3%（181/300） |
| **plain** | **62.3%**（187/300） | **63.0%**（189/300） | **62.7%**（188/300） | 63.0%（189/300） |

→ fused 曲线 **step50(61.0%) → step100(62.0%,峰) → step200(60.3%) → step300(60.3%)**：
**先欠拟合、step100 到顶、之后过拟合平台**。最优在 step100，**不是更早的 step50**。
→ plain 曲线 **step50(62.3%) → step100(63.0%) → step200(62.7%) → step300(63.0%)**：微升后平台，**完全不退化**
   （与 fused 的 step100→300 塌方成鲜明对照）。
→ **过拟合是 fused 特异的**：每一 checkpoint fused 都低于 plain
   （step50: 61.0 vs 62.3；step100: 62.0 vs 63.0；step300: 60.3 vs 63.0），
   且 fused 随步数退化而 plain 不退化。用户原假说"融合 kv 容易过拟合"✅ 证实，确定为 fused-only。
→ 但 fused 即便最优 step100(62.0) 仍 < plain 任意点(63.0)：说明 fused 跑不赢 plain **不只是过拟合**
   （纯过拟合应有"早停点≈plain"），而是 **训练/测试失配（KV 拐杖考试不在）为主因 + 过拟合为辅**。
→ **6 点全出（2026-08-01 14:22）**，过拟合诊断完成，结论见下。

**最终结论（过拟合诊断）**：
1. **融合 teacher KV 训练会随步数过拟合，且是 fused 特异的**：fused 在 step100(62.0%) 到顶后
   退化至 step200/300(60.3%)；plain 全程平台（step50/100/200/300 = 62.3 / 63.0 / 62.7 / 63.0），
   **不退化**。用户在阶段七初的直觉"融合 kv 容易过拟合"✅ 证实。
2. **早停有效但救不回 fused**：取 fused 最优 step100(62.0%) 比 step300(60.3%) 高 +1.7pp，
   但 step50(61.0%) 反而更低（欠拟合），故最优点是 **step100 而非更早**。
3. **fused 全点被 plain 压制（62.0 峰 < 63.0）**：说明 fused 跑不赢 plain **不只是过拟合**
   （纯过拟合早停点应≈plain），主因是 off-policy 评测下的 **train/test 失配**（训练塞了 KV 拐杖、
   考试不给药）→ 失配为主 + 过拟合为辅。与阶段七原结论 3（on-policy EOPD 才公平）一致。

> 经验：**跨卡/磁盘 offload 的模型在「当前设备」上临时落权重**是本次 OOM 的隐蔽来源。
> apex 这种单卡仅 ~7 GiB 的环境，遇到 `device_map="auto"` 触发磁盘 offload 时要特别警惕；
> 且 **CPU-offload 优化器在 apex 上慢 9 倍**，优先用「把附属模块挪到空闲卡」来换显存。

---

## 阶段八：对齐 C2C 官方层映射（进行中）

### 触发原因

阶段七加载的官方 fuser 配置确认使用 `mapping: last_aligned`，但本项目的
`FusedKVBuilder` 默认使用相对深度映射。对于 teacher 36 层、student 28 层：

```text
官方 C2C last_aligned: student 0 → teacher 8，student 27 → teacher 35
旧 Builder relative_depth: student 0 → teacher 0，student 27 → teacher 35
```

因此阶段七的官方 projector 很可能一直接收了错误层的 teacher KV；在修正前，不能把
阶段七的 fused 训练结果作为 C2C 方法本身的结论。

### 已完成的代码修正

- `fused_kv.py` 的 `build_layer_mapping` 现在支持 `relative_depth`、`last_aligned`、
  `k_nearest` 三种策略；当前 Builder 仍保持单个 student 层对应单个 teacher 层，
  对齐 C2C 官方训练中的 `K=1` 设置。
- `train_student_distill.py` 默认使用 `last_aligned`，因为它加载官方 fuser。
- `train_projector.py`、`eval_math_acc.py`、`eval_projector_kl.py`、`eval_fused_kv.py`
  均增加 `--layer-mapping`；旧 v6 projector 默认继续使用 `relative_depth`。
- `c2c_hf_rollout.py` 支持 rollout 配置中的 `layer_mapping`/`mapping` 字段。
- 本地纯 Python 映射验证通过：36→28 的 `last_aligned` 为 `0→8`、`27→35`；全部相关
  文件语法编译通过。

### 远端已有数据的补充观察

远端已有的 step200、`fusion_scale=0.3` 评测尚未跑满 300 题，目前 290 题为：

```text
baseline 240/290，fused 238/290
paired gain 16，loss 18
```

相较强融合时的破坏性有所减弱，但不能替代映射修正后的正式实验。

### 下一步

- [x] 将修正后的映射代码同步到 apex-llm，并保留远端旧版备份
- [x] 完成原始 0.6B + 官方 fuser 的 5/20 题 relative_depth vs last_aligned 小样本对照
- [x] `eval_math_acc.py` 支持直接加载官方 `projector_{idx}.pt/.json`
- [x] 合并 apex 原有的显式 teacher 分片、独立 projector GPU 和 fused AdamW，避免阶段八同步
      时丢失已有 OOM 修复
- [ ] 完成 `fusion_scale=0.3` 的 300 题评测
- [ ] 使用正确映射重新训练 fused student，并与 plain SFT 做同预算对比
- [ ] 再进入带/不带 C2C 的端到端 EOPD 对照

### 阶段八冒烟结果（apex-llm，2026-08-01）

已将阶段八代码同步到 `/home/kejiechen/CacheEOPD`，旧文件备份在
`cache_eopd/.mapping_backup_20260801`。使用官方
`qwen3_0.6b+qwen3_4b_base_Fuser/final`，同一批 GSM8K 前 5 题、max-new-tokens=256：

| 层映射 | baseline | official fuser | cache 自检 |
|---|---:|---:|---|
| `last_aligned` | 2/5 | 1/5 | B==C 1/1 ✅ |
| `relative_depth` | 2/5 | 2/5 | B==C 1/1 ✅ |

两种映射的生成长度和答案已经不同，说明层映射确实进入了实际推理路径；5 题结果不具备
统计意义，不能据此判断哪种策略更好。运行时仍出现 teacher 部分磁盘 offload 提示，正式
评测需要继续使用显式连续 GPU 分片，避免把性能问题混入 accuracy 结论。

### 20 题正式小样本与弱融合结果

合并显式连续分片后重新评测，运行时不再出现磁盘 offload：

| 设置 | baseline | official fuser | paired gain | paired loss |
|---|---:|---:|---:|---:|
| `last_aligned`, scale=1.0 | 13/20 | 7/20 | 0 | 6 |
| `last_aligned`, scale=0.3 | 13/20 | 11/20 | 0 | 2 |
| `relative_depth`, scale=1.0 | 13/20 | 7/20 | 1 | 7 |

`scale=0.3` 明显减少了破坏性，但暂未产生纠偏收益。`last_aligned` 与
`relative_depth` 的生成长度和文本不同，说明映射确实影响实际路径；两种策略在 20 题上
都明显低于 baseline，因此“映射错误”不是官方 fuser 负收益的唯一原因。以上样本仍只用于
筛选方向，正式结论需要扩大到 300 题。

### 阶段八正式评测登记（2026-08-01）

启动 300 题官方 fuser 弱融合评测，配置固定为：

- student/teacher：Qwen3-0.6B / Qwen3-4B base；官方 fuser final
- layer mapping：`last_aligned`
- fusion scale：`0.3`
- dataset：`GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl`
- generation：`max_new_tokens=512`
- teacher 显式分片：`cuda:2,3`；student：`cuda:1`；projector：`cuda:4`
- output：`/home/kejiechen/CacheEOPD/logs/eval_official_fuser_last_aligned_scale03_300.jsonl`

目标是确认 20 题上“弱融合减少破坏但没有 gain”的现象是否稳定，并记录 baseline/fuser
准确率、paired gain/loss 及平均生成长度。评测结束后若仍无 paired gain，将转向使用 GSM8K
teacher trajectory、正确 `last_aligned` 映射重新训练 projector，而不再把 OpenHermes 官方
fuser 作为主要实验结论。

### 正式评测第一次启动故障（2026-08-01）

第一次启动已完成模型加载和 `B==C 3/3` cache 自检，但在第一条结果写盘时因 apex 共享
`/home` 分区整体 100% 满退出（`OSError: [Errno 28] No space left on device`），不是
显存或模型代码错误。输出 jsonl/log 均为 0 字节。已将本项目 8 个旧 smoke checkpoint
（约 5.9G）移动到 `/tmp/cacheeopd-archive-20260801`，可恢复；清理后 `/home` 获得约
5.9G 空间，准备重新启动同一配置。

重启后实时检查点：已完成 `10/300` 题，baseline `8/10`、official fuser `7/10`；进程和
显式 teacher 分片均正常，评测继续进行中。

第二个检查点：`20/300` 题时 baseline `13/20`、official fuser `11/20`，与此前 20 题
弱融合结果一致；这只是进度观察，正式统计仍待全量完成。

当前评测进一步完成 `40/300`：baseline `26/40`、official fuser `23/40`；任务仍在运行。

半程前检查点：`50/300` 题时 baseline `32/50`、official fuser `28/50`，仍与弱融合负收益
方向一致，但最终判断以 300 题逐题配对统计为准。

评测完成 `100/300`：baseline `67/100`、official fuser `58/100`；中途结果仍显示弱融合
低于 baseline，完整配对统计待剩余 200 题完成。

评测完成 `130/300`：baseline `88/130`、official fuser `77/130`；显式分片和磁盘空间仍正常。

评测完成一半 `150/300`（实时监视）：磁盘仍有约 5.9G，进程保持运行；累计准确率以评测
进程最终输出和逐题 jsonl 为准。

后半段检查点：已完成 `200/300`，进程仍在运行，`/home` 余量约 5.9G；未发生新的异常。

### 阶段八正式评测结果（2026-08-01，已完成）

300 题任务最终正常完成，完整结果文件为：
`/home/kejiechen/CacheEOPD/logs/eval_official_fuser_last_aligned_scale03_300.jsonl`。
配置与登记一致：官方 fuser、`last_aligned`、`fusion_scale=0.3`、`max_new_tokens=512`、
student `cuda:1`、teacher 显式连续分片 `cuda:2,3`。

| 设置 | 正确率 | 平均生成长度 |
|---|---:|---:|
| baseline | 184/300 = 61.3% | 263.88 |
| official fuser, scale=0.3 | 162/300 = 54.0% | 234.02 |

逐题配对统计：

- 两者都对：136；两者都错：90
- baseline 对、fuser 错（loss）：48
- baseline 错、fuser 对（gain）：26
- 净 paired 变化：`26 - 48 = -22` 题，即 `-7.3pp`
- 两臂均抽取到答案，cache 自检 `B==C 3/3`；因此不是答案抽取失败或 cache decode
  路径不一致造成的假象。

按 paired 类别的平均生成长度进一步看，`baseline 对/fuser 错` 的 48 题从 281.0 降到
249.9 tokens，`baseline 错/fuser 对` 的 26 题也从 282.2 降到 240.7；两者都对的 136 题
从 226.7 降到 201.0。说明弱融合整体改变了终止/轨迹长度分布，而不是只在少数题上做
局部知识纠偏。

与 20 题结果（baseline 13/20、fuser 11/20、gain 0/loss 2）方向一致；扩大到 300 题后，
弱融合没有带来纠偏收益，反而降低正确率并缩短生成。由此确认：**修正为官方
`last_aligned` 映射后，OpenHermes 官方 fuser 仍不能在 GSM8K 上直接帮助 0.6B student**。
“旧 Builder 映射错误”不是负收益的唯一根因；更主要的问题是官方 fuser/训练数据与当前
GSM8K 任务的领域和轨迹分布失配，以及 student 在测试时依赖了不一致的 teacher KV 前缀。

### 阶段八决策

- [x] 正确映射下完成官方 fuser 300 题验收
- [ ] 不再继续以 OpenHermes 官方 fuser 做主要 scale sweep
- [ ] 用 GSM8K teacher trajectory、`last_aligned` 映射重新训练 projector
- [ ] 在同一数据、步数、学习率和评测协议下做 fused student vs plain SFT
- [ ] 再进入真正 on-policy EOPD 的 w/ vs w/o C2C 对照

下一实验应优先验证“任务域/轨迹对齐”而不是继续调融合强度：teacher 和 student 对同一
GSM8K prompt 生成的 teacher trajectory 应用于 projector 训练；projector 的每个 student
层按 C2C 的 `last_aligned` 映射取 teacher 层 KV。训练侧必须保留 teacher 显式分片、student
和 projector 独立放卡的 OOM 修复。验收时同时记录 student 原生、zero-cache、fused 三路，
并使用 paired gain/loss；只有 fused 在原生 student 不可用的题上产生稳定 gain，才进入
EOPD 端到端实验。

### 阶段九：GSM8K trajectory projector 重训登记（进行中）

远端已有 `/home/kejiechen/CacheEOPD/data/teacher_traj_gsm8k1500.jsonl`，实际包含 1368
条带 `problem/prompt/solution/label` 的 teacher trajectory，可直接作为 C2C SFT-CE 输入；
其数据域是 GSM8K，而不是官方 fuser 使用的 OpenHermes 轨迹。当前重训计划：

- student/teacher：Qwen3-0.6B / Qwen3-4B base
- data：`teacher_traj_gsm8k1500.jsonl`，前 64 条 holdout
- mapping：`last_aligned`
- projector：per-layer、zero-init、可学习 gate；600 steps、lr `1e-4`、grad accum `8`
- resources：student `cuda:1`，teacher 显式连续分片 `cuda:2,3`
- output：`/home/kejiechen/CacheEOPD/ckpt_projector_v7_last_aligned`

重训前修正了 `train_projector.py`：此前它仍使用 `device_map=auto`，可能把 teacher 层放到
磁盘并在前向时临时搬到 student 卡；现在 auto 模式复用显式连续 teacher 分片逻辑，与评测
和 fused student 训练一致。代码已通过本地 `py_compile` 与 `git diff --check`。

重训实时检查点：step20 holdout CE `0.2318` / token acc `0.921`，相对 student baseline
CE `0.2535` / acc `0.916` 已改善；step40 holdout CE `0.2272` / acc `0.924`，28 个 gate
仍全部开启。训练进程继续运行，尚未进行下游生成验收。

重训已到 step100：holdout CE `0.2275` / token acc `0.925`，已保存
`ckpt_projector_v7_last_aligned/projector_step100.pt`；step80 的 holdout CE 曾达到
`0.2236`，后续继续训练以观察是否过拟合。

step180 达到当前最佳 holdout CE `0.2171` / token acc `0.928`；step200 为 CE `0.2194` /
acc `0.926`，并已保存 `projector_step200.pt`。训练继续，后续下游验收优先检查 step140--180
而不是默认只用最终 checkpoint。

step220 holdout 回落到 CE `0.2303` / acc `0.922`，初步出现 projector 过拟合/门控退化迹象；
继续跑到后续保存点，用 step200 作为当前可复现实验 checkpoint。

训练最终完整跑到 step600（此前尝试中断时进程已继续完成），并保存
`projector_step100/200/300/400/500/600.pt` 及 `projector_final.pt`。holdout 曲线显示：

| step | holdout CE | token acc |
|---:|---:|---:|
| 200 | 0.2194 | 0.926 |
| 300 | 0.2190 | 0.925 |
| 400 | 0.2553 | 0.921 |
| 500 | 0.2520 | 0.922 |
| 600 | 0.2815 | 0.921 |

最佳未保存的中间点约为 step180（CE 0.2171 / acc 0.928），可复现实验先使用已保存的
step200 与 step300；final/step600 明显过拟合。28 个 key/value gate 在最终均开启，说明
本轮 gate 没有学会关闭有害层，后续可将 gate 初始化/学习率作为单独消融，而不应把最终
checkpoint 当作默认模型。

### 阶段九小样本下游验收（已完成）

在同一 50 题 GSM8K 前缀、`max_new_tokens=512`、`last_aligned` 和显式 teacher 分片下：

| projector | baseline | fused | paired gain/loss | 平均长度 baseline → fused |
|---|---:|---:|---:|---:|
| step200 | 32/50 = 64.0% | 34/50 = 68.0% | 4 / 2 | 254.2 → 313.7 |
| step300 | 32/50 = 64.0% | 32/50 = 64.0% | 3 / 3 | 254.2 → 299.4 |

step200 是当前候选最佳，但 50 题标准误较大，不能作为最终结论。明细文件分别为
`eval_projector_v7_step200_50.jsonl` 与 `eval_projector_v7_step300_50.jsonl`。
已将明显过拟合的 step400/500/600/final 权重（约 1.4G）移动到
`/tmp/cacheeopd-archive-20260801/projector_v7`，保留 step100/200/300 在实验目录。

### 阶段九正式验收登记（进行中）

对 `projector_step200.pt` 运行完整 300 题 GSM8K 验收，输出：
`/home/kejiechen/CacheEOPD/logs/eval_projector_v7_step200_300.jsonl`。配置与阶段八 300
题验收一致，目标是确认 50 题的 `+4pp` 和 paired `4/2` 是否稳定；若全量仍为正向，下一步
再与 plain SFT 做同预算对照。

### 阶段九正式验收结果（2026-08-01，已完成）

`projector_step200.pt` 的完整 300 题结果：
`/home/kejiechen/CacheEOPD/logs/eval_projector_v7_step200_300.jsonl`。

| 设置 | 正确率 | 平均生成长度 |
|---|---:|---:|
| baseline student | 184/300 = 61.3% | 263.88 |
| GSM8K trajectory projector step200 + `last_aligned` | 192/300 = 64.0% | 308.17 |

逐题配对：两者都对 162、两者都错 86；gain（baseline 错、fused 对）30；loss（baseline
对、fused 错）22；净变化 `+8` 题，即 `+2.7pp`。50 题预验收为 34/50 vs 32/50、gain/loss
4/2，因此正向方向在扩大样本后保持。两臂均抽取到答案，cache 自检 `B==C 3/3`。

**结论**：step200 已证明“GSM8K 域 teacher trajectory + C2C 正确 `last_aligned` 映射”
比官方 OpenHermes fuser 更合理，并在当前 300 题上取得正向点估计；它把官方 fuser 的
`61.3% → 54.0%` 负收益扭转为 `61.3% → 64.0%`。但 paired discordant 样本只有 52 题，
gain/loss 差为 8，按单臂标准误/配对检验尚不足以宣称统计显著，所以必须继续做同预算
plain SFT 对照，不能把这次结果直接归因于 C2C 的全部收益。

平均长度从 263.9 增至 308.2，说明 projector 不再像官方 fuser 那样整体缩短轨迹；它更
可能恢复了数学题所需的解题过程。不过长度增加本身不是正确率证据，后续仍以 paired
accuracy 为主。

### 阶段九决策与下一步

- [x] GSM8K trajectory + `last_aligned` projector 300 题验收完成
- [ ] 用相同 300 题/teacher trajectory、训练步数和资源做 plain SFT student 对照
- [ ] 比较 plain SFT、trajectory projector fused student、base student 三者
- [ ] 若 fused 优于 plain，再进入 on-policy EOPD 的 w/ vs w/o C2C 对照

下一轮优先复用已验证的显式 teacher 分片和磁盘清理策略，训练 plain SFT；评测固定使用同
一 GSM8K 300 题、同一 chat template、`max_new_tokens=512` 和逐题 paired 统计。这样可以
区分“GSM8K teacher trajectory 本身带来的 SFT 收益”和“KV projector/C2C 额外收益”。

### 与既有 plain SFT 的同题对照补充

已有 `eval_plain_student.jsonl` 在同一 300 题上的 plain SFT 结果为 `189/300=63.0%`；
新 projector step200 为 `192/300=64.0%`，base student 为 `184/300=61.3%`。逐题配对得到：

- projector vs plain SFT：gain/loss `26/23`，净增 3 题（约 `+1.0pp`）
- plain SFT vs base：gain/loss `32/27`，净增 5 题（约 `+1.7pp`）
- 平均长度：projector `308.2`、plain SFT `293.1`、base `263.9`

这说明目前主要收益很可能来自 GSM8K trajectory/任务域对齐；C2C KV projector 相比普通
plain SFT 只有小幅点估计增益，paired 差异尚不足以称为显著。下一阶段不应直接宣称
“C2C 已胜出”，而应做更严格的同预算训练对照、多个 seed 或更大评测集，并优先研究为何
projector 让轨迹变长但只带来约 1pp 的额外正确率。

### 阶段十：EOPD baseline 对照（进行中，2026-08-01）

用户要求补充与 EOPD baseline 的公平对照。此前远端 EOPD 日志只对应 Qwen2.5-3B/
Qwen3-8B、GSM8K、10 步 smoke，不能直接拿来和当前 Qwen3-0.6B/Qwen3-4B 的 CacheEOPD
结果比较。本轮固定当前实验的 student/teacher 和 GSM8K 任务，先用 `Baselines/EOPD` 的
`OnPolicyDistillTrainer` 做 5 步 smoke，再扩展到正式训练；评测仍使用同一 300 题、同一
chat template、greedy、`max_new_tokens=512` 和同一答案抽取器。

本轮对照定义：

- base：未训练 Qwen3-0.6B；
- plain：GSM8K teacher trajectory 的普通 SFT；
- EOPD：学生 on-policy rollout + teacher token log-prob 的 EOPD；
- CacheEOPD：GSM8K trajectory projector step200 的 `last_aligned` KV 融合。

最终除了总准确率，还要输出 EOPD/CacheEOPD 的逐题 gain/loss：EOPD 对而 CacheEOPD
错、CacheEOPD 对而 EOPD 错，以及三者共同正确/共同错误，并记录每类生成长度和答案文本。

EOPD HF smoke 已完成：使用当前 Qwen3-0.6B/Qwen3-4B、student `cuda:1`、teacher
显式分片 `cuda:2,3`，学生 on-policy 采样 64 tokens，连续更新 2 步，过程无 OOM。step1/2
loss 为 `1.5132/1.1457`，teacher entropy 均值为 `0.074/0.124`，满足 EOPD 默认阈值
`0.8` 的 Soft-KD token 比例仅 `3.1%/1.6%`。因此本实验的 EOPD 主要由 clipped
reverse-KL 项驱动；正式结果需同时记录 Soft-KD 覆盖率，不能只看最终准确率。

根据后续对照要求，正式 EOPD runner 使用 max response `256`、300 steps，并保存
`step100/200/300` HF checkpoint。最终固定比较两组：EOPD step200 vs CacheEOPD projector
step200，以及 EOPD step300 vs CacheEOPD projector step200；两组都在同一 300 题上做
paired accuracy 和逐题题目/答案翻转分析。

### 阶段十补充：EOPD step200/300 严格 self-KV 对照（已完成）

为避免 batch `generate` 与 CacheEOPD 自定义 cache decode 的协议差异，EOPD step200 改用
`eval_math_acc.py` 的 self-KV 解码路径，并将 300 题拆成三路并行评测后恢复全局索引。结果如下：

| 设置 | 正确率 | 平均生成长度 |
|---|---:|---:|
| plain SFT | 189/300 = 63.0% | 293.1 |
| EOPD step200（严格 self-KV） | 186/300 = 62.0% | 280.2 |
| CacheEOPD step200 | 192/300 = 64.0% | 308.2 |

CacheEOPD vs EOPD step200 的 paired 结果为：两者都对 160、两者都错 82；CacheEOPD
额外答对 32 题，EOPD 额外答对 26 题，净增 6 题（+2.0pp）。与 batch 结果的细微差异
来自 bf16 自回归 decode 的路径，而不是答案抽取器；后续以严格 self-KV 结果为准。

严格对照中，CacheEOPD 的代表性增益包括：
- idx26（348）：EOPD 答 288，CacheEOPD 答 348，修正了兔子/狗/猫总数的列式；
- idx39（272）：EOPD 答 208，CacheEOPD 答 272，正确处理每周课程小时数；
- idx42（15）：EOPD 答 -15，CacheEOPD 答 15，正确比较自己做税与雇会计的净收益；
- idx88（1198）：EOPD 答 186，CacheEOPD 答 1198，正确合并奖品、刻字、胸针和绶带费用。

代表性损失包括：
- idx23（44）：EOPD 答 44，CacheEOPD 答 11，CacheEOPD 少算了旅行年数对应的衬衫数；
- idx28（72）：EOPD 答 72，CacheEOPD 答 180，内盒厚度导致的内部体积计算被扰动；
- idx30（42）：EOPD 答 42，CacheEOPD 答 120，把 7:13 比例题误读成糖用量等于总量；
- idx60（276000）：EOPD 答 276000，CacheEOPD 答 30000，税费与注册费的合计被破坏。

这说明 CacheEOPD 的收益确实集中在 EOPD 本身算错的题上，但仍会扰动 EOPD 已经答对的
题；因此当前证据支持“同一步数下 CacheEOPD 可能更有效”，尚不能支持“CacheEOPD step200
必然超过 EOPD step300”。

step300 严格 self-KV 结果如下：

| 设置 | 正确率 | 平均生成长度 |
|---|---:|---:|
| plain SFT | 189/300 = 63.0% | 293.1 |
| EOPD step300（严格 self-KV） | 193/300 = 64.3% | 282.2 |
| CacheEOPD step200 | 192/300 = 64.0% | 308.2 |

CacheEOPD vs EOPD step300 的 paired 结果为：两者都对 166、两者都错 81；CacheEOPD
额外答对 26 题，EOPD 额外答对 27 题，净变化 `-1` 题（-0.3pp）。因此“CacheEOPD
step200 超过 EOPD step300”没有得到支持；更准确的结论是 CacheEOPD 在 step200 已达到
与 EOPD step300 几乎相同的效果，但本次严格评测仍略低 1 题。

跨步数对照的趋势是：CacheEOPD step200 相比 EOPD step200 为 `32 gain / 26 loss`，净增
6 题；EOPD 继续训练到 step300 后，相比 CacheEOPD 变为 `27 gain / 26 loss`，净增 1 题。
这支持“CacheEOPD 可能更快获得有效监督/达到较好点”的弱结论，不支持“KV 融合必然提高
最终上限”。此外两者 step 不能直接视为等价计算量：EOPD 更新 student 权重，CacheEOPD
训练的是 projector 且评测时仍需要 teacher KV。

严格结果文件：`logs/three_way_strict_step200.jsonl`、
`logs/three_way_strict_step200_summary.json`、`logs/three_way_strict_step300.jsonl`、
`logs/three_way_strict_step300_summary.json`（远端 CacheEOPD 实验目录）。

严格 paired 题号集合（idx 从 0 开始）：
- step200 CacheEOPD gain：26, 39, 42, 45, 54, 59, 62, 88, 91, 97, 99, 100, 101, 118, 119, 124, 140, 143, 149, 179, 185, 193, 201, 205, 207, 224, 231, 245, 265, 267, 276, 286；loss：23, 28, 30, 60, 86, 141, 166, 173, 176, 186, 190, 196, 198, 203, 209, 212, 229, 234, 236, 238, 239, 242, 248, 254, 255, 280。
- step300 CacheEOPD gain：24, 26, 39, 57, 62, 88, 91, 97, 99, 100, 101, 130, 142, 143, 157, 164, 179, 185, 193, 201, 207, 219, 232, 245, 267, 279；loss：23, 28, 50, 60, 63, 73, 84, 86, 148, 166, 176, 181, 186, 190, 196, 198, 209, 212, 217, 229, 234, 236, 238, 239, 242, 254, 255。

### 阶段十一：CacheEOPD student 权重无 teacher 独立评测（已完成，2026-08-02）

针对“评测时不应依赖 teacher KV”的修正，本阶段使用真正不加载 teacher 的
`eval_student_batch.py`，直接加载 `ckpt_student_fused/student_stepN` 或
`ckpt_student_plain/student_stepN`，只调用 student 原生 batch greedy generation。所有模型
使用相同的 300 题、chat template、batch size 8 和 `max_new_tokens=512`；这与之前
`eval_math_acc.py` 的 self-KV 单题路径不同，因此结果只在本阶段内部横向比较。

| checkpoint | standalone 正确率 | 平均生成长度 |
|---|---:|---:|
| base student | 181/300 = 60.3% | 257.9 |
| plain step100 | 189/300 = 63.0% | 295.0 |
| fused step100 | 185/300 = 61.7% | 263.4 |
| plain step200 | 177/300 = 59.0% | 299.9 |
| fused step200 | 179/300 = 59.7% | 272.1 |
| plain step300 | 182/300 = 60.7% | 292.5 |
| fused step300 | 170/300 = 56.7% | 268.2 |

fused vs plain 的 paired gain/loss 为：step100 `25/29`（净 -4）、step200 `32/30`（净 +2）、
step300 `25/37`（净 -12）。因此在真正独立的 student 权重评测下，C2C fused 训练没有稳定
超过 plain SFT；step300 明显退化，表现为 fused 特异的过拟合或训练目标失配。之前
`projector_step200` 的 `192/300` 不能作为独立 student 结果，它属于 teacher KV 辅助推理。

本阶段结果文件：`logs/eval_standalone_{fused,plain}_step{100,200,300}.jsonl`，以及
`logs/standalone_three_way_step{100,200,300}_summary.json`（远端 CacheEOPD 实验目录）。

### 阶段十二：student-only mixed / anneal 交替注入实验（进行中，2026-08-02）

本阶段目标是训练 student，而不是继续训练 projector。projector 在训练前加载并冻结，optimizer
只接收 student 参数；评测统一不加载 teacher，也不使用 teacher KV。训练数据仍为
`data/teacher_traj_gsm8k1500.jsonl`，保存和 holdout 检查间隔均为 50 步。

当前先使用此前验证过的本项目 projector：
`ckpt_projector_v7_last_aligned/projector_step200.pt`。该 projector 的 teacher-assisted
探针为 192/300，相比 base 184/300，因此暂时保留；它不代表独立 student 结果。

当前运行组：

| 组别 | projector | KV 注入策略 | 配置 | 状态 |
|---|---|---|---|---|
| mixed_v7 | prior v7 | 每个 micro-batch 随机注入 | `p=0.5`, 300 steps, `lr=1e-5`, `grad_accum=8` | 第 100 步，已保存 step50/100 |
| anneal_v7 | prior v7 | 注入概率线性退火 | `p: 1.0 -> 0.0`, 300 steps | 待启动 |

`mixed_v7` 当前日志中的 plain holdout CE：step50=`0.2346`，step100=`0.2288`。最终判断以
各 step 的无 teacher 独立准确率为准，并与 no-distillation/plain、EOPD dense checkpoint
以及已有 EOPD 结果做 paired 题目级比较。

训练在首次运行至 step250 时因远端 `/home` 磁盘空间不足而在保存半成品时退出；step50--200
完整权重和日志未受影响。已将旧的 `ckpt_student_fused`、`ckpt_student_plain` 权重目录以及
失败的 step250 半成品移至远端 `/tmp` 的可恢复归档目录，释放空间后用相同 seed/config 重启
`mixed_v7`，以确保 step250/300 也完整保存。

`mixed_v7` 重启后已完成全部 300 步，plain holdout CE 为：step50=`0.2346`、step100=`0.2286`、
step150=`0.2227`、step200=`0.2210`、step250=`0.2206`、step300=`0.2171`。六个 checkpoint
均已保存。当前正在 GPU1--5 并行进行真正不加载 teacher 的 300 题 greedy 评测；评测输出为
`logs/eval_mixed_v7_step{50,100,150,200,250,300}.jsonl`。

mixed 已完成的独立准确率：step50=`181/300`、step100=`180/300`、step150=`186/300`、
step200=`192/300`、step250=`190/300`、step300=`188/300`。对应平均生成长度依次为
288.35、292.55、291.90、289.27、301.92、289.75；六个结果文件均已完成。

`anneal_v7` 已启动（PID 1366451）：同一 prior v7 projector、同一 student-only 目标和资源，
仅使用 `anneal-start-prob=1.0`、`anneal-end-prob=0.0`、`anneal-steps=300`，同样每 50 步保存。

`anneal_v7` 已完成 step50--300 全部保存；plain holdout CE 为：step50=`0.2514`、
step100=`0.2387`、step150=`0.2276`、step200=`0.2223`、step250=`0.2181`、step300=`0.2131`。
当前正在 GPU1--5 并行做 anneal 的 student-only 300 题评测，输出为
`logs/eval_anneal_v7_step{50,100,150,200,250,300}.jsonl`。

为完成三方对照，已将 mixed 权重移至远端 `/tmp/cacheeopd_ckpt_student_mixed_v7_archive_20260802`
（结果 JSONL 和日志仍在 `CacheEOPD/logs`），并启动 EOPD dense baseline：student-only 更新、
teacher 分片 GPU2/3、student GPU1、`lr=1e-6`、300 steps、`save-every=50`，输出目录为
`ckpt_eopd_dense`，日志为 `logs/train_eopd_dense.log`。

EOPD dense 已完成 300 steps，`step50`、`step100`、`step150`、`step200`、`step250`、`step300`
全部保存。当前正在 GPU1--5 并行进行同一 300 题、同一 chat template、batch=8、
`max-new-tokens=512` 的 student-only 独立评测，输出为
`logs/eval_eopd_dense_step{50,100,150,200,250,300}.jsonl`。

### 阶段十三：mixed / anneal / EOPD 三方独立结果（已完成，2026-08-02）

三组都直接加载 student checkpoint，评测时不加载 teacher、不构造 teacher KV；统一使用
GSM8K-COT 300 题、同一 chat template、greedy、batch size 8、`max-new-tokens=512`。
此前同协议 base student 为 `181/300`，平均生成长度 257.92。

| step | mixed | anneal | EOPD dense |
|---:|---:|---:|---:|
| 50 | 181/300 (288.35) | 178/300 (268.38) | 188/300 (268.39) |
| 100 | 180/300 (292.55) | 188/300 (284.62) | 175/300 (274.81) |
| 150 | 186/300 (291.90) | 187/300 (290.57) | 183/300 (273.80) |
| 200 | **192/300 (289.27)** | 184/300 (288.72) | 185/300 (275.56) |
| 250 | 190/300 (301.92) | 189/300 (306.53) | 190/300 (279.99) |
| 300 | 188/300 (289.75) | 189/300 (295.50) | 186/300 (282.34) |

括号内为平均生成 token 数。相对 base，mixed 最佳 step200 净增 11 题（+3.67pp）；
anneal 最佳 step250 净增 8 题（+2.67pp）；EOPD dense 最佳 step50/250 净增 7/9 题，
但 EOPD 曲线在 step100--200 有明显波动。三组 raw 逐题结果分别为：
`logs/eval_mixed_v7_step{50,100,150,200,250,300}.jsonl`、
`logs/eval_anneal_v7_step{50,100,150,200,250,300}.jsonl`、
`logs/eval_eopd_dense_step{50,100,150,200,250,300}.jsonl`。

#### 同一步数 paired 对照

下表格式为“候选相对 EOPD 的 gain / loss / 净变化”，gain 表示候选答对而 EOPD 答错，
loss 表示 EOPD 答对而候选答错：

| step | mixed vs EOPD | anneal vs EOPD | anneal vs mixed |
|---:|---:|---:|---:|
| 50 | 20 / 27 / -7 | 20 / 30 / -10 | 19 / 22 / -3 |
| 100 | 28 / 23 / +5 | 36 / 23 / +13 | 32 / 24 / +8 |
| 150 | 25 / 22 / +3 | 26 / 22 / +4 | 17 / 16 / +1 |
| 200 | 31 / 24 / +7 | 26 / 27 / -1 | 10 / 18 / -8 |
| 250 | 27 / 27 / 0 | 21 / 22 / -1 | 16 / 17 / -1 |
| 300 | 31 / 29 / +2 | 31 / 28 / +3 | 24 / 23 / +1 |

这表明交替注入不是稳定的单调收益：mixed 在 step100--200 更有优势，anneal 在 step100
和 step300 更好；两者都没有在所有训练区间稳定胜过 EOPD。

#### 重点题目级比较

最有代表性的公平比较是 mixed step200（192）对 EOPD step200（185）：mixed 额外答对
31 题，丢失 24 题，净增 7 题。gain 题号（括号为标准答案）为：
`33(10), 40(80), 45(95), 47(350), 48(23), 57(54), 59(90), 84(9), 97(12), 119(16),
121(2), 124(29), 130(82), 139(6), 143(312), 150(330000), 157(25), 158(200), 169(22),
174(4), 175(250), 204(8), 207(6), 215(17), 259(93), 263(7300), 267(20), 271(19),
277(520), 286(54), 289(27)`；loss 题号为：
`26(348), 42(15), 46(100), 54(114,200), 63(27000), 93(350), 141(30), 145(4800),
146(1050), 162(1), 181(16), 190(30), 196(10), 201(480), 203(15), 205(5), 212(2000),
229(6000), 231(45), 236(160), 249(113), 255(25), 284(147), 294(60)`。

跨训练长度的对照 mixed step200（192）对 EOPD step300（186）仍为 28 gain / 22 loss，
净增 6 题。gain 为：
`24(22), 40(80), 45(95), 91(240000), 97(12), 121(2), 124(29), 130(82), 139(6),
140(8), 143(312), 150(330000), 157(25), 158(200), 164(4), 169(22), 175(250), 184(19),
204(8), 207(6), 215(17), 219(30), 232(12), 263(7300), 267(20), 271(19), 286(54),
289(27)`；loss 为：
`26(348), 30(42), 46(100), 54(114,200), 63(27000), 88(1198), 93(350), 145(4800),
146(1050), 162(1), 181(16), 190(30), 196(10), 200(1), 210(60), 212(2000), 225(15),
231(45), 249(113), 255(25), 284(147), 294(60)`。

代表性 gain 题型包括：闹钟/跑步/土豆的多步比例或速率题（idx24/40/45）、设备折旧与
成本题（idx91/150）、年龄差和数量关系题（idx97/121/219）、洗衣用水与产量题
（idx143/263）。代表性 loss 包括：兔狗猫总数列式（idx26）、糖水比例（idx30）、
化妆师成本（idx63）、奖品总成本（idx88）、珠宝价格（idx145）、徽章速度（idx190）、
水池抽水（idx212）、图片平均分配（idx231）。因此 CacheEOPD 确实修正一部分 EOPD
错误，但也会扰动 EOPD 原本正确的列式、比例和费用题；收益不是无条件的能力提升。

本阶段结论：在当前 0.6B student、1500 条轨迹和 prior v7 projector 下，student-only
的 mixed/anneal 交替策略是可行且有局部收益的，mixed step200 达到 64.0%，高于 base
60.3%、同一步 EOPD 61.7%，也略高于 EOPD step300 的 62.0%。但它没有证明 KV 注入提高
最终上限；更准确的结论是“适度交替注入可能改变有效训练速度和最优 checkpoint 区间”，
其中 mixed 比 anneal 更稳定，后续大规模实验应优先围绕 mixed 的注入比例、注入时机和
更密的 step150--250 区间做消融。

anneal 已完成的独立准确率：step50=`178/300`（平均 268.38 tokens）、step100=`188/300`
（284.62）、step150=`187/300`（290.57）、step200=`184/300`（288.72）、step250=`189/300`
（306.53）、step300=`189/300`（295.50）。六个结果文件均已完成。

### 阶段十四：mixed 多 seed 稳定性消融（已完成，2026-08-02）

为检验 mixed step200 的 `192/300` 是否只是单次随机种子波动，新增两个完全相同配置的
多 seed 训练：`seed=41718` 和 `seed=41719`。两组均训练 student，不更新 projector；训练时
每个 micro-batch 以 `p=0.5` 注入 teacher KV，训练 300 步、`grad_accum=8`、`lr=1e-5`，
每 50 步保存 checkpoint，并使用同一个 prior v7 projector：
`ckpt_projector_v7_last_aligned/projector_step200.pt`。

评测协议预先固定为 student-only：加载学生 checkpoint 时不加载 teacher、不加载 projector、
不注入 teacher KV，只使用 student 自己的 KV cache；与阶段十三相同的 GSM8K-COT 300 题、
greedy、batch size 8、`max-new-tokens=512`。训练输出分别为：
`ckpt_student_mixed_v7_seed41718`、`ckpt_student_mixed_v7_seed41719`；训练日志分别为：
`logs/train_mixed_v7_seed41718.log`、`logs/train_mixed_v7_seed41719.log`。

`seed=41718` 已完成训练和 student-only 独立评测。训练日志为
`logs/train_mixed_v7_seed41718.log`，六个 raw 评测文件为
`logs/eval_mixed_v7_seed41718_step{50,100,150,200,250,300}.jsonl`。结果如下：

| step | mixed seed41718 | 平均生成长度 |
|---:|---:|---:|
| 50 | 188/300 (62.67%) | 284.31 |
| 100 | 184/300 (61.33%) | 298.02 |
| 150 | 186/300 (62.00%) | 289.26 |
| 200 | 189/300 (63.00%) | 295.52 |
| 250 | **199/300 (66.33%)** | 298.33 |
| 300 | 198/300 (66.00%) | 300.52 |

该 seed 的最佳点移动到 step250，暂时没有复现 seed41717 的 step200=`192/300`，但整体
验证了 mixed 的有效区间可能在 step200--300，并说明单个 seed 的最优 checkpoint 不足以
代表稳定峰值。评测过程中发现训练脚本不会把 tokenizer 同步到 step 子目录，已在远端为
每个 checkpoint 补齐 tokenizer 文件；同时统一使用实际数据文件
`taopd-baseline/data/GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl`。

为释放 `/home` 空间，seed41718 权重完整归档到远端
`/dev/shm/cacheeopd_ckpt_student_mixed_v7_seed41718_archive_20260802`，raw 结果和日志仍在
`CacheEOPD/logs`。

`seed=41719` 已完成训练和 student-only 独立评测。训练日志为
`logs/train_mixed_v7_seed41719.log`，评测文件为
`logs/eval_mixed_v7_seed41719_step{50,100,150,200,250,300}.jsonl`。结果如下：

| step | mixed seed41719 | 平均生成长度 |
|---:|---:|---:|
| 50 | 191/300 (63.67%) | 292.82 |
| 100 | 190/300 (63.33%) | 297.07 |
| 150 | 189/300 (63.00%) | 296.06 |
| 200 | 186/300 (62.00%) | 296.53 |
| 250 | **195/300 (65.00%)** | 303.20 |
| 300 | 195/300 (65.00%) | 290.52 |

当前三个 seed 的最佳点分别为：原 seed41717 的 step200=`192/300`、seed41718 的
step250=`199/300`、seed41719 的 step250/300=`195/300`。这已经显示 mixed 的收益区间
可能稳定落在 200--300 步附近，但最佳 step 会随 seed 移动；最终均值/标准差和 paired
统计待官方 fuser 对照完成后统一计算。

为释放 `/home` 空间，seed41719 权重随后归档到远端
`/dev/shm/cacheeopd_ckpt_student_mixed_v7_seed41719_archive_20260802`，raw 结果和日志仍在
`CacheEOPD/logs`。

#### 阶段十四补充：官方 fuser 泛化性对照（已完成）

为检验 mixed 的收益是否依赖于专门针对 GSM8K 重新训练的 projector，新增一组官方 fuser
对照。训练仍使用同一 student-only mixed 策略（`p=0.5`、300 步、`lr=1e-5`、
`grad_accum=8`、每 50 步保存），只将 projector 替换为官方：
`/home/kejiechen/taopd-baseline/modelweights/qwen3_0.6b+qwen3_4b_base_Fuser/final`。
为便于比较，计划沿用 `seed=41717`，输出到
`ckpt_student_mixed_official_v7_seed41717`，并重点独立评测 step150/200/250/300；
评测时仍不加载 teacher、不加载 projector、不注入 teacher KV。

训练日志为 `logs/train_mixed_official_v7_seed41717.log`，student-only 评测文件为
`logs/eval_mixed_official_v7_seed41717_step{150,200,250,300}.jsonl`。结果如下：

| projector | step150 | step200 | step250 | step300 |
|---|---:|---:|---:|---:|
| prior v7（同 seed41717） | 186 | **192** | 190 | 188 |
| 官方 fuser（同 seed41717） | 187 | 186 | **194** | 176 |

官方 fuser 的平均生成长度分别为 302.73、299.09、304.07、291.95 tokens。相对同 seed
prior-v7 mixed 的逐题 paired gain/loss 为：step150=`26/25`（净+1）、step200=`26/32`
（净-6）、step250=`28/24`（净+4）、step300=`21/33`（净-12）。因此官方 fuser 并非
完全不能用于 mixed：它在 step250 达到 `194/300`，与 prior-v7 三 seed 的均值接近；
但它不能复现 prior-v7 的 step200=`192/300`，且 step300 明显不稳定。结论是 mixed 训练
策略对 projector 有一定泛化性，但 projector 质量和训练步数共同决定最优区间；不能把
官方 fuser 的单个峰值或 prior-v7 的单个峰值直接当作普遍规律。

三个 prior-v7 seed 的 student-only accuracy 均值/总体标准差如下（seed41717/41718/41719）：

| step | 三 seed 原始结果 | 均值 ± std |
|---:|---:|---:|
| 50 | 181, 188, 191 | 186.67 ± 4.19 |
| 100 | 180, 184, 190 | 184.67 ± 4.11 |
| 150 | 186, 186, 189 | 187.00 ± 1.41 |
| 200 | 192, 189, 186 | 189.00 ± 2.45 |
| 250 | 190, 199, 195 | **194.67 ± 3.68** |
| 300 | 188, 198, 195 | **193.67 ± 4.19** |

多 seed 结果不支持“mixed 必然在 step200 达到 192”，但支持更稳健的表述：在当前配置
下，mixed 的较优区间集中在 step200--300，三 seed 的平均最佳点为 step250；官方 fuser
也在 step250 出现局部峰值。这增强了“交替 KV 改变有效训练速度/最优 checkpoint 区间”
的证据，但仍不能证明它提高最终能力上限。所有上述 accuracy 都是独立 student-only
评测，不加载 teacher、不加载 projector、不注入 teacher KV。

### 阶段十五：按 EOPD 原论文扩大对照（启动于 2026-08-02）

本阶段转向严格的官方 EOPD 训练口径，目标不是继续把 GSM8K 上的 projector 结果外推，而是
在同一训练入口、同一学生/教师规模和同一评测协议下，比较官方 EOPD 与 CacheEOPD：

- 论文：`Entropy-Aware On-Policy Distillation of Language Models`，Jin et al.，ICML 2026，
  arXiv:2603.07079v3，代码仓库 `WLS04/EOPD`。
- 论文配置：Qwen3-1.7B-Base student、Qwen3-8B teacher（non-thinking）、MATH 训练集、
  3 epochs、batch 128 / mini-batch 32、teacher top-k 16、`tau=0.8`、`alpha=1.0`、
  训练 response 上限 4096。
- 论文评测：MATH500、AMC23、Minerva、OlympiadBench、AIME24、AIME25，temperature 1.0、
  top-p 0.8、每题 8 次采样，报告 Avg@8 与 Pass@8。

#### 对照设计

1. 官方 EOPD：官方 `OnPolicyDistillTrainer` + vLLM rollout，使用 clipped reverse-KL 和
   teacher entropy-gated top-k forward-KL。
2. CacheEOPD-EOPD：保持同一 EOPD loss、数据、优化器和训练预算，只把 student rollout
   换为 `c2c_hf`，在生成前缀阶段注入 teacher KV；生成完成后仍按普通 EOPD 方式计算 loss。
3. 评测两组都只加载保存后的 student 权重，不加载 teacher KV、不依赖 teacher 推理；先按
   50 step 保存 checkpoint，再在相同 checkpoint 上做 student-only 六基准评测。

#### 当前状态

- 已阅读论文原文及 Appendix A/F，确认本阶段不能把前面 0.6B、GSM8K、student-only HF
  训练结果冒充论文级 EOPD 对照。
- 已核对本地官方代码入口 `eopd-baseline/examples/on_policy_distillation/on_policy_it.sh`，
  以及 CacheEOPD 的 `c2c_hf` rollout 接入点 `verl/workers/fsdp_workers.py`。
- 已在 apex-llm 上完成资源/显存冒烟；正式训练日志、checkpoint、六基准分数和逐题对照
  尚未产生。若严格三 epoch 的 4096-token 配置超出共享机器吞吐，将先保留相同预算的
  缩短 response pilot，并明确标注为 pilot，不与论文最终数字混写。

#### 阶段十五补充：vLLM 映射与多卡冒烟记录（2026-08-02）

本次继续按 `kejiechen/CacheEOPD` 工作区推进，没有修改 `knhdu` 或 `yuanxigu` 的任何文件。
首先修复了一个实际的 Hydra 配置映射错误：`soft_kd_entropy_threshold` 和
`soft_kd_loss_coef` 原先被错误放在 `ActorConfig`，但 yaml 实际实例化的是
`PolicyLossConfig`，因此训练在配置实例化阶段报 `unexpected keyword argument`。现在两个
字段已归位到 `PolicyLossConfig`，本地和 kejiechen 远端均已同步。

vLLM 后端本身已完成 TP4 冒烟：在 GPU 1、2、4、5 上加载 Qwen3-0.6B，
`tensor_parallel_size=4`、`max_model_len=2048`，health check 和 completion 请求均成功。
远端 FlashInfer sampler 的 JIT 编译因系统缺少 `math.h` 失败，使用
`VLLM_USE_FLASHINFER_SAMPLER=0` 后正常；这属于环境 workaround，不是 vLLM 映射错误。

端到端 1-step 训练依次排除了以下配置问题：

1. 默认 `critic` 会读取不存在的 `~/models/deepseek-llm-7b-chat`；EOPD 应使用
   `algorithm.adv_estimator=on_policy`，从而关闭 critic 并在 policy loss 内计算 dummy
   advantage。
2. 当前 `RLHFDataset` 入口只接受 Parquet，已改用 `/home/kejiechen/data/gsm8k/*.parquet`；
   `val_files` 不能为 null，冒烟时使用 GSM8K test 文件。
3. 4 卡归一化下 global `ppo_mini_batch_size=2` 会变成 0，已改为 batch/mini-batch=8，
   每卡 micro-batch=1。
4. 远端 `flash-attn` wheel 需要 glibc 2.32，而系统版本不满足；训练侧改用
   `attn_implementation=sdpa`，vLLM 仍独立使用自己的 attention backend。

清理此前遗留的 kejiechen TP4 vLLM server 后，GPU 资源已恢复；不触碰 GPU0 上的 knhdu
进程。随后用干净 GPU 反复尝试 teacher Qwen3-4B 的 FSDP 初始化，分别测试了：
`fsdp2 + bf16`、`fsdp1 + bf16`、关闭 remove-padding、关闭 torch.compile 和参数 offload。
它们都能完成 Hydra 校验、数据集构建、Gloo 建组、teacher checkpoint 读取，并在
`Qwen3ForCausalLM` FSDP wrap 阶段发生原生 SIGSEGV，尚未进入 rollout 或 optimizer step。
因此目前不能声称已完成 vLLM 端到端训练，也不能把这次失败归因于 CacheEOPD loss 或
KV 映射；更准确的阻塞是 apex-llm 当前 torch/Ray/FSDP/Qwen3 teacher 初始化组合的原生
兼容性问题。正式 2048/4096 对照训练和评测尚未启动。

本阶段同时更新了 `scripts/eopd/run_eopd_vllm.sh`：默认开启 TP4、2048+2048 总长度、
`sdpa`、FlashInfer sampler workaround、on-policy advantage、teacher bf16/offload，
并保留 `MAX_PROMPT`、`MAX_RESPONSE`、`MAX_MODEL_LEN`、`TRAIN_BATCH_SIZE` 等环境变量，
可直接切换到 4096 总长度进行后续环境修复后的验证。

补充验证：将 teacher 临时替换为同目录的 Qwen3-1.7B，保持 4 卡、2048 总长度、SDPA、
FSDP1、bf16 和 offload 不变，仍在相同的 reference FSDP wrap 阶段 SIGSEGV。因此问题
与 Qwen3-4B 的显存规模无关，也不是单纯 OOM；更可能是当前远端 torch/Ray/FSDP 组合在
该 worker 初始化路径上的二进制或原生运行时兼容问题。该 1.7B 试验未进入训练 step，
不产生任何实验结果。

#### 阶段十五补充：修复 tokenizer 映射并完成 vLLM 端到端冒烟（2026-08-02）

上一轮卡住的直接原因不是 teacher FSDP，而是 `verl/workers/actor/dp_actor.py` 中用于
Soft-KD 调试日志的 tokenizer 代码：`ActorConfig` 本身没有 `model.path`，代码因此回退到
硬编码的 `Qwen/Qwen3-1.7B-Base`，在离线机器上反复访问 Hugging Face。修复为由
`FSDPWorker` 把已经按 `actor_rollout_ref.model.path` 离线加载的 tokenizer 显式传给
actor/ref，移除隐式联网和错误 fallback；`dp_actor.py`、`fsdp_workers.py` 均通过环境
Python 语法检查。

随后重跑 4 卡（GPU 1,2,4,5）、TP1×DP4、Qwen3-0.6B student / Qwen3-4B teacher、
GSM8K Parquet、prompt/response=256、总长度=512、batch=8、1 step 的 EOPD vLLM 冒烟，
日志为 `logs/train_vllm_smoke2.log`。本次确认完整链路成功：

- vLLM 从本地 student 路径启动，参数实际为 `max_model_len=512`、TP=1、
  `max_new_tokens=256`，没有 Qwen3-1.7B-Base 或 Hugging Face 网络请求。
- teacher Qwen3-4B 完成 FSDP 加载；`compute_ref_log_prob` 产出 teacher entropy/top-k
  信息，`update_policy` 进入 EOPD clipped reverse-KL + entropy-gated Soft-KD。
- 完成 optimizer update，日志出现 `training/global_step=1`、`soft_kd_loss=0.0607`、
  `soft_kd_token_ratio=0.0840`、`grad_norm=50.75`；vLLM rollout 和独立 validation
  也完成，单步吞吐约 18.23 token/s。

这次 validation 只用于链路冒烟（response 上限 256 且仅训练 1 step），得到的
`0.0083` 不具有实验比较意义。vLLM 仍需设置 `VLLM_USE_FLASHINFER_SAMPLER=0`；其余
`Failed to import Triton kernels` 是 vLLM 环境 warning，不影响本次训练完成。下一步先用
相同脚本将总长度提升到 2048，再评估显存和吞吐后决定是否运行 4096；正式 CacheEOPD 与
官方 EOPD 对照仍必须使用相同长度、步数和 student-only 评测协议。

### 阶段十五补充：2048 长度验证 + 4096 计划（2026-08-02，进行中）

运行环境（apex-llm）：micromamba `ta_opd_faithful`
（`/home/kejiechen/micromamba/envs/ta_opd_faithful/bin/python3`），
`PYTHONPATH=/home/kejiechen/CacheEOPD`，GPU 1/2/4/5（GPU0 被别人占满），
数据 `/home/kejiechen/data/gsm8k/train.parquet` + `test.parquet`，
student `Qwen3-0.6B`、teacher `Qwen3-4B`，checkpoint 与 Ray 临时目录改到 `/dev/shm`
（根文件系统 `/` 含 `/home`、`/tmp` 已 99% 满，仅 13G 余）。

- **2048 验证**（MAX_PROMPT=1024 / MAX_RESPONSE=1024，total 2048，10 步，batch 8）：
  vLLM `max_seq_len=2048` 正常启动、无 Hugging Face 联网，step1–9 全部正常完成。
  代表性指标：soft_kd_loss 0.110→0.063、rev_kl 0.636→0.545、grad_norm 22.7→41.0、
  throughput 48–63 tok/s、峰值显存聚合 ~23.4 GB（远未 OOM）。**结论：2048 总长度的
  vLLM EOPD 链路可跑通、不 OOM、指标正常。** step10 在收尾阶段卡死（>10 min，GPU 仍
  85% 但无新 step 日志）。
  - 根因：根文件系统 99% 满，Ray object-spill 目录 `/tmp/ray/session_*` 无法扩展，
    最后一步写大对象时阻塞；与之前多阶段 `/home` 满盘属同一类磁盘问题。
  - 修复：后续运行设 `RAY_TMPDIR=/dev/shm/ray`（`/dev/shm` 余 228G），checkpoint 也写
    `/dev/shm`；已清理陈旧的 ray session。
- **4096 验证**（MAX_PROMPT=2048 / MAX_RESPONSE=2048，total 4096 = 论文 response 上限、
  脚本默认）**已完成 10/10**：vLLM `max_seq_len=4096` 正常，10 步全部跑通、step10 正常
  收尾（此前 2048 卡死的位置），`/tmp` 满盘告警 **0 次**（`RAY_TMPDIR=/dev/shm` 修复生效）。
  代表性指标：soft_kd_loss 0.050→0.082、grad_norm 22.5→（稳定）、峰值显存聚合
  **31.7 GB**（2048 时为 23.4 GB，符合 2× 序列预期，远未 OOM）、末尾 GSM8K val
  `acc/mean@1 = 0.3139`（仅 10 步冒烟，无比较意义，但证明 val 链路可用）。总耗时 ~27 min /
  10 步（~38s/步 起步，后段因 4096 响应更长变慢）。**结论：论文 4096 长度在共享机上可跑通、
  不 OOM、指标正常，磁盘卡死问题已用 `/dev/shm` 解决。**

待补：
- [x] 4096 验证跑满 10 步并确认无 step 收尾卡死（`RAY_TMPDIR=/dev/shm` 生效）
- [x] 确定正式对照长度（倾向 4096 = 论文口径；共享机吞吐不足则用缩短 pilot 并标注）
- **官方 EOPD 300 步基线已启动（2026-08-02）**：按用户决定，先跑官方 EOPD（w/o C2C）
  作为 control。配置：4096 总长度、batch 8、save_freq 50、checkpoint 与 Ray temp 全部
  放在 `/dev/shm`（根文件系统满、且无任何持久可写大盘——`/ext0`/`/ext1`/`/mnt/tidalfs`
  均 Permission denied，仅 `/home/kejiechen` 13G 可写但已满）。已确认 step1–4 正常、
  ~37s/step、ETA ~3h、无早期 OOM、GSM8K val 每 50 步触发一次。
  ⚠️ `/dev/shm` 是内存盘，机器重启/ OOM-kill 即丢失；需事后把 ckpt 拷到持久存储
  （如 scp 回本地或共享盘）才能保留。CacheEOPD 侧仍待 `c2c_hf` 注入接入 vLLM 路径后再对照。
  - **step50 里程碑（2026-08-02）**：已完成 step 50/300，GSM8K val `acc/mean@1 = 0.3086`
    （30.86%，与 10 步冒烟的 0.3139 基本持平，符合 EOPD 早期平缓、随步数抬升的预期）；
    checkpoint `global_step_50`（含 optimizer，3.4G）已落本地 `_backups/eopd_baseline_300/`
    （rsync 保险验证可用；中途曾因 rsync 周期比 ckpt 写入早 2 分钟而显示空，强制补传后正常）。
    运行健康，~37s/step，ETA 含 6 次 val 约 4–5h。
## 阶段十六：模型配置纠偏——回到论文口径 1.7B-Base + 8B（2026-08-02）

**问题**：上述 300 步「官方 EOPD 基线」用的是 **Qwen3-0.6B 学生 + Qwen3-4B 教师**，
这是照抄阶段十五 smoke2 冒烟配置的结果，**并非 EOPD 论文口径**。经用户指出后核对
`Baselines/EOPD/EOPD复现.md`，论文 Table 2 那一行是 **Qwen3-1.7B-Base + Qwen3-8B**。

**该次错误配置的运行结果（作废，仅留存为负面记录）**：跑到 step 167/300 后终止。
GSM8K val **单调下降**：step50 = 0.3086 → step100 = 0.2669 → step150 = 0.2654。
`soft_kd_token_ratio ≈ 0.18`，即 **82% 的 token 走纯 PG 而非 KL 蒸馏**，蒸馏信号很弱，
叠加学生模型过小，acc 下降不意外。ckpt（step 50/100/150，共 11G）已备份到本地
`_backups/eopd_baseline_300/`。

**与官方口径的全部偏差（已修正）**：

| 项 | 官方 EOPD | 错误运行 | 现状 |
|---|---|---|---|
| 学生 | Qwen3-1.7B-Base | Qwen3-0.6B | 已下载 1.7B-Base |
| 教师 | Qwen3-8B | Qwen3-4B | 已下载 8B |
| `ref.topk_logits` | 16 | 32 | 脚本已参数化，默认改 16 |
| batch / mini-batch | 128 / 32 | 8 / 8 | 脚本已拆出 `MINI_BATCH_SIZE` |
| 训练数据 | MATH | GSM8K | 机器上已有 `/home/kejiechen/data/math`（7500 条） |
| τ / α | 0.8 / 1.0 | 0.8 / 1.0 | ✓ 本来就对 |

**环境准备**：
- `/home` 原仅剩 13G。经用户确认删除 pip 缓存 7.8G + `outputs/slime_opd` 5.0G +
  `Qwen3-1.7B`(Instruct) 3.8G + `Qwen3-1.7B_torch_dist` 3.3G，腾到 33G。
  保留 `ckpt_eopd_dense` 6.8G 与 dapo-math 数据集 2.0G。
- 模型经 `HF_ENDPOINT=https://hf-mirror.com` + `hf download` 拉取（apex 直连 HF 不通，
  hf-mirror 可用、modelscope API 404）：
  - `Qwen3-1.7B-Base` 3.3G，单文件 `model.safetensors`（无 index），
    `eos=151643`、`do_sample=false`、`max_pos=32768` → 确认是**真 Base 版**
    （对比 Instruct 版 `eos=151645`、`max_pos=40960`）。
  - `Qwen3-8B` 16G，5 分片，36 层 / hidden 4096 / kv_heads 8。
  - 下载后 `/home` 剩 14G。
- ⚠️ 机器上**没有** README 提到的官方 `eopd-baseline` 仓库（`/home/kejiechen/eopd-baseline`
  不存在）和 `eopd` conda 环境（只有 `ta_opd_faithful`）。因此仍走本地
  `CacheEOPD/verl` + `scripts/eopd/run_eopd_vllm.sh`，靠参数对齐论文口径。

**脚本改动**（`scripts/eopd/run_eopd_vllm.sh`，本地与远程已同步）：
新增 `MINI_BATCH_SIZE`（默认 = TRAIN_BATCH_SIZE）、`TOPK_LOGITS`（默认 16）、
`ENTROPY_THRESHOLD`（默认 0.8）、`SOFT_KD_COEF`（默认 1.0）四个可调环境变量，
把原先硬编码的 topk=32 改为默认 16。

**1.7B-Base + 8B 两步冒烟结果（2026-08-02，已完成 2/2）**
配置：MATH 数据、1024+1024、batch 8、`GPU_MEMORY_UTILIZATION=0.25`、4 卡（GPU1/2/4/5）。

| 指标 | 值 | 说明 |
|---|---|---|
| 是否 OOM | **否** | 峰值显存 21.2 → 22.8 GB / 48GB，余量充足 |
| step 1 耗时 | **54 s** | 训练本身很快 |
| 2 步总耗时 | 1:12:04 | ⚠️ 绝大部分是**末步 validation**（MATH test **5000 题** × 1024 token），非训练开销 |
| `soft_kd_token_ratio` | **0.2257** | 8B 教师（对比 4B 教师的 0.18） |
| `soft_kd_avg_kl` | **0.9466** | 4B 教师时为 0.4575，**KL 强度翻倍** |
| `ref_entropy` | mean 0.5913 / max 2.694 | 单批 `entropy>1.0` 达 **978/2048（47.8%）**，另批 846/2048 |

**结论**：
1. **显存和吞吐都不是瓶颈**（22.8G/48G，54s/step）。
2. **教师换 8B 后蒸馏信号显著增强**：`avg_kl` 0.458 → 0.947（翻倍），高熵 token 比例
   0.18 → 0.23。这正面印证了先前 0.6B+4B 跑出 acc 单调下降的一个重要成因是
   **蒸馏信号过弱**（82% token 退化为纯 PG）。
3. ⚠️ **validation 开销必须控制**：MATH test 全量 5000 题跑一次约 1 小时。正式 300 步
   若每 50 步 val 一次（6 次）将额外耗掉 ~6h。**正式运行前需把 val 子采样**
   （如取 500 题）或拉长 `test_freq`。

冒烟后已 `pkill` + `ray stop`，GPU1-5 全部释放。

---

## 阶段十七：C2C projector（1.7B-Base + 8B）预训练——调研完成，**暂缓执行**（2026-08-02）

按用户指示「先别跑，存 PROGRESS」，本节仅记录调研结论，**未启动任何训练**。

### 训练入口与配置

- **入口**：`Baselines/C2C/script/train/SFT_train.py`（全仓库唯一能产出
  `final/projector_0..27.pt` + `projector_N.json` 这种 rosetta 格式的脚本）。
  启动方式见 `Baselines/C2C/bash/train/sft_train.sh:12-24`：
  ```
  torchrun --nproc_per_node=N --master_port=29501 \
      script/train/SFT_train.py --config recipe/train_recipe/C2C_1.7+8.json
  ```
  CLI 参数极少（`SFT_train.py:871-875`），一切靠 config JSON；
  注意 `--output_dir` **无效**，会被 config 里的 `output.output_dir` 覆盖（`:895`）。
- ⚠️ **不要混用** `cache_eopd/train_projector.py`——那是自研线（CE/SFT 口径，产出
  `projector_final.pt` + `.weights`），**不产出 rosetta 格式**。
- **配置模板**：`Baselines/C2C/recipe/train_recipe/C2C_0.6+0.5.json`
  （机器上那份 `qwen3_0.6b+qwen3_4b_base_Fuser/config.json` 即由它派生）。
  改 4 处即可：`model.base_model` → `Qwen3-1.7B-Base`、`model.teacher_model` → `Qwen3-8B`、
  `output.output_dir`、`projector.params.anneal_steps`（须 ≈ total_steps，
  见 `SFT_train.py:1159-1161`；0.6+0.5 用 1929、0.6+4b 用 1953）。

### 关键结论：projector 张量形状与 0.6B+4B **完全一致**

`SFT_train.py:602-607` 显示维度是**全自动推导**，且投影的是 **per-head KV 维度**而非
hidden_dim：
```python
602: base_dim    = k_proj.out_features / base_model.config.num_key_value_heads
603: teacher_dim = k_proj.out_features / teacher_model.config.num_key_value_heads
618: num_projectors = slm_num_layers   # = 学生层数 28
```
Qwen3 全系 `head_dim=128, kv_heads=8`，所以 4B→8B 换教师后 `teacher_dim` 仍是 128，
0.6B→1.7B 换学生后 `base_dim` 仍是 128。**config 里 `hidden_dim`/`intermediate_dim`=1024
原封不动即可**，变的只是教师隐层语义，接口不变。

层映射 `mapping="last_aligned"` → `rosetta/train/model_utils.py:97-150` 的
`last_aligned_sources(28, 36, K=1)`：`offset = 36-28 = 8`，学生层 t ← 教师层 `8+t`
（`model_utils.py:134`）。与 0.6B+4B 时同为 28↔36，**映射行为完全相同**。
（`SFT_train.py:418` 的 `build_layer_mapping` 是遗留死代码，未被调用。）

### 三个已知坑的当前代码状态

| 坑 | 位置 | 状态 |
|---|---|---|
| KL reduction 量纲 | — | ✅ 官方路径不受影响（`SFT_train.py` 用 CE，无 KL）。⚠️ 但 `cache_eopd/eval_projector_kl.py:30` 仍 `from cache_eopd.train_projector import token_mean_kl`，该函数已删 → **该脚本现在 import 即崩** |
| save_projector 不存权重 | `rosetta/model/projector.py:1513-1523` 只序列化 `_init_args` 成 JSON | ✅ 官方训练脚本已规避：`SFT_train.py:1447-1449` 先 `torch.save(state_dict)` 再写 `.json` |
| **gate 默认关闭** | `rosetta/model/projector.py:1374-1375` `key/value_gate_logit = Parameter(0.0)`；推理硬门控 `:1479-1480` `(gate_logit > 0).float()` | ❌ **上游未修**。logit==0 判 False → **融合全关，静默失败**。训练期靠 Gumbel 噪声（`:1468-1476`）让梯度能流，需足够步数把 logit 推正；**短跑必翻车** |

### 执行前必须先决策的三件事

1. **语料规模**：官方 500k / 1929 步是 **0.5B 教师**的配方；换 8B 教师后单步开销高一个
   量级。需决定 50k（~200 步，先验证链路）/ 150k（~600 步）/ 500k（严格对齐）。
2. **gate 初值**：是否把 `projector.py:1374-1375` 的 0.0 改成正值（如 1.0），
   或沿用自研线的 `--gate-init 1.0 + --gate-lr-mult 20.0`（`train_projector.py:98-109`）。
   不改则需在训练后检查有多少层 `gate_logit > 0`。
3. **语料存放**：`rosetta/train/dataset_adapters.py:1404` **硬编码**
   `load_dataset("teknium/OpenHermes-2.5")`，**无 `data_path` 参数**，只能靠
   `HF_HOME`/`HF_ENDPOINT`。数据集约 1.6G，`/home` 现剩 14G。
   （若要用本地数据须改走 `LLMGeneratedChatDataset`，`dataset_adapters.py:1170`，
   它接受 `kwargs.data_path`，模板见 `recipe/train_recipe/include_response.json:56-64`。）

### 其他注意事项

- 服务器上**有** `/home/kejiechen/CacheEOPD/rosetta` 库，但**没有 `SFT_train.py`**，
  执行前需从本地 `Baselines/C2C/` scp 过去。
- `dataset_adapters.py:1427` 的长度过滤器硬编码用 `Qwen/Qwen3-0.6B` 的 tokenizer 计数；
  Qwen3 同族词表相同，1.7B-Base 下无影响，但仍需能下到该 tokenizer。
- 产物目录里是 `projector_config.json`（`SFT_train.py:1449`），而非 HF 发布版的
  `aggregator_config.json`——本仓库无代码生成后者，下游加载器需对应。

- [ ] **关键缺口**：CacheEOPD 的 `c2c_hf` teacher-KV 注入目前是 HF rollout，尚未接入本
    vLLM EOPD 路径。正式 CacheEOPD vs 官方 EOPD 对照前，需先实现「vLLM rollout 注入
    teacher KV」或改走「HF rollout + EOPD loss」的等价入口；两者必须同长度/步数/
    student-only 评测协议，否则不可比。
- [ ] 正式对照：官方 EOPD（w/o C2C）vs CacheEOPD（w/ C2C），300 步，6 基准 student-only 评测。
- [ ] **C2C projector 配对缺口**：现有 fuser 只有 `qwen3_0.6b+qwen3_4b_base_Fuser` 一对
  （config 写死 `base_model=Qwen3-0.6B` / `teacher_model=Qwen3-4B-Base`，28 层 projector
  各 37MB）。改用 1.7B-Base + 8B 后，CacheEOPD 实验臂**没有可用 projector，必须重新
  预训练**（OpenHermes500k 语料）。注意已知三陷阱：KL reduction 量纲、`save_projector`
  不存权重、gate 默认关闭——三者都是静默失败。

---

## 0804 后续研究路线：从随机注入到自主选择

### 当前判断

已有 mixed 结果说明：训练时随机注入 Teacher KV 可能提高早期收敛效率，最终独立评测
通常有约 0.5--2% 的局部收益。但当前证据还不能说明学生学会了自主判断何时需要 Teacher
KV，也不能确认最终能力上限稳定提升。

当前 `train_student_distill.py` 中的 mixed 是 **micro-batch 级别随机选择**：每个
micro-batch 以固定概率使用 Teacher KV 或 Student KV（见 `:367`）。该脚本使用预先生成的
teacher trajectory 做 response CE，尚未让学生自主生成不同 trajectory，也没有根据最终
reward 计算 fuse 策略的 advantage。因此，后续 trajectory-level / advantage 实验应以
`train_eopd_hf.py` 的 on-policy EOPD 逻辑为基础；其中已有 student rollout、teacher
signals 和 `advantage = teacher_log_probs - old_log_probs`（见 `:159`）。

### 第一阶段：巩固 mixed 的结论

在扩大模型前，固定数据、训练步数、评测集和 checkpoint 间隔，至少运行以下多 seed 对照：

```text
plain / EOPD / mixed(p=0.25) / mixed(p=0.5) / mixed(p=0.75) / anneal
```

每个 seed 在 step 50、100、150、200、250、300 保存，并使用 student-only、无 Teacher
KV、无 projector 的独立评测。记录：

- 固定 step 的均值、标准差和 paired seed 差值；
- 最佳 checkpoint 的均值，而不是只报告单个最佳 seed；
- 达到指定准确率所需的 step（time-to-threshold）；
- 50--300 步准确率曲线下面积（AUC）；
- 题目级别的 plain 正确、fused 正确、plain 正确但 fused 错误。

只有在多 seed 的均值和 paired 差值稳定为正时，才能把结论表述为“提升”；否则应表述为
“可能改善收敛速度或改变最佳 checkpoint 区间”。

### 第二阶段：coverage / hit ratio 分析

coverage 不能只统计预设的 `fused_prob`，因为该数值本身就是实验设定。应同时记录：

1. **injection rate**：实际使用 Teacher KV 的 micro-batch 数占比；
2. **active KV ratio**：layer-token-head 中实际产生有效 projector 增量的比例；
3. **beneficial hit ratio**：同一题目上，fused 相比 plain 从错变对或降低 next-token CE
   的比例；
4. **harmful hit ratio**：plain 正确而 fused 错误，或 fused 提高 CE 的比例。

需要按题目、token 位置、student layer、KV head 以及 teacher/student divergence 分组，
观察 Teacher KV 是否只在少数题目或少数 token 上有效。`C2CProjector` 已经计算了每个
token/head 的 scalar 权重，但全局 key/value gate 仍是 layer 级标量（见
`rosetta/model/projector.py:1464`）；当前只保存最后一次 forward 的诊断值，因此需要新增
逐层聚合日志或单独的 coverage 分析脚本。

### 第三阶段：trajectory-level fuse + advantage

对同一个 prompt 让当前 student 产生多个候选 trajectory：

```text
A：完全不使用 Teacher KV
B：使用 Teacher KV
C：只使用部分 Teacher KV
```

对同一题目比较最终 reward 或 EOPD token-level signal，计算：

```text
advantage(strategy) = reward(strategy) - same-prompt baseline reward
```

第一版建议先实现 trajectory-level bandit，不要立即训练复杂的 token selector：

- fused trajectory 的 advantage 为正时，提高该类题目使用 fused 的概率；
- advantage 为负时，降低使用概率；
- 使用同一 prompt、相同评测规则和 paired rollout，减少题目难度与采样噪声影响。

该实验要回答的问题是：学生能否学会判断“这道题是否需要老师帮助”，而不仅是随机接受
老师帮助。评测阶段仍然必须完全移除 Teacher KV 和 projector。

### 第四阶段：token importance 反事实探针

不要一开始就直接训练 token selector，先离线比较每个 token 的三种重要性：

1. **representation divergence**

   `D_t = ||projected_teacher_KV_t - student_KV_t||`

   表示 Teacher 与 Student 的状态差异，但不代表注入一定有益。

2. **当前 token 信息收益**

   `G_t = CE_plain(t) - CE_fused(t)`

   大于零表示 Teacher KV 降低了当前 token 的预测损失。

3. **forward-looking effect**

   `F_t = sum_{u=t}^{t+H}(CE_plain(u) - CE_fused(u))`

   衡量某个 token 的注入是否改善后续多个 token，而不只是当前一步。

需要比较 `D_t`、`G_t`、`F_t` 与最终答题正确率的相关性，从而验证“有方向的 divergence”、
“information loss 更少”和“forward-looking effect”哪个比单纯 KV L2 距离更有价值。

### 第五阶段：selective KV 训练

在反事实探针确认有效指标后，再在 `fused_kv.py` 中加入 token-level fusion mask，使每个
prefix token 可以选择：

```text
靠近 Teacher / 保持 Student 自主状态 / 抑制 Teacher 信号
```

训练初期使用 soft mask，后期再逐渐变为 hard selection。selector 的监督信号应来自
反事实 reward 或 forward-looking gain，并对标签计算停止梯度，避免 selector 通过改变
评分方式作弊。可选目标为：

```text
L = L_EOPD + λ1 * L_selective_KD + λ2 * L_sparsity
```

推荐新增或修改的文件：

- `cache_eopd/train_eopd_hf.py`：作为 on-policy trajectory / advantage 基础；
- `cache_eopd/fused_kv.py`：增加 token-level mask、KV delta 和诊断输出；
- `cache_eopd/probe_token_importance.py`：离线计算 D_t、G_t、F_t；
- `cache_eopd/coverage.py`：汇总 injection、beneficial hit 和 harmful hit；
- `cache_eopd/train_cache_policy.py`：训练 trajectory-level 或 token-level selector。

### 推荐执行顺序

```text
多 seed 复现
→ coverage / beneficial hit 分析
→ trajectory-level advantage 选择
→ token-level 反事实 importance
→ selective KV 训练
→ 更大模型和更大数据实验
```

最终阶段性目标不是证明“Teacher KV 越多越好”，而是证明：Teacher KV 只在特定题目、
特定 token 或特定状态下有益，并且学生可以通过 advantage 学会主动请求或拒绝这部分帮助。

### 阶段十九：官方 projector 多 seed 对照启动（2026-08-04，进行中）

用户要求验证第一阶段结论：固定三组 seed，比较 plain、EOPD、mixed，并比较 anneal 的
线性、二次和根号调度。为避免 prior-v7 projector 成为变量，本轮统一使用官方
`qwen3_0.6b+qwen3_4b_base_Fuser/final`，对应 Qwen3-0.6B student → Qwen3-4B teacher。

固定配置：

```text
seeds = 41717, 41718, 41719
steps = 300；checkpoint = 每 50 步
plain / EOPD / mixed(p=0.5)
anneal: linear / quadratic / sqrt，Teacher KV 概率 1.0 -> 0.0
student-only 独立评测：GSM8K test 前 500 题，max_new_tokens=512
```

调度定义为概率插值进度 `f(s)`：linear=`s`、quadratic=`s²`、sqrt=`sqrt(s)`，其中
`s=step/300`。因此 quadratic 在前期保留 Teacher KV 更久，sqrt 在前期更快撤掉 Teacher
KV。训练入口新增 `--anneal-schedule {linear,quadratic,sqrt}`。

本轮采用已有 1368 条 GSM8K teacher trajectory 训练，holdout 前 64 条不训练；plain、
mixed、anneal 共用同一数据和 student-only 评测口径。EOPD 使用当前 HF EOPD runner 的
on-policy loss（clipped policy-gradient + entropy-gated top-16 soft-KD），同样训练 300
steps、每 50 步保存。

500 题评测集为 `/home/kejiechen/CacheEOPD/data/gsm8k_cot_test_500.jsonl`，由缓存的
GSM8K test split 固定取前 500 题生成。完整 checkpoint 暂存于：
`/dev/shm/cacheeopd/multiseed_official_20260804`；日志和逐 checkpoint 评测结果保存于：
`/home/kejiechen/CacheEOPD/logs/multiseed_official_20260804`。批量启动器为
`cache_eopd/run_multiseed_official.sh`。

本轮启动前，官方 Qwen3-1.7B→8B EOPD 基线在 step200 被用户主动停止。为保证以后可恢复，
已将其完整 step200（模型、optimizer、extra state、环境导出、原始启动命令和数据校验）
持久化到 `/home/kejiechen/CacheEOPD/persistent/eopd_1p7_8b_300_step200`。恢复时使用：

```text
trainer.resume_mode=resume_path
trainer.resume_from_path=/home/kejiechen/CacheEOPD/persistent/eopd_1p7_8b_300_step200
```

当前状态：批量 runner PID `3187582` 已启动，正在运行 `plain seed=41717`，训练已正常到
step40，尚未出现 OOM；结果完成后继续追加三 seed 均值、标准差、time-to-threshold、AUC
以及各 checkpoint 的 500 题准确率。

### 阶段二十：vLLM V1 融合接入第一阶段（2026-08-04，代码已写，远端冒烟待恢复）

本阶段开始把此前只在 HF `past_key_values` 上验证的 C2C 融合迁移到 vLLM。关键约束是：
vLLM 的公开生成接口只接收 token ids/embeddings，不能把 HF `DynamicCache` 直接塞进
`SamplingParams`；而且如果先让 vLLM 做普通 prefill，再写回 KV，prefill 已经覆盖了注入
内容，首个 response token 也可能已经采样。因此第一版采用 vLLM V1 KV-transfer 的正确顺序：

```text
HF teacher/student + C2C projector
    → student-shaped fused KV packet
    → request extra_args.kv_transfer_params
    → scheduler 把前 L-1 个 prompt token 标记为已计算
    → connector 写入前 L-1 个 paged KV blocks
    → vLLM 计算最后一个 prompt token，再从 fused prompt KV decode
```

新增代码：

- `cache_eopd/vllm_kv_packet.py`：把 `FusedKVBuilder.build()` 的 `DynamicCache` 转成单请求
  packet，保存输入 token、prompt 长度、逐层 key/value，并提供 `build_packet_from_models()`。
- `cache_eopd/prepare_vllm_kv_packet.py`：独立 CLI，可加载官方 fuser 或本项目 projector，
  生成 packet 并打印 JSON request metadata。
- `cache_eopd/vllm_kv_connector.py`：vLLM V1 `KVConnectorBase_V1` 实现。scheduler 侧报告
  packet 覆盖的前 `L-1` 个 prompt token 数，worker 侧按 block id 和 block offset 把
  `[num_kv_heads, prompt_len, head_dim]` 写入 vLLM 的
  `[2, num_blocks, block_size, num_kv_heads, head_dim]` cache layout。
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`：启用 `rollout.c2c` 时配置
  connector，传递 `SamplingParams.extra_args.kv_transfer_params`，并在 packet 缺失时硬失败。
- `verl/trainer/config/rollout/rollout.yaml`：加入 vLLM C2C 配置默认值。
- `cache_eopd/smoke_test_vllm_connector.py`：不启动模型的 CPU 冒烟，验证 block offset、
  层排序、`L-1` 注入边界和 token hash。
- `cache_eopd/smoke_test_vllm_engine.py`：真实 vLLM engine smoke，支持 self-KV 或外部
  teacher-projector packet。

这一阶段还没有宣称端到端 EOPD 已完成：packet 目前由独立入口生成，尚未自动绑定每轮
EOPD 的 student 权重同步和 rollout request。这样可以先独立验证 vLLM 的 block 映射和
首 token 逻辑，再接入在线 provider；在此之前不会把普通 vLLM 结果冒充融合结果。

本地静态检查已通过：5 个新增/修改 Python 文件 AST 解析通过，`run_eopd_vllm.sh` 通过
`bash -n`，`git diff --check` 通过。当前环境没有安装 torch，无法在本地执行 tensor smoke；
本轮多次尝试连接 `apex-llm` 均在 SSH 握手阶段被关闭，因此 vLLM 0.13 的真实 import、
connector factory 注册和 GPU paged-cache smoke 尚未执行，恢复 SSH 后优先运行
`PYTHONPATH=. python -m cache_eopd.smoke_test_vllm_connector`，再做单请求 vLLM decode。

远端验证更新（2026-08-04 17:51）：SSH 已恢复。`apex-llm` 的 vLLM 版本确认为 0.13.0，
`CacheEOPDConnector` 在真实 vLLM 环境中 import 成功，`__abstractmethods__` 为空；通过
`KVConnectorFactory._get_connector_class_with_compat()` 动态加载成功，CLI JSON 也能解析为
`KVTransferConfig(kv_connector=CacheEOPDConnector, kv_role=kv_both)`。远端执行
`PYTHONPATH=. python -m cache_eopd.smoke_test_vllm_connector` 输出 `VLLM_CONNECTOR_OK`，
并完成远端 `py_compile`。

根据 vLLM 官方 `ExampleConnector` 接口又修正了三点：`get_num_new_matched_tokens` 返回
`(tokens, False)`；外部 token 数向 block size 向下对齐；metadata 继承
`KVConnectorMetadata`，connector 构造接收 `kv_cache_config`。因此 prompt 长度为 `L` 时，
当前首版实际注入 `floor((L-1)/block_size)*block_size` 个 KV，剩余尾部由 vLLM student
正常 prefill，避免错误的首 token logits。GPU engine 尚未启动，原因是当前 GPU0 已被 vLLM
服务占用、GPU1/2/4/5 仍有其他实验任务；下一步使用空闲 GPU 做单请求 engine smoke。

GPU engine smoke 更新（2026-08-04 17:56）：使用 GPU1 和 Qwen3-0.6B 做了真实 vLLM
端到端单请求测试。脚本先用 HF student 生成 self-KV packet，再启动 vLLM 0.13.0，
通过 `KVTransferConfig` 加载 `CacheEOPDConnector`，请求携带
`SamplingParams.extra_args.kv_transfer_params`，最终输出 `VLLM_ENGINE_OK`。日志确认
factory 创建 connector、KV cache 初始化完成且请求成功生成；测试进程已退出，GPU1 显存
恢复到约 2.4GB。由此证明 vLLM scheduler → connector metadata → paged KV → decode
的第一阶段链路已真实跑通。该 smoke 使用 self-KV 只验证传输/映射，不代表 teacher/projector
融合效果；下一阶段仍需把每轮 EOPD student 权重下的 teacher+projector packet 自动生成
接入 rollout request。

随后使用官方 Qwen3-0.6B→Qwen3-4B fuser 在 GPU1 生成真实 fused packet：28 层，prompt
长度 10，单层 KV 形状 `(8, 10, 128)`。将该 packet 直接交给 Qwen3-0.6B 的 vLLM 0.13.0
engine 后同样输出 `VLLM_ENGINE_OK`。这一步确认的已经是 teacher→official projector→packet
→vLLM paged cache→decode 链路；但仍是单请求静态 packet，尚未证明 EOPD 每轮更新 student
权重后能自动、低开销地在线生成 packet。

### 阶段二十一：mixed probability ablation（2026-08-04 21:16--22:57，已完成）

用户要求暂不完成 seed41719，并在已经完成的 seed41717/41718 上补做 mixed 的两档融合
概率。原多 seed 调度器在收到请求前已经于 21:06:59 启动了 seed41719 的 plain、EOPD、
mixed-0.5 和 anneal-linear；已核对 PID 后仅终止这四个 seed41719 任务及其调度器，未触碰
其他用户进程或已完成结果，也阻止了后续 seed41719 的 quadratic/sqrt 启动。

新的独立 runner：`cache_eopd/run_mixed_prob_ablation.sh`。四个任务于 21:16:52 并行启动：

```text
GPU1: mixed p=0.25, seed41717
GPU2: mixed p=0.75, seed41717
GPU4: mixed p=0.25, seed41718
GPU5: mixed p=0.75, seed41718
```

每个任务训练 300 steps，每 50 steps 保存 checkpoint，并在同一 GPU 上对固定 500 题做
student-only 独立评测。结果目录为 `/dev/shm/cacheeopd/mixed_prob_ablation_20260804`，
日志目录为 `/home/kejiechen/CacheEOPD/logs/mixed_prob_ablation_20260804`；启动时四项状态
均已写入 `status.tsv`。seed41719 当前没有残留进程。

本轮最终状态：四个任务均于 22:50--22:57 完成，24 个 checkpoint 均通过固定 500 题
student-only 独立评测；GPU1/2/4/5 已释放。两 seed 的准确率（seed41717 / seed41718）如下：

| step | mixed p=0.25 | mixed p=0.75 |
|---:|---:|---:|
| 50  | 332/500 (66.4%) / 329/500 (65.8%) | 318/500 (63.6%) / 324/500 (64.8%) |
| 100 | 333/500 (66.6%) / 326/500 (65.2%) | 329/500 (65.8%) / 313/500 (62.6%) |
| 150 | 309/500 (61.8%) / 327/500 (65.4%) | 334/500 (66.8%) / 308/500 (61.6%) |
| 200 | 328/500 (65.6%) / 332/500 (66.4%) | 313/500 (62.6%) / 325/500 (65.0%) |
| 250 | 329/500 (65.8%) / 333/500 (66.6%) | 334/500 (66.8%) / 321/500 (64.2%) |
| 300 | 323/500 (64.6%) / 326/500 (65.2%) | 301/500 (60.2%) / 314/500 (62.8%) |

按两 seed 平均，p=0.25 在 step50/100/150/200/250/300 分别为
66.1%/65.9%/63.6%/66.0%/66.2%/64.9%；p=0.75 分别为
64.2%/64.2%/64.2%/63.8%/65.5%/61.5%。此前 p=0.5 的对应平均为
66.5%/64.3%/64.4%/65.9%/65.7%/64.8%。因此 p=0.25 整体最接近且在 step250
略高于 p=0.5，但没有超过 p=0.5 的最佳 step50；p=0.75 明显更不稳定，step300
下降尤其明显。当前证据支持继续优先研究中低概率 mixed，而不支持高概率 p=0.75。

### 阶段二十二：CacheEOPD 三策略同口径复现实验（2026-08-04 23:21，已停止并作废）

用户要求在 seed41719 统一实验前，先对已完成的 seed41717/41718 做纯 CacheEOPD、
mixed 和 anneal 三种策略的同条件对照。三种策略均使用官方 Qwen3-0.6B→Qwen3-4B
projector、同一 1304 条 teacher trajectory、学生 Qwen3-0.6B、300 steps、grad accumulation
8、学习率 1e-5、每 50 steps 保存，并对同一固定 500 题做 student-only 独立评测：

- `fused`：每个训练 microbatch 都注入 teacher KV；
- `mixed`：`fused_prob=0.5`；
- `anneal`：线性从注入概率 1.0 退火到 0.0，退火 300 steps。

10 步冒烟已全部通过：fused、mixed、linear anneal 均完成 step1--10，holdout plain CE
有限且正常保存 step5/10。最初 smoke 暴露的是 runner 中旧的 fuser 路径错误，已改用
kejiechen 工作区实际存在的官方 fuser；未修改 knhdu 或 yuanxigu 的内容。

曾启动 runner `cache_eopd/run_cacheeopd_three_methods.sh`，但在正式训练开始后立即停止：

```text
GPU1: fused seed41717    GPU2: mixed p=0.5 seed41717
GPU4: anneal linear seed41717    GPU5: fused seed41718
```

该 runner 实际调用的是 `train_student_distill.py`，属于离线 teacher trajectory 的学生 CE，
并没有 student on-policy rollout、teacher log-prob/entropy 检测和 EOPD clipped reverse-KL
+ entropy-gated top-k forward-KL。因此这批任务不符合本实验定义，已终止且不计入结果；没有
保留或使用其中的 checkpoint。seed41719 仍未启动。

### 阶段二十三：EOPD+C2C 正确训练入口（2026-08-05，开发中）

新增 `cache_eopd/train_eopd_cacheeopd_hf.py`：以 `train_eopd_hf.py` 的 EOPD loss 为基线，
仅替换 rollout：fused 分支先对同一 student prompt 做 teacher/student prefix forward，
经官方 projector 融合成 student 维度 KV，再由 student 进行采样 rollout；并记录该 fused
行为策略的 old log-prob。随后 teacher 对 student 生成的完整 sequence 计算 log-prob、entropy
和 top-k 分布，使用原 EOPD 的 clipped reverse-KL policy-gradient 与 entropy-gated top-k
forward-KL 更新 student。mixed 以概率决定每个 rollout 是否注入，linear anneal 使概率从
1.0 退火到 0.0；三者只改变 rollout 引导，不改变 EOPD 检测与 loss。

正确 smoke 结果：fused、mixed、linear anneal 均完成 3 steps；fused 日志每步均为
`fused_rollout=true`，mixed 在概率 0.5 下出现 fused/non-fused 两类 rollout，anneal 的
概率记录为 `1.0 → 0.667 → 0.333`。三者均产生非零 `pg_loss`、`soft_kd_loss`、teacher
entropy 和 `soft_kd_token_ratio`，证明 C2C rollout 与 EOPD loss 已接通。

正式 runner `cache_eopd/run_correct_eopd_cacheeopd.sh` 已于 2026-08-05 00:55:51 启动：
两 seed 均测试 fused、mixed p=0.5、linear anneal，300 steps，EOPD 参数为 max response
384、lr=1e-6、top-k=16、entropy threshold=0.8、soft-KD coef=1.0、每 50 步保存；全部
checkpoint 使用固定 500 题 student-only 评测。第一波 GPU1/2/4/5 分别运行
fused-41717、mixed-41717、anneal-41717、fused-41718；第一波结束后 GPU1/2 运行
mixed-41718、anneal-41718。seed41719 仍未启动。

用户已授权自动完成后续流程。已预置并同步 `cache_eopd/watch_and_run_seed41719.sh`，它等待
本阶段 status 出现 `finished_at` 后自动启动 `run_seed41719_all.sh`。seed41719 的全套条件为
plain SFT、原始 EOPD、CacheEOPD fused、CacheEOPD mixed p=0.5、CacheEOPD linear anneal，
均为 300 steps、每 50 步保存、固定 500 题 student-only 评测；当前 watcher 已启动但尚未
触发，避免与当前两 seed 任务竞争 GPU。

进度更新（2026-08-05 09:10）：正确 EOPD+C2C 训练的其余 30 个 500 题独立评测文件已完成。
当前可用结果如下（列顺序为 step50/100/150/200/250/300）：

| strategy | seed41717 | seed41718 |
|---|---:|---:|
| fused | 306/318/305/314/313/312 | 299/308/312/305/297/314 |
| mixed p=0.5 | 306/307/303/306/319/321 | 304/311/320/317/314/313 |
| linear anneal | 303/300/311/309/318/309 | 待修复 |

每项分母均为 500。两 seed 中 fused 的平均准确率序列为
60.5%/62.6%/61.7%/61.9%/61.0%/62.6%，mixed p=0.5 的平均序列为
61.0%/61.8%/62.3%/62.3%/63.3%/63.4%。anneal seed41718 已完成训练到 step300，
但保存最终 checkpoint 时遇到 `/dev/shm` 磁盘满，未产生可评测的 step300 文件；现已清理
我方旧 smoke/作废 checkpoint，repair 任务从头重跑，目前约 step10。seed41719 的第一次
自动启动因 runner 局部变量 bug 和磁盘满失败，状态已备份；修复后将在 anneal41718 repair
和评测结束后重新启动，不计入当前结果。
当前与原始 EOPD 的同 500 题对照（EOPD 两 seed 平均）为
62.0%/63.1%/62.5%/61.8%/64.2%/64.2%。因此 fused 的对应平均
60.5%/62.6%/61.7%/61.9%/61.0%/62.6%，在 step250 的差距最大（-3.2pp）；
mixed p=0.5 为 61.0%/61.8%/62.3%/62.3%/63.3%/63.4%，在 step200 反而高
0.5pp，step300 低 0.8pp。当前证据是“纯 fused 不如 EOPD，mixed 接近但尚未超过”，
而不是所有 CacheEOPD 都远远不如 EOPD。此前 64%--66% 的 mixed p=0.25/0.75 结果
属于离线 teacher-trajectory SFT，不是本阶段正确 EOPD+C2C 协议，不能与这里直接混比。

### 阶段二十四：SFT mixed 与 EOPD+C2C 融合细节复核（2026-08-05）

对照 `train_student_distill.py`、`train_eopd_cacheeopd_hf.py`、`fused_kv.py` 和官方
`rosetta/model/projector.py` 后，未发现当前正式 HF EOPD+C2C 训练中的层索引或 KV 形状
映射错误：Qwen3-0.6B 学生 28 层逐层使用 `projector_0..27`，每个学生层 `i` 对应
Qwen3-4B 教师层 `8+i`，即 `last_aligned`，与官方 C2C 配置一致；teacher/student 的
KV head 数和 head dimension 也分别从模型配置自动读取。

门控需要区分两层含义：`mixed/anneal` 的概率门控决定本次 rollout 是否使用 fused KV；
官方 projector 内部的 `key_gate_logit/value_gate_logit` 是每个 projector 的两个层级
开关，eval 时按 `logit > 0` 硬门控，另外还有按 token、KV head 计算的 sigmoid scalar
权重。两版都冻结官方 projector 并置于 eval，因此使用的是同一套硬门控；此前检查到
官方 fuser 的 key gate 为 27/28 层开启、value gate 为 28/28 层开启，而不是训练时每个
token 重新随机开关。

两版的 KV 边界也已对齐：只融合 prompt 的前 `L-1` 个 token，最后一个 prompt token
保留学生自身 KV，并在 decode 首步重新喂入。此前新版 EOPD 有一个真正的上下文不一致：
fused rollout 的 old log-prob 来自 fused prefix，但 current log-prob 曾经用 plain full
forward 计算；现已改为 `student_fused_response_logits`，使 old/current 都在同一个 fused
prefix 下计算。

需要明确的是，SFT mixed 与 EOPD mixed 不是同一个 loss：旧版按 micro-batch 决定是否
融合并对 teacher trajectory 做 response CE；新版按 rollout 决定是否融合，随后仍对学生
自己生成的 response 使用 EOPD 的 clipped reverse-KL 和 entropy-gated top-k forward-KL。
这是训练目标的有意差异，不是 projector 映射差异。

另外修正了 `c2c_hf_rollout.py` 的潜在默认值：当 vLLM/HF 适配层未显式传 mapping 时，现
默认使用官方 fuser 的 `last_aligned`，避免它意外退回 `relative_depth`。正式独立脚本
`train_eopd_cacheeopd_hf.py` 本身已经显式指定 `last_aligned`。

### 阶段二十五：修复后优先重跑 mixed（2026-08-05）

用户决定优先验证表现最稳定的 `mixed p=0.5`。远端原有 v2 任务已确认加载修复后的
`train_eopd_cacheeopd_hf.py`：`mixed-41717` 在 GPU2 运行，日志已出现 fused 与 plain
两类 rollout；`fused-41717`、`anneal-41717`、`fused-41718` 继续运行。为不等待第一波
结束，`mixed-41718` 已提前调度到 GPU3，使用同一官方 projector、300 steps、每 50 步
保存和 500 题 student-only 评测协议。

第一次手动启动因遗漏 `PYTHONPATH=/home/kejiechen/CacheEOPD` 立即退出，未生成 checkpoint；
已记录为启动错误并修正，第二次启动正常进入模型加载和 GPU3 训练。旧 runner 调度壳已停止，
不影响正在运行的训练进程，以避免后续重复启动 `mixed-41718`。

`mixed-41719` 原计划继续抢占空闲 GPU0，但检查发现 GPU0 被其他用户进程 PID 3771587
占用约 46.9GB；任务在模型加载阶段 OOM 后立即退出，未生成 checkpoint，未触碰该进程。
待自有 GPU 释放后再启动 seed41719。

09:40 状态：`mixed-41717=step110`、`mixed-41718=step20`；同一批次的
`fused-41717=step120`、`anneal-41717=step120`、`fused-41718=step130`。GPU1/2/3/4/5
均为 kejiechen 自有训练，GPU0 仍由其他用户占用。按当前吞吐，单个 300-step 训练约
35--45 分钟，训练完成后六个 500 题 checkpoint 评测预计再需 30--90 分钟；三个 mixed
seed 的完整结果预计约 2--3 小时。远端已启动十分钟状态记录器。

09:53 状态：`mixed-41717=step210`、`mixed-41718=step120`；`fused-41717=step220`、
`anneal-41717=step220`、`fused-41718=step240`。mixed 两个训练日志均持续产生非零
`pg_loss`/`soft_kd_loss`，没有 NaN 或进程异常；当前已保存 mixed-41717 的 step50/100/150/200，
mixed-41718 的 step50/100。训练尚未完成，因此暂时没有新的准确率结果。

10:02 状态：`anneal-41717` 已训练完成并保存到 step300，`fused-41718` 也已完成 300 步；
`fused-41717=step290`、`mixed-41717=step280`、`mixed-41718=step190` 仍在训练。GPU4
和 GPU5 已转入 step50 的 500 题 student-only 评测，GPU1/2/3 继续训练；当前评测结果
文件尚未完成，因此暂时仍无新的准确率数字。

10:12 状态：`fused-41717`、`mixed-41717`、`anneal-41717` 和 `fused-41718` 均已完成
step300；`mixed-41718=step270`，预计很快完成。GPU1/2/4/5 正在并行评测各自的 step50
checkpoint，GPU3 继续训练 mixed-41718；目前尚未产生完整 JSONL 评测文件，准确率仍待评测
进程完成后统计。

10:44 状态：五个训练条件均已完成 300 步，当前 5 个 GPU 评测进程运行中，其中 GPU3
刚开始补跑 `mixed-41718` 的 step50--300 评测。已完成 11 个 500 题评测文件，初步计数为：
`anneal-41717` step50/100/150 = 310/304/308，`fused-41717` = 309/295/305，
`fused-41718` = 307/314/296，`mixed-41717` step50/100 = 305/311。它们只是部分
checkpoint 结果，不能替代完整六步汇总。

10:51 状态：已完成 12 个 checkpoint 评测文件；新增 `mixed-41717 step150=316/500`。
当前部分序列为：`fused-41717` step50/100/150=`309/295/305`，`mixed-41717`=`305/311/316`，
`anneal-41717`=`310/304/308`，`fused-41718`=`307/314/296`。GPU1/2/4/5 继续评测已有
checkpoint，GPU3 正在补跑 `mixed-41718`；其余 step200--300 仍未完成，不能据此下最终结论。

11:19 状态：已完成 22/30 个 checkpoint 评测。当前新增结果：`anneal-41717` step200/250
为 `320/314`，`fused-41717` step200/250 为 `304/303`，`fused-41718` step200/250 为
`300/308`，`mixed-41717` step200/250 为 `312/317`，`mixed-41718` step100=`303`。
当前仍有 6 个评测进程运行，剩余主要是四个 step300 以及 mixed-41718 的 step150--300。

11:23 查询：step300 评测文件尚未完成；`fused-41717`、`fused-41718`、`anneal-41717`、
`mixed-41717` 的 step300 正在 GPU1/2/4/5 并行评测。`mixed-41718` 的 step150 评测在
GPU3 运行，约已处理 400/500 个样本，完成后继续 step200--300。

GPU2 实时查询：`mixed-41717 step300` 已处理 `440/500` 题，暂时正确 `269` 题；进程
仍在运行，当前中间准确率约 61.1%，尚不是最终结果。

11:31 mixed 更新：`mixed-41717` 六个 checkpoint 已完成，准确率为
`305/311/316/312/317/310`，平均 `311.8/500=62.37%`；`mixed-41718` 已完成
step50/100/150=`305/303/307`，step200--300 仍在 GPU3 评测。旧 runner 曾错误重启
一个 GPU1 的重复 mixed-41718 训练，但它只运行到 step20、尚未保存 checkpoint，已停止，
不会覆盖 GPU3 版本的已保存 checkpoint。

### 阶段二十六：rollout 直接收益诊断（2026-08-05）

为回答“teacher-fused KV 是否真的让学生当场生成更好的答案”，启动固定 500 题的配对
评测：同一学生初始 checkpoint、同一题目，分别用 plain student KV 和 official fuser
生成，均采用 greedy decode，层映射显式固定为 `last_aligned`。输出逐题记录 plain/fused
答案，统计 both-correct、plain-only-correct、fused-only-correct、both-wrong 以及净提升。
该测试不改变学生权重，也不把 teacher KV 带入最终 student-only 评测。

远端任务 `rollout_gain_20260805/base_plain_vs_fused_500.jsonl` 已在 GPU1 启动；完成后
再根据 fused-only 与 plain-only 的配对覆盖率决定是否实现“fused 有益才保留，否则回退
plain”的自适应 rollout 训练。

### 阶段二十七：论文尺度 vLLM 交付包与在线注入边界（2026-08-05）

按“只比较 EOPD 与 CacheEOPD”重新整理了
`experiments/full_scale_cacheeopd/`：

- `setup_conda_env.sh`：创建 Conda 环境并安装 vLLM `>=0.13.0`、verl/EOPD 与数学评测依赖；
- `prepare_c2c_projector_data.py`：把 C2C 使用的 `teknium/OpenHermes-2.5` 转成 projector
  训练所需的 `messages/prompt/solution` JSONL；
- `train_projector_8b_to_1p7b.sh`：冻结 Qwen3-8B/Qwen3-1.7B，按 C2C 的三层、1024 宽、
  `lr=1e-4`、`grad_accum=8`、`last_aligned` 配置预训练 projector；
- `run_eopd_cacheeopd_vllm.sh`：固定论文复现记录中的 EOPD 参数（MATH、top-k 16、
  entropy 0.8、soft-KD 1.0、batch/mini-batch 128/32、3 epoch、保存间隔 50），并把
  student/teacher/projector/data/GPU 路径集中到 `env.example`；
- README 与实验计划明确最终评测只加载 student checkpoint，使用 MATH500、AMC23、
  Minerva、OlympiadBench、AIME24、AIME25 的 `k=8`、temperature 1.0、top-p 0.8、
  max tokens 8192，报告 Avg@8 与 Pass@8。

静态验收通过：三个 shell 启动器 `bash -n`、两个修改/新增 Python 文件 AST 解析、
`git diff --check` 均成功。当前工作区是只读挂载，`py_compile` 不能写 `__pycache__`，
故未在本地生成字节码；这不代表 Python 语法失败。

必须保留的结论：vLLM V1 connector 已在真实 engine smoke 中验证
`fused packet -> paged KV -> decode`，但这只是**静态 packet**。正式训练时 student 权重会在每次
优化后变化；现有 async server 不会为每个 request 用“当前已同步的 vLLM student 权重”生成
teacher KV、student KV 和 fused packet。因此不能把静态初始 checkpoint 的 packet 重复用于多步
CacheEOPD 训练，也不能把它作为正式 EOPD 对照结果。

为防止静默退化，交付脚本目前允许 EOPD 正常启动，而 `cacheeopd` 分支会明确退出并说明该
在线 provider 缺口。下一项工程工作必须位于 vLLM model-runner/prefill 路径：以当前同步权重
取得 student prefix KV，计算同 prompt 的 teacher prefix KV，经 frozen projector 融合后在最后一
个 prompt token prefill 前回写 paged KV；完成后需做“更新一次 student 权重后，packet 与当前
模型对应”的 smoke，才可启动论文尺度 CacheEOPD。

### 阶段二十八：仓库结构导读（2026-08-05）

新增仓库根目录 `CODEBASE_GUIDE.md`，集中说明 `verl`、`rosetta`、`cache_eopd`、
`scripts/eopd` 与论文尺度交付目录的职责；文档还按“EOPD 训练主链路 → C2C fused KV →
HF research runner → vLLM packet/connector → student-only 评测”的顺序给出阅读路线，
并明确 vLLM 当前只完成静态 packet 注入、尚缺当前权重在线 packet provider 的工程边界。
