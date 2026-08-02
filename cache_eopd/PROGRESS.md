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
