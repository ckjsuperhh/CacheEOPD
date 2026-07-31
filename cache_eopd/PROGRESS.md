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
