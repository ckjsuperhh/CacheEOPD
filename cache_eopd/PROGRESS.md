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

600 步正式训练运行中（1105 条轨迹），结果待填。

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

- [ ] v6（SFT 口径）600 步训完 → 300 题 GSM8K 重测，看能否翻过 86.0% 的 baseline
- [ ] 若 v6 仍不及 baseline：查逐条 diff，重点看融合是否让 student 复制了
      teacher 的**题面理解**还是仅仅是表层措辞
- [ ] 扩数据：GSM8K train 有 7473 题，当前只用了 1.1k
- [ ] Part II：teacher logits 接入 `core_algos.compute_policy_loss_on_policy_distill`
      的 token 重要性加权
- [ ] 端到端 EOPD 训练跑通（需完整 verl 环境 + ray）
- [ ] 路线 B：vLLM/SGLang server 内注入 fused KV（规模化）

---

## 复现命令

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
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. $PY \
  -m cache_eopd.train_projector \
    --teacher $M/Qwen3-4B --student $M/Qwen3-1.7B \
    --data-path ./data/train_v6.jsonl \
    --device cuda:3 --teacher-device auto --teacher-gpus 4,5 --attn-impl sdpa \
    --steps 600 --grad-accum 8 --lr 1e-4 --anneal-steps 600 --gate-init 1.0 \
    --eval-every 100 --holdout 64 --out-dir ./ckpt_projector_v6

# 3) 数学正确率验收（含 harness 三路自检；这才是唯一可信的判据）
PYTHONPATH=. $PY -m cache_eopd.eval_math_acc \
    --teacher $M/Qwen3-4B --student $M/Qwen3-1.7B \
    --data-path /home/kejiechen/taopd-baseline/data/GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl \
    --device cuda:3 --teacher-device auto \
    --num-samples 300 --max-new-tokens 512 --sanity 5 \
    --projector-path ./ckpt_projector_v6/projector_final.pt --out ./gsm8k_math_v6.jsonl
```

checkpoint：apex-llm `~/CacheEOPD/ckpt_projector_v6/projector_final.pt`（+ `.weights`）
v5 的 ckpt 已作废（训练口径错误，见阶段四）。
