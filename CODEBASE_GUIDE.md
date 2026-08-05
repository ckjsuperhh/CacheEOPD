# CacheEOPD 代码导读

本仓库是在 `verl` 训练框架上扩展 EOPD（Entropy-gated On-Policy
Distillation）和 CacheEOPD 的实验代码。它同时保留了通用 RL 训练框架、C2C
projector 实现、HF 研究原型和 vLLM 工程接入；阅读时应先区分这些层次，避免把
小规模 HF 原型和论文尺度 vLLM 实验混为一谈。

## 目录地图

| 目录 | 职责 | 是否为本项目核心 |
| --- | --- | --- |
| `verl/` | 通用训练框架：Ray 调度、FSDP actor、rollout 后端、PPO/EOPD trainer | 是，EOPD 主干在这里 |
| `rosetta/` | C2C 的 projector、fuser 和相关工具 | 是，KV 融合的模型部件在这里 |
| `cache_eopd/` | 本项目新增的融合、projector 训练、HF 实验和评测工具 | 是，最应优先阅读 |
| `scripts/eopd/` | 原始 EOPD 的 vLLM 启动脚本 | 是 |
| `experiments/full_scale_cacheeopd/` | 可移植的 Conda、projector、EOPD/CacheEOPD 论文尺度交付包 | 是 |
| `examples/`、`recipe/`、`tests/` | verl 的其他算法示例、配方和测试 | 通常不需要为 CacheEOPD 修改 |

## EOPD：不使用 KV 融合的基线

EOPD 的完整训练流程是：

```text
MATH prompt
  -> 当前 student 用 vLLM rollout
  -> teacher 评估 student 生成的完整轨迹
  -> EOPD policy-gradient + entropy-gated top-k soft KD loss
  -> FSDP 更新 student 权重
  -> 下一轮把新权重同步到 vLLM
```

主要入口如下：

- `scripts/eopd/run_eopd_vllm.sh`：传入模型、数据、batch、EOPD 超参数和 vLLM 参数。
- `verl/trainer/main_ppo.py`：Hydra 命令行入口，构造训练器。
- `verl/trainer/ppo/on_policy_distill_trainer.py`：取 batch、请求 rollout、组织 teacher
  信号并调用 actor 更新。
- `verl/trainer/ppo/core_algos.py`：EOPD 的损失函数实现。这里包含 clipped
  on-policy 项和基于 teacher 熵筛选的 top-k soft-KD 项。
- `verl/workers/fsdp_workers.py`：FSDP student/teacher actor，以及训练和 rollout
  模式之间的权重同步。
- `verl/workers/rollout/vllm_rollout/`：vLLM rollout worker 和 HTTP/async server。

最终评测必须只加载保存下来的 student checkpoint；不能加载 teacher、projector，也不能
注入 teacher KV。否则测到的是“teacher 辅助生成”，不是学生学会的能力。

## CacheEOPD：在 rollout 前融合 KV

CacheEOPD 只应改变 student rollout 的初始状态，不改变后续 EOPD 的 teacher 检测和
loss。正确的概念流程是：

```text
同一个 prompt
  -> teacher prefix forward，得到 teacher KV
  -> 当前 student prefix forward，得到 student KV
  -> frozen C2C projector 按层映射投影并融合
  -> fused student-shaped KV
  -> student 从 fused KV 继续 rollout
  -> teacher 按普通 EOPD 方式评估 student response
  -> 更新 student
```

核心文件：

- `cache_eopd/fused_kv.py`：`FusedKVBuilder`。它读取 teacher/student 的 KV head 数、
  head dimension 和层数，构造 teacher layer 到 student layer 的映射，并逐 student layer
  使用 projector 得到 fused KV。
- `rosetta/model/projector.py`：C2C projector/fuser 的结构和门控。
- `cache_eopd/train_projector.py`：冻结 teacher/student，训练 projector 让 fused prefix
  对 response token 的 teacher-forcing CE 更低。
- `cache_eopd/prepare_c2c_projector_data.py`：把 OpenHermes 对话转为
  `messages + prompt + solution` 数据格式。

目前使用 `last_aligned` 映射：若 teacher 层数比 student 多，则 student 第 `i` 层匹配
teacher 的末尾对齐层。融合只覆盖 prompt 的前 `L-1` 个 token；最后一个 prompt token
保留 student 原生 KV，并由解码首步重新计算。这避免 KV 与首个 response token logits
错位。

projector 内还有两种不同层级的“门”：

1. `mixed`/`anneal` 的概率门，决定某次 rollout 是否使用 fused KV；
2. projector 的 key/value gate，决定每一层投影增量是否生效。

二者不要混淆。前者是训练策略，后者是 frozen projector 的内部参数。

## 已跑通的 HF 研究路径

`cache_eopd/c2c_hf_rollout.py` 将 `HFRollout` 改为直接把 `DynamicCache` 传给
Transformers 模型；这是最直接、最容易验证数学正确性的路径。

`cache_eopd/train_eopd_cacheeopd_hf.py` 是小规模独立实验脚本：

- `fused`：每次 rollout 都融合；
- `mixed`：以固定概率融合；
- `anneal`：融合概率随训练步数退火，可选 linear、quadratic、sqrt。

它使用 fused 行为策略下的 old log-prob，并在 fused 分支用相同 fused prefix 计算
current log-prob，避免 PPO/EOPD 的行为策略和训练策略不一致。该脚本用于现有 GSM8K
pilot，不是论文尺度 vLLM 交付入口。

## vLLM KV 注入路径

vLLM 不接收 Transformers 的 `DynamicCache`，并使用 paged KV cache。因此需要转换层：

```text
HF teacher + HF student + projector
  -> fused DynamicCache
  -> request-level .pt KV packet
  -> vLLM V1 KV connector
  -> vLLM paged KV blocks
  -> 计算最后一个 prompt token
  -> student decode
```

- `cache_eopd/vllm_kv_packet.py`：将 fused cache 保存成单请求 packet，并附带 prompt
  token hash 和长度。
- `cache_eopd/prepare_vllm_kv_packet.py`：从模型路径和 prompt token 离线生成 packet。
- `cache_eopd/vllm_kv_connector.py`：vLLM V1 connector。它让 scheduler 将对齐后的前
  `L-1` token 视为已计算，并在 worker 中把 packet 写入 paged KV block。
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`：启用 C2C 时注册 connector，
  并在缺少 packet 时硬失败，避免静默退化为普通 EOPD。

静态 packet 到 vLLM decode 的 engine smoke 已验证。但这不是完整训练：student 每次
optimizer update 后权重都会变，旧权重计算出的 student KV 不能用于新权重的 rollout。

因此正式在线 CacheEOPD 尚缺少 model-runner/prefill 级的实现：每个 rollout 都必须以
**当前已同步的 student 权重**生成 student prefix KV，再与同 prompt 的 teacher KV 和
frozen projector 融合，最后在 vLLM 计算最后一个 prompt token 前写回 paged KV。不能将
训练开始时生成的 packet 重复用于多步训练。

## 评测和实验记录

- `cache_eopd/eval_student_batch.py`：批量 student-only 评测。
- `cache_eopd/eval_math_acc.py`：数学正确率及 plain/fused 生成辅助评测。
- `cache_eopd/eval_fused_kv.py`：检查 zero-init、层映射和融合本身的正确性。
- `cache_eopd/eval_projector_kl.py`：projector 的 token-level 诊断。
- `cache_eopd/PROGRESS.md`：实验命令、结果、失败原因和结论的唯一连续记录。

论文尺度交付在 `experiments/full_scale_cacheeopd/`：

1. `setup_conda_env.sh` 创建含 vLLM 的 Conda 环境；
2. `env.example` 集中填写模型、数据、GPU 与输出路径；
3. `train_projector_8b_to_1p7b.sh` 训练 Qwen3-8B 到 Qwen3-1.7B projector；
4. `run_eopd_cacheeopd_vllm.sh eopd <seed>` 运行 EOPD。

该目录中的 `cacheeopd` 分支目前会主动停止，正是为了阻止使用陈旧 packet 产生无效的
“CacheEOPD”对照。

## 推荐阅读顺序

```text
cache_eopd/PROGRESS.md
  -> cache_eopd/fused_kv.py
  -> cache_eopd/train_eopd_cacheeopd_hf.py
  -> verl/trainer/ppo/on_policy_distill_trainer.py
  -> verl/trainer/ppo/core_algos.py
  -> cache_eopd/vllm_kv_packet.py
  -> cache_eopd/vllm_kv_connector.py
  -> verl/workers/rollout/vllm_rollout/vllm_async_server.py
```

按这个顺序可以先理解实验目标和 KV 融合语义，再理解 EOPD 如何训练，最后处理 vLLM
工程问题。
