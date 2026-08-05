# CacheEOPD · C2C 融入学生 Rollout（路线 A）

把 C2C（Cache-to-Cache）的「教师 KV-Cache 投影融合进学生生成」嵌入 EOPD 的**学生 rollout 阶段**。
对应方案文档 `C2C融入学生Rollout_路线A.md`（在上层目录）。

## 仓库结构

```
CacheEOPD/
├── verl/               # EOPD 全量代码（fork 自 verl），改动见下
├── rosetta/            # 从 C2C 仓库 vendor 的投影器实现（C2CProjector 等）
└── cache_eopd/         # 本项目新增代码
    ├── fused_kv.py            # 核心：teacher KV → 投影融合 → fused KV
    ├── c2c_hf_rollout.py      # C2CHFRollout：HFRollout + fused KV 注入
    ├── prototype_generate.py  # 最小原型：三路生成对比（不依赖 verl）
    ├── smoke_test_rollout.py  # C2CHFRollout 冒烟测试（不依赖 ray/FSDP）
    ├── eval_fused_kv.py       # 量化评测：KV 偏移 / 分布 KL / 教师对齐
    ├── train_projector.py     # projector 预训练（冻结 teacher/student）
    ├── eval_projector_kl.py   # 验收：response 全位置 per-token KL
    ├── eval_math_acc.py       # 下游验收：GSM8K 数学正确率
    └── PROGRESS.md            # 过程记录：改动 / 实验 / 结果 / 踩坑
```

对 EOPD 的改动只有 `verl/workers/fsdp_workers.py` 两处（新增 rollout 后端 `c2c_hf`），
原 `hf_rollout.py` 未修改，`enable=False` 时行为与原版完全一致。

## 核心思路

```
prompt ──┬──▶ teacher forward ──▶ teacher KV (sharer)
         └──▶ student forward ──▶ student KV (base)
                     │
                     ▼
        C2CProjector: fused = student + gate · w · proj(teacher)
                     │
                     ▼
        student 带 fused KV 自回归续写 ──▶ rollout response
```

层映射（`build_layer_mapping`）支持三种策略：`relative_depth`、`last_aligned` 和
`k_nearest`。官方 C2C fuser 的 recipe 使用 `last_aligned`；例如 Qwen3-4B 的 36 层
到 Qwen3-0.6B 的 28 层对应 teacher 第 8 到 35 层。旧 v6 projector 使用
`relative_depth`，评测旧 checkpoint 时必须显式保留该策略。

## 已验证结论（apex-llm, 6×A6000）

| 检验 | 结果 |
|---|---|
| self-KV 注入 vs 原生 `generate` 逐 token | **完全一致**（注入路径无 bug） |
| zero_init projector 下 fused vs self-KV | **完全一致**（融合公式恒等性正确） |
| 异构 teacher（4B 36层 → 1.7B 28层）跨层映射 | 形状对齐，生成不报错 |
| `C2CHFRollout.generate_sequences` 输出字段 | 形状符合 EOPD 训练循环约定 |

量化指标（Qwen3-1.7B ← Qwen3-4B，4 条 prompt，`eval_fused_kv.py`）：

| 指标 | zero_init | random_init |
|---|---|---|
| KV 最大绝对差 vs student | 0.0 | 22.25 |
| KV 相对 L2 偏移 (key/value) | 0.0 / 0.0 | 0.105 / 0.127 |
| 末位 argmax 一致率 vs student | 1.000 | 1.000 |

zero_init 三项全为恒等 → 融合公式与层映射实现正确；random_init 下 KV 明显偏移
→ 融合通路确实生效。

> 注 1：projector 随机初始化时生成质量会退化，这是预期的——投影器需要训练。
> `zero_init=True` 提供了「初始等价于纯 EOPD、训练中逐步引入融合」的安全起点。
>
> 注 2：`eval_fused_kv.py` 的 `kl_to_teacher` 区分度不足（基线仅 1.4e-4），因为它只看
> prompt 末位一个 token，而该位置 teacher/student 都几乎确定输出 `<think>`，分布近乎重合。
> 已由 `eval_projector_kl.py` 取代——在整段 response 上按 token 平均。

## projector 预训练结果

冻结 teacher/student 只训 `C2CProjector`，400 步后在 50 条 held-out 上：

| 设置 | per-token KL(student ‖ teacher) |
|---|---|
| baseline（student 自身） | 0.4030 |
| zero（融合恒等） | 0.4070 |
| random（未训练） | 0.6888 |
| **pretrained** | **0.1812（-55.0%）** |

训练脚本 `train_projector.py`，验收 `eval_projector_kl.py`，下游正确率 `eval_math_acc.py`。
阶段性简报见 **[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)**；完整实验过程与踩坑记录见
**[PROGRESS.md](PROGRESS.md)**。

## 复现

```bash
# 1. 最小原型（三路对比）
PYTHONPATH=. python -m cache_eopd.prototype_generate \
    --student <student_path> --teacher <teacher_path> \
    --device cuda:3 --teacher-device auto --zero-init

# 2. rollout 冒烟测试
PYTHONPATH=. python -m cache_eopd.smoke_test_rollout \
    --student <student_path> --teacher <teacher_path> --device cuda:3

# 3. 量化评测
PYTHONPATH=. python -m cache_eopd.eval_fused_kv \
    --student <student_path> --teacher <teacher_path> \
    --device cuda:3 --teacher-device auto --out metrics.json

# 4. 评估官方 C2C fuser；官方 recipe 的层映射必须使用 last_aligned
PYTHONPATH=. python -m cache_eopd.eval_math_acc \
    --student <student_path> --teacher <teacher_path> \
    --fuser-dir <official_fuser>/final --layer-mapping last_aligned \
    --device cuda:3 --teacher-device auto
```

## 在 EOPD 训练中启用

```yaml
actor_rollout_ref:
  rollout:
    name: c2c_hf
    c2c:
      enable: true
      teacher_path: /path/to/teacher       # EOPD 中即 teacher_model.path
      teacher_device: auto                 # 或 null（同卡）/ "cuda:N"
      projector_path: null                 # C2C 预训练 projector 权重；null 则随机初始化
      zero_init: false                     # true = 初始等价于纯 EOPD
      layer_mapping: last_aligned          # 官方 C2C；旧 v6 checkpoint 改为 relative_depth
```

## vLLM V1 packet injection

HF 的 `past_key_values` 不能直接传给 vLLM。vLLM 路径使用 V1 KV connector：先用
`prepare_vllm_kv_packet.py` 生成 student-shaped fused KV，再通过
`SamplingParams.extra_args.kv_transfer_params` 传入 packet 路径。connector 会把前
`L-1` 个 prompt token 的 KV 写入 vLLM 的 paged cache，再让 vLLM 正常计算最后一个
prompt token；这样首个 response token 的 logits 仍然正确，也不会重复覆盖 fused KV。

```bash
PYTHONPATH=. python -m cache_eopd.prepare_vllm_kv_packet \
  --student <student_path> --teacher <teacher_path> \
  --input-ids prompt.pt --output /dev/shm/cacheeopd/request-001.pt \
  --fuser-dir <official_fuser>/final
```

先验证 paged-cache 映射时可运行：

```bash
PYTHONPATH=. python -m cache_eopd.smoke_test_vllm_connector
```

在远端 vLLM 环境中运行完整 engine smoke：

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHONPATH=. python -m cache_eopd.smoke_test_vllm_engine \
  --model /path/to/Qwen3-0.6B
```

请求参数需要包含：

```python
{"kv_transfer_params": {
    "packet_path": "/dev/shm/cacheeopd/request-001.pt",
    "prompt_len": 128,
}}
```

配置 `actor_rollout_ref.rollout.c2c.enable=True` 后，vLLM 会强制启用
`CacheEOPDConnector`、关闭 chunked prefill/prefix caching，并在 packet 缺失时直接报错，
避免把未注入的普通 vLLM 结果误记为 CacheEOPD。当前 packet 生成器是 correctness-first 的
独立入口；下一步再把它绑定到 EOPD 当前 student 权重同步与每个 rollout request。

## 待办

- [x] 训练 projector —— held-out per-token KL 降 55%，见 [PROGRESS.md](PROGRESS.md)
- [ ] 下游验收：GSM8K 数学正确率（`eval_math_acc.py`，进行中）
- [ ] Part II：把 teacher logits（`FusedKVBuilder.build` 的 `extras`，
      需设 `return_teacher_logits=True`）接入 `core_algos.compute_policy_loss_on_policy_distill`
      的 token 重要性加权
- [ ] 端到端 EOPD 训练跑通（需完整 verl 环境 + ray）
- [x] 路线 B 第一阶段：vLLM V1 connector、paged KV 写入和 packet 生成入口
- [ ] 路线 B 第二阶段：把 packet 生成绑定到 EOPD 在线 student 权重与 rollout request
