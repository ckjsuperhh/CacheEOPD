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
    └── eval_fused_kv.py       # 量化评测：KV 偏移 / 分布 KL / 教师对齐
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

层映射（`build_layer_mapping`）按相对深度对齐，处理 teacher/student 层数不同的情况
（如 Qwen3-4B 36 层 → Qwen3-1.7B 28 层）。

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
> 注 2：`kl_to_teacher` 指标当前区分度不足（基线仅 1.4e-4）。原因是它只看 prompt
> 末位一个 token，而该位置 teacher/student 都几乎确定输出 `<think>`，分布近乎重合。
> 训练 projector 后应改为在整段 response 上按位置平均，才能作为有效验收指标。

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
```

## 待办

- [ ] 训练 projector（当前随机初始化，融合信号无意义）
- [ ] Part II：把 teacher logits（`FusedKVBuilder.build` 的 `extras`，
      需设 `return_teacher_logits=True`）接入 `core_algos.compute_policy_loss_on_policy_distill`
      的 token 重要性加权
- [ ] 端到端 EOPD 训练跑通（需完整 verl 环境 + ray）
- [ ] 路线 B：vLLM/SGLang server 内注入 fused KV（规模化）
