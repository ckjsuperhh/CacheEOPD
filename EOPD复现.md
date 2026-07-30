<!--
  EOPD复现.md - EOPD 论文复现实验报告
  中文摘要：本文件记录了 EOPD（Entropy-gated On-Policy Distillation）论文的复现实验。
  使用 Qwen3-1.7B-Base 作为学生模型、Qwen3-8B 作为教师模型，
  在 MATH 数据集上复现论文 Table 2 的 OPD baseline 和 EOPD 两个变体的对照实验。
  包含方法设定、环境配置、训练流程和结果分析。
-->
# EOPD 复现报告（Qwen3-1.7B-Base + Qwen3-8B）

> 目标：复现论文 EOPD 的 Table 2（Qwen3-1.7B-Base 学生 / Qwen3-8B 教师 / MATH 训练数据这一行），
> 含 **OPD baseline** 与 **EOPD** 两个变体对照。
> 仓库本体是 verl 的一个 fork，EOPD/OPD 的实现已包含在内（见 §3）。

---

## 0. 方法设定（来自论文 §5.1 / Appendix F）

| 项 | 取值 |
|---|---|
| 学生模型 | **Qwen3-1.7B-Base** |
| 教师模型 | **Qwen3-8B**，**关闭 thinking mode**（论文 §5.1） |
| 训练数据 | MATH（`dapo_math17k` 等） |
| teacher top-k | **k = 16** |
| τ（熵阈值） | **0.8**（仅在高熵 token 上加 forward-KL） |
| α（soft-KD 系数） | **1.0** |
| batch / mini-batch | 128 / 32，每迭代 4 步梯度更新 |
| 评测采样 | temperature=1.0, top_p=0.8, max_tokens=8192, 每题 **k=8** |
| 指标 | **Avg@8**（8 条中正确比例均值）、**Pass@8**（至少 1 条正确的题占比） |

**EOPD = clipped reverse-KL (OPD) + entropy-gated forward-KL on high-entropy teacher tokens。**
**OPD = 仅 clipped reverse-KL**（把 `soft_kd_entropy_threshold` 设为很大使 forward-KL mask 恒空）。

---

## 1. 环境 (Environment)

- **conda 环境：`eopd`**（`/opt/conda/envs/eopd/bin/python3`，VM；apex 上为同名环境）。
  - verl `0.7.0.dev0`（本 fork）；评测**必须**用 eopd 环境，否则缺 DTensor/math_verify。
  - 评测依赖 **vLLM**（离线生成）+ `math_verify` + `transformers` + `pandas` + `ray` + `fsdp`。
- **flash_attn**：若机器 glibc < 2.32 无法装真 flash_attn，`scripts/eopd/flash_attn_shim` 会被自动注入 `PYTHONPATH`，让 `use_remove_padding=True` 走 sdpa 后端生效。
- **VM 特有问题**：多卡容器内设 `NCCL_P2P_DISABLE=1`；**切勿**设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`（与 vLLM memory pool 冲突报错）。
- **多卡提速**：1.7B 小模型用**数据并行**（`NUM_SHARDS=N` 个 TP=1 的 vLLM 实例并发），不要用 `tensor_parallel_size>1`（通信开销盖过计算）。
- 安装：`pip install -e .` + `pip install -r requirements-cuda.txt`（详见 `setup_env.sh`）。

---

## 2. 数据处理 (Data Process)

- **训练数据（MATH）**：`examples/data_preprocess/math_dataset.py` → 输出到 `DATA_DIR`（默认 `$HOME/data`，与评测共用）。
- **评测 6 基准预处理**：`scripts/eopd/data_preprocess/` 下
  `math500_test.py` / `amc23_test.py` / `minerva_test.py` / `olympiadbench_test.py` / `aime24_test.py` / `aime25_test.py`，
  统一由 `run_all_preprocess.sh` 一键生成。
  - 输出 schema：`{"prompt", "reward_model": {"ground_truth"}}`。
  - **OlympiadBench 子集名称**：`OE_TO_maths_en_COMP`（论文所用子集；不同机器取到的题目数不同，见 §5 注意）。
- **评测 prompt**：zero-shot，含 `Please reason step by step, and put your final answer within \boxed{}.`

---

## 3. 训练脚本 (Train Script)

- **EOPD 启动**：`bash scripts/eopd/run_eopd.sh`
  - 必需：`STUDENT_MODEL_PATH`（Qwen3-1.7B-Base）、`TEACHER_MODEL_PATH`（Qwen3-8B）
  - 可选：`DATA_DIR` / `CKPT_DIR` / `CUDA_VISIBLE_DEVICES` / `N_GPUS_PER_NODE` / `OFFLINE`
- **OPD baseline 启动**：`bash scripts/eopd/run_opd.sh`（= `run_eopd.sh` 但 `METHOD=opd`，soft-KD 项不激活）
- 脚本最终调用 `examples/on_policy_distillation/on_policy_it.sh`。
- **关键超参**（对应 `examples/on_policy_distillation/README.md` + 论文）：

  | 配置键 | 取值 | 含义 |
  |---|---|---|
  | `trainer.trainer_class` | `OnPolicyDistillTrainer` | 训练器（把教师当 ref/teacher worker） |
  | `algorithm.adv_estimator` | `on_policy` | on-policy 优势估计 |
  | `actor_rollout_ref.actor.policy_loss.loss_mode` | `on_policy_distill` | EOPD/OPD 损失 |
  | `actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab` | `True` | 全词表蒸馏 |
  | `actor_rollout_ref.ref.topk_logits` | `16` | k=16（论文） |
  | `actor_rollout_ref.actor.policy_loss.soft_kd_entropy_threshold` | `0.8` (EOPD) / `100` (OPD) | τ |
  | `actor_rollout_ref.actor.policy_loss.soft_kd_loss_coef` | `1.0` | α |
  | batch / mini-batch | `128` / `32` | 每迭代 4 步梯度 |

- **损失实现**：`verl/trainer/ppo/core_algos.py` 的 `compute_policy_loss_on_policy_distill`
  （clipped reverse-KL + entropy-gated forward-KL；`soft_kd_entropy_threshold` 对高熵 token 加 mask）。
- **权重导出**：训练自动导出 HF 权重常失败，需手动
  `python -m verl.model_merger merge --backend fsdp` 合并 FSDP 分片 → `actor/huggingface/model.safetensors`。
- **监控**：关注 `actor/soft_kd_*` 指标——EOPD 应出现非零 `soft_kd_token_ratio`（高熵 token 被蒸馏），OPD 恒为 0。

---

## 4. 评测流程 (Eval Pipeline)

- **离线生成**：`examples/on_policy_distillation/generate_offline_vllm.py`
  （`--num_shards` / `--shard_id` / `--merge` / `--input_glob`；分片写 `__idx__` 保证合并后顺序一致）。
- **6 基准主流程**：`examples/on_policy_distillation/eval_six_benchmarks.sh`
  （`NUM_SHARDS` 自动探测 nvidia-smi；>1 时每 benchmark 启 N 个 vLLM 进程并合并）。
- **评分**：`examples/on_policy_distillation/score_avg_pass_at_k.py`
  （Avg@8 / Pass@8，双层 tqdm，math_verify 静默 WARNING；逐条 try/except 防坏数据拖垮）。
- **机器配置**：VM 8 卡 → `NUM_SHARDS=8`；apex 1 卡 → `NUM_SHARDS=1`。
- **产物**：`eval_results/<run>/actor_huggingface_scores.{json,txt}`。

---

## 5. 结果 (Results)

### 5.1 论文 Table 2 目标（EOPD，Qwen3-1.7B-Base / MATH）vs 本次复现

| Benchmark | N | 论文 EOPD (目标) Avg@8 / Pass@8 | apex EOPD (s174) Avg@8 / Pass@8 | VM EOPD (s330) Avg@8 / Pass@8 |
|---|---|---|---|---|
| MATH500 | 500 | 68.73 / 87.60 | 43.85 / 82.60 | 33.75 / 75.60 |
| AMC23 | 40 | 41.88 / 75.00 | 23.13 / 67.50 | 17.50 / 52.50 |
| Minerva | 272 | 30.15 / 50.74 | 14.20 / 38.97 | 8.78 / 28.68 |
| OlympiadBench | 见注 | 30.28 / 51.11 | 19.75 / 47.77 (n=674) | 7.19 / 22.81 (n=1517) |
| AIME24 | 30 | 10.42 / 23.33 | 5.42 / 23.33 | 2.92 / 13.33 |
| AIME25 | 30 | 5.83 / 16.67 | 3.75 / 16.67 | 0.83 / 6.67 |
| **MEAN** | — | **31.22 / 50.74** | **18.35 / 46.14** | **11.83 / 33.26** |

### 5.2 解读与注意

- **分数偏低属正常**：两机都只训到数百 step（apex s174 / VM s330），远少于论文 3 epoch。评测协议（temp=1.0, top_p=0.8, max 8192, k=8, zero-shot, `\boxed{}`）与论文一致，**不是评测 bug**。
- **⚠️ OlympiadBench 子集规模不一致**：apex 取 n=674、VM 取 n=1517、论文子集约 2126。跨机平均仅作参考；**单 benchmark 的 Pass@8 仍可比**，但 MEAN 受该 bench 权重影响。
- **VM s330 反而低于 apex s174**：VM 训练步数更多却分数更低，初步怀疑① VM 的 OlympiadBench 评测不完整（仅 1517/2126 行，见下方待办）② 两机训练数据/seed/超参细节可能有差异。需进一步核对 VM 训练日志与 Oly 评测完整性。
- **对照结论**：论文核心结论是 EOPD 相对 OPD 的 Pass@8 差值（~+2.4 @1.7B）。当前两机均只跑了 EOPD；**OPD baseline 评测尚未完成**，待补齐后才能在 Table 2 中给出 EOPD−OPD 差值。

### 5.3 结果存档

- `eval_results/apex_eopd_step174/actor_huggingface_scores.{json,txt}`
- `eval_results/vm_eopd_step330/actor_huggingface_scores.{json,txt}`

---

## 6. 日志与路径 (Log Paths)

| 内容 | VM（8 卡） | apex（1 卡） |
|---|---|---|
| 仓库 | `/inspire/hdd/project/multi-agent/zhangweinan-24046/dk/CacheOPD/eopd-baseline` | `/home/kejiechen/eopd-baseline` |
| 数据 `DATA_DIR` | `/inspire/hdd/project/multi-agent/zhangweinan-24046/dk/data` | `/home/kejiechen/data` |
| 训练 ckpt（FSDP 分片） | `/root/ckpts/EOPD` | `/home/kejiechen/ckpts` |
| 评测产物 | `/inspire/.../dk/data/eval_results/` | `/home/kejiechen/data/eval_results/` |
| 训练日志 | 见 `指令_vm_全流程.txt` 指定 | `eopd_train.log` / `eopd_train_eopd.log` / `eopd_paper_repro_*.log` / `eopd_monitor.log` |

- 训练后 HF 权重需手动合并到各 ckpt 下的 `actor/huggingface/`（见 §3）。
- wandb：各机器默认上报，按需查看。

---

## 7. 复现指令索引（仓库内 `指令_*.txt`）

| 文件 | 用途 |
|---|---|
| `指令_vm_全流程.txt` | VM 完整流程：EOPD 训练→转换→评测 + OPD 训练→转换→评测 → Table 2 |
| `指令_vm_重跑评测.txt` | 停单卡→拉新→8 卡重评测 |
| `指令_vm_修评分.txt` | 只重评分（不重生成） |
| `指令_vm_诊断合并.txt` / `指令_vm_重合并并评分.txt` | 合并失败诊断 / 重合并+评分 |
| `指令_vm_导出结果.txt` / `指令_apex_导出结果.txt` | 把评测分数导出进仓库并 push |
| `指令_vm_git_ssh.txt`（待建） | VM 配 SSH 推 GitHub，避免 token 反复输入 |

> 规范：所有远程执行（训练/评测/同步/诊断/修复）都先写成仓库内 `指令_*.txt`，远程 `git pull` 后 `bash` 执行；详见 `经验总结_EOPD评测部署.md`。

---

## 8. 已知问题 / 待办

1. **OPD baseline 评测未完成** → 按 `指令_vm_全流程.txt` 跑 OPD 训练+评测，补齐 Table 2 的 EOPD−OPD 差值。
2. **VM OlympiadBench 评测不完整**（1517/2126 行）→ 核对该 bench 分片生成是否漏跑，补齐后重评。
3. **跨机 OlympiadBench 子集规模不一致** → 统一为论文子集 `OE_TO_maths_en_COMP` 全量后再横向比较。
4. 评测脚本不依赖已 broken 的 `verl.trainer.main_generation`，改用离线 vLLM（`generate_offline_vllm.py`）。
5. 详见 `经验总结_EOPD评测部署.md`（协作规范、提速、已知 bug 与修复、诊断姿势）。
