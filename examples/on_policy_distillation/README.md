## On-Policy Distillation (EOPD/OPD 核心目录)

<!--
本目录是 EOPD（Entropy-gated On-Policy Distillation）算法的使用入口。
核心文件（on_policy_it.sh、generate_offline_vllm.py、eval_six_benchmarks.sh、
score_avg_pass_at_k.py）在复现时创建，详见 EOPD复现.md。

EOPD 完整流程：
  1. 数据预处理：python examples/data_preprocess/math_dataset.py
  2. 训练：bash scripts/eopd/run_eopd.sh（内部调用 on_policy_it.sh）
  3. 权重合并：python -m verl.model_merger merge --backend fsdp
  4. 评测：bash eval_six_benchmarks.sh（生成 + 评分）
-->

After installing `verl`, run `examples/on_policy_distillation/on_policy_it.sh`.
Prepare the required datasets by preprocessing in `examples/data_preprocess`.

### On-Policy Distillation Settings (from `on_policy_it.sh`)

以下配置项来自 `on_policy_it.sh`，是 EOPD/OPD 训练的核心超参：

- `algorithm.adv_estimator=on_policy`: 启用 on-policy 优势估计（区别于 PPO 的 GAE）。
  <!-- enables on-policy advantage estimation. -->
- `actor_rollout_ref.teacher_model.path=Qwen/Qwen3-8B`: 教师模型路径，用于蒸馏。
  <!-- teacher model used for distillation. -->
- `actor_rollout_ref.actor.policy_loss.loss_mode=on_policy_distill`: 使用 on-policy 蒸馏损失（clipped reverse-KL + entropy-gated forward-KL）。
  <!-- use on-policy distillation loss. -->
- `actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab=True`: 对学生模型的全词表分布进行蒸馏（而非仅 top-k）。
  <!-- distill against full-vocab teacher distribution. -->
- `actor_rollout_ref.ref.topk_logits=32`: 教师分布取 top-k=32 的 logits 计算 forward-KL。
  <!-- use top-k teacher logits for FKL. -->
- `trainer.trainer_class=OnPolicyDistillTrainer`: 使用 on-policy 蒸馏训练器（将教师模型同时作为 ref 和 teacher worker）。
  <!-- uses the on-policy distillation trainer. -->

### EOPD 特有配置（在 `scripts/eopd/run_eopd.sh` 中设置）

- `soft_kd_entropy_threshold=0.8` (EOPD) / `100` (OPD): 熵阈值 τ。
  EOPD 仅对教师熵 > τ 的 token 计算 forward-KL；
  OPD 将 τ 设为极大值，等价于不使用 forward-KL。
- `soft_kd_loss_coef=1.0`: forward-KL 损失系数 α。