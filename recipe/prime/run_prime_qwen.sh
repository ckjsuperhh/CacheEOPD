#!/usr/bin/env bash
# ============================================================================
# PRIME 训练启动脚本 - GSM8K + MATH 数学推理任务
#
# 本脚本使用 PRIME 算法在 GSM8K 和 MATH 数据集上训练 Eurus-2-7B-SFT 模型。
#
# 训练任务：PRIME（在线奖励模型 + RLOO 优势估计）
# 模型：PRIME-RL/Eurus-2-7B-SFT（7B 参数 SFT 模型）
# 数据集：GSM8K（小学数学） + MATH（高中/竞赛数学）
#
# 关键参数说明：
#   - train_batch_size=64: 训练批次大小
#   - max_prompt_length=1024: prompt 最大 token 数
#   - max_response_length=3072: 响应最大 token 数
#   - rollout.n=4: 每个 prompt 生成 4 个响应
#   - reward_model.model.update=before: 先更新 RM 再前向推理
#   - reward_model.model.beta_train=0.05: DPO 温度系数
#   - accuracy_lower_bound/upper_bound: 过滤过难/过易的样本
#   - oversample_factor=4: 过采样因子，生成 4x 数据后过滤
#
# 数据下载：https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data
# ============================================================================
set -x


gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

# download from https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

model_path=PRIME-RL/Eurus-2-7B-SFT
# model_path=Qwen/Qwen2.5-0.5B-Instruct

python3 -m recipe.prime.main_prime \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.2 \
    data.accuracy_upper_bound=0.8 \
    data.oversample_factor=4 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.model.path=$model_path \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.update=before \
    reward_model.model.beta_train=0.05 \
    reward_model.model.optim.lr=1e-6 \
    reward_model.model.optim.grad_clip=10.0 \
    reward_model.model.input_tokenizer=null \
    reward_model.mini_batch_size=64 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='prime_example' \
    trainer.experiment_name='Eurus-2-7B-SFT-gsm8k' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=64 \
    trainer.test_freq=64 \
    trainer.total_epochs=15 $@
