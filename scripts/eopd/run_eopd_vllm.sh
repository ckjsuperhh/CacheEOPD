#!/usr/bin/env bash
set -euo pipefail

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,4,5}

STUDENT_PATH=${STUDENT_PATH:?set STUDENT_PATH}
TEACHER_PATH=${TEACHER_PATH:?set TEACHER_PATH}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE}
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}
NGPU=${NGPU:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
MAX_PROMPT=${MAX_PROMPT:-2048}
MAX_RESPONSE=${MAX_RESPONSE:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT + MAX_RESPONSE))}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-$((MAX_MODEL_LEN * 2))}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
TOPK_LOGITS=${TOPK_LOGITS:-16}
ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:-0.8}
SOFT_KD_COEF=${SOFT_KD_COEF:-1.0}
TOTAL_STEPS=${TOTAL_STEPS:-300}
LR=${LR:-1e-6}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
MODEL_ATTN_IMPLEMENTATION=${MODEL_ATTN_IMPLEMENTATION:-sdpa}
MODEL_USE_REMOVE_PADDING=${MODEL_USE_REMOVE_PADDING:-False}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/eopd_vllm}
SAVE_FREQ=${SAVE_FREQ:-50}
C2C_ENABLE=${C2C_ENABLE:-False}
C2C_TEACHER_PATH=${C2C_TEACHER_PATH:-$TEACHER_PATH}
C2C_PROJECTOR_PATH=${C2C_PROJECTOR_PATH:-}
C2C_FUSER_DIR=${C2C_FUSER_DIR:-}

DATA_ARGS=(
  data.train_files="$TRAIN_FILE"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.max_prompt_length="$MAX_PROMPT"
  data.max_response_length="$MAX_RESPONSE"
  data.prompt_key=prompt
  data.truncation=error
  data.filter_overlong_prompts=True
)

DATA_ARGS+=(data.val_files="$VAL_FILE")

C2C_ARGS=(
  actor_rollout_ref.rollout.c2c.enable="$C2C_ENABLE"
  actor_rollout_ref.rollout.c2c.kv_connector=CacheEOPDConnector
  actor_rollout_ref.rollout.c2c.kv_connector_module_path=cache_eopd.vllm_kv_connector
)
if [[ "$C2C_ENABLE" == "True" ]]; then
  C2C_ARGS+=(++actor_rollout_ref.rollout.c2c.teacher_path="$C2C_TEACHER_PATH")
  if [[ -n "$C2C_PROJECTOR_PATH" ]]; then
    C2C_ARGS+=(++actor_rollout_ref.rollout.c2c.projector_path="$C2C_PROJECTOR_PATH")
  fi
  if [[ -n "$C2C_FUSER_DIR" ]]; then
    C2C_ARGS+=(++actor_rollout_ref.rollout.c2c.fuser_dir="$C2C_FUSER_DIR")
  fi
fi

python3 -m verl.trainer.main_ppo \
  "${DATA_ARGS[@]}" \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path="$STUDENT_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  ++actor_rollout_ref.model.override_config.attn_implementation="$MODEL_ATTN_IMPLEMENTATION" \
  ++actor_rollout_ref.model.use_remove_padding="$MODEL_USE_REMOVE_PADDING" \
  ++actor_rollout_ref.teacher_model.path="$TEACHER_PATH" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  ++actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  ++actor_rollout_ref.ref.fsdp_config.param_offload=True \
  ++actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  ++actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
  ++actor_rollout_ref.actor.use_dynamic_bsz=True \
  ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN" \
  actor_rollout_ref.actor.policy_loss.loss_mode=on_policy_distill \
  ++actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab=True \
  ++actor_rollout_ref.actor.policy_loss.soft_kd_entropy_threshold="$ENTROPY_THRESHOLD" \
  ++actor_rollout_ref.actor.policy_loss.soft_kd_loss_coef="$SOFT_KD_COEF" \
  ++actor_rollout_ref.ref.topk_logits="$TOPK_LOGITS" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT" \
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
  ++actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  "${C2C_ARGS[@]}" \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.adv_estimator=on_policy \
  ++trainer.trainer_class=OnPolicyDistillTrainer \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.total_epochs=1 \
  trainer.logger="['console']" \
  trainer.project_name=cacheeopd \
  trainer.experiment_name=eopd-vllm \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.val_before_train=False \
  trainer.test_freq=50 \
  +rollout.n_gpus_per_node="$NGPU" \
  +rollout.nnodes=1 \
  +ray_kwargs.ray_init.include_dashboard=False
