#!/usr/bin/env bash
set -euo pipefail

method=${1:?usage: $0 {eopd|cacheeopd} [seed]}
seed=${2:-${SEED:-42}}
if [[ "$method" != "eopd" && "$method" != "cacheeopd" ]]; then
    echo "method must be eopd or cacheeopd" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
if [[ -f "$script_dir/.env" ]]; then
    set -a
    source "$script_dir/.env"
    set +a
fi

: "${PYTHON_BIN:=python}"
: "${STUDENT_PATH:?set STUDENT_PATH in .env}"
: "${TEACHER_PATH:?set TEACHER_PATH in .env}"
: "${TRAIN_FILE:?set TRAIN_FILE in .env}"
: "${VAL_FILE:?set VAL_FILE in .env}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT in .env}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NGPU:=8}"
: "${ROLLOUT_TP:=1}"
: "${ROLLOUT_DP:=$NGPU}"
: "${MAX_PROMPT:=2048}"
: "${MAX_RESPONSE:=8192}"
: "${MAX_MODEL_LEN:=$((MAX_PROMPT + MAX_RESPONSE))}"
: "${MAX_BATCHED_TOKENS:=32768}"
: "${TRAIN_BATCH_SIZE:=128}"
: "${MINI_BATCH_SIZE:=32}"
: "${MICRO_BATCH_SIZE:=1}"
: "${TOPK_LOGITS:=16}"
: "${ENTROPY_THRESHOLD:=0.8}"
: "${SOFT_KD_COEF:=1.0}"
: "${TOTAL_EPOCHS:=3}"
: "${TOTAL_STEPS:=}"
: "${LR:=1e-6}"
: "${TRAIN_TEMPERATURE:=1.0}"
: "${TRAIN_TOP_P:=0.8}"
: "${GPU_MEMORY_UTILIZATION:=0.70}"
: "${MODEL_ATTN_IMPLEMENTATION:=sdpa}"
: "${MODEL_USE_REMOVE_PADDING:=False}"
: "${SAVE_FREQ:=50}"
: "${TEST_FREQ:=50}"
: "${PROJECT_NAME:=eopd_cacheeopd_full}"
: "${OUTPUT_DIR:=$OUTPUT_ROOT/${method}_seed${seed}}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

data_args=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT"
    data.max_response_length="$MAX_RESPONSE"
    data.prompt_key=prompt
    data.truncation=error
    data.filter_overlong_prompts=True
    data.seed="$seed"
)

c2c_args=(
    actor_rollout_ref.rollout.c2c.enable=False
    actor_rollout_ref.rollout.c2c.kv_connector=CacheEOPDConnector
    actor_rollout_ref.rollout.c2c.kv_connector_module_path=cache_eopd.vllm_kv_connector
)
if [[ "$method" == "cacheeopd" ]]; then
    : "${PROJECTOR_PATH:?set PROJECTOR_PATH in .env for cacheeopd}"
    cat >&2 <<'EOF'
CacheEOPD vLLM is blocked intentionally: the connector can inject a fused KV packet,
but this launcher has no online provider that rebuilds that packet from the current
student weights for every rollout request. Refusing to run a stale-packet experiment.
EOF
    exit 3
    c2c_args+=(
        actor_rollout_ref.rollout.c2c.enable=True
        ++actor_rollout_ref.rollout.c2c.teacher_path="$TEACHER_PATH"
        ++actor_rollout_ref.rollout.c2c.projector_path="$PROJECTOR_PATH"
        ++actor_rollout_ref.rollout.c2c.layer_mapping=last_aligned
    )
fi

step_args=(trainer.total_training_steps=null)
if [[ -n "$TOTAL_STEPS" ]]; then
    step_args=(trainer.total_training_steps="$TOTAL_STEPS")
fi

exec "$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${data_args[@]}" \
    "${c2c_args[@]}" \
    "${step_args[@]}" \
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
    actor_rollout_ref.rollout.data_parallel_size="$ROLLOUT_DP" \
    actor_rollout_ref.rollout.temperature="$TRAIN_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$TRAIN_TOP_P" \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE" \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
    ++actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.adv_estimator=on_policy \
    ++trainer.trainer_class=OnPolicyDistillTrainer \
    trainer.seed="$seed" \
    trainer.n_gpus_per_node="$NGPU" \
    trainer.nnodes=1 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.logger="['console']" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="${method}_seed${seed}" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.val_before_train=False \
    trainer.test_freq="$TEST_FREQ" \
    +rollout.n_gpus_per_node="$NGPU" \
    +rollout.nnodes=1 \
    +ray_kwargs.ray_init.include_dashboard=False
