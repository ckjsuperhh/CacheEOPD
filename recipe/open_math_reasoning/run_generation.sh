# ============================================================
# 脚本用途：使用训练好的模型生成数学推理响应
# ============================================================
# 启动的任务：使用 vLLM 推理引擎对 AIME24/AIME25 测试题进行批量生成
# 关键参数：
#   - rollout.n=32: 每个 prompt 采样 32 个响应
#   - rollout.temperature=1.0: 采样温度
#   - rollout.response_length=20480: 最大响应长度
#   - GEN_TP: 张量并行大小
# ============================================================
#!/usr/bin/env bash

MODEL_PATH=${MODEL_PATH:-/path/to/ckpt/global_step_19751/huggingface}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
OUTPUT_PATH=${OUTPUT_PATH:-$HOME/data/gen/qwen_8b_gen_test.parquet}
GEN_TP=${GEN_TP:-1}  # Default tensor parallel size to 2

aime24_test_path=${HOME}/data/math-ai/aime24_test.parquet
aime25_test_path=${HOME}/data/math-ai/aime25_test.parquet
train_files="['$aime24_test_path', '$aime25_test_path']"

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=20480 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=32 \
    data.train_files="$train_files" \
    data.prompt_key=prompt \
    +data.output_path="${OUTPUT_PATH}" \



