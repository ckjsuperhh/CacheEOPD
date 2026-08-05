#!/usr/bin/env bash
set -euo pipefail

method=${1:?usage: $0 {fused|mixed|anneal} SEED}
seed=${2:?usage: $0 {fused|mixed|anneal} SEED}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PROJECT_ROOT:-$(cd "$script_dir/../.." && pwd)}
if [[ -f "$script_dir/.env" ]]; then
    set -a
    source "$script_dir/.env"
    set +a
fi

: "${PYTHON_BIN:?set PYTHON_BIN in .env}"
: "${STUDENT:?set STUDENT in .env}"
: "${TEACHER:?set TEACHER in .env}"
: "${FUSER_DIR:?set FUSER_DIR in .env}"
: "${TRAJECTORY:?set TRAJECTORY in .env}"
: "${DEVICE:?set DEVICE in .env}"
: "${TEACHER_GPUS:?set TEACHER_GPUS in .env}"
: "${PROJ_DEVICE:?set PROJ_DEVICE in .env}"
: "${OUT_ROOT:?set OUT_ROOT in .env}"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
out_dir="$OUT_ROOT/cacheeopd_${method}_seed${seed}"
mkdir -p "$out_dir"

strategy_args=()
case "$method" in
    fused) strategy_args+=(--mode fused) ;;
    mixed) strategy_args+=(--mode mixed --fused-prob 0.5) ;;
    anneal) strategy_args+=(--mode anneal --anneal-start-prob 1.0 \
        --anneal-end-prob 0.0 --anneal-steps "$STEPS" --anneal-schedule linear) ;;
    *) echo "unknown method: $method" >&2; exit 2 ;;
esac

exec "$PYTHON_BIN" -m cache_eopd.train_eopd_cacheeopd_hf \
    --teacher "$TEACHER" --student "$STUDENT" --data-path "$TRAJECTORY" \
    --fuser-dir "$FUSER_DIR" --device "$DEVICE" --teacher-gpus "$TEACHER_GPUS" \
    --proj-device "$PROJ_DEVICE" "${strategy_args[@]}" \
    --max-prompt-len "$MAX_PROMPT_LEN" --max-new-tokens "$MAX_NEW_TOKENS" \
    --steps "$STEPS" --lr "$LR" --topk "$TOPK" \
    --entropy-threshold "$ENTROPY_THRESHOLD" --soft-kd-coef "$SOFT_KD_COEF" \
    --clip-ratio "$CLIP_RATIO" --save-every "$SAVE_EVERY" --log-every 10 \
    --out-dir "$out_dir" --seed "$seed"
