#!/usr/bin/env bash
set -euo pipefail

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
: "${PROJECTOR_DATA:?set PROJECTOR_DATA in .env}"
: "${PROJECTOR_OUT:?set PROJECTOR_OUT in .env}"
: "${PROJECTOR_DEVICE:=cuda:0}"
: "${PROJECTOR_TEACHER_DEVICE:=auto}"
: "${PROJECTOR_TEACHER_GPUS:=1,2,3,4,5,6,7}"
: "${PROJECTOR_STEPS:=15000}"
: "${PROJECTOR_SAVE_EVERY:=500}"
: "${PROJECTOR_EVAL_EVERY:=100}"
: "${PROJECTOR_HOLDOUT:=5000}"
: "${PROJECTOR_MAX_PROMPT:=2048}"
: "${PROJECTOR_MAX_ANSWER:=8192}"
: "${PROJECTOR_LR:=1e-4}"
: "${PROJECTOR_GRAD_ACCUM:=8}"
: "${PROJECTOR_SEED:=42}"

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m cache_eopd.train_projector \
    --teacher "$TEACHER_PATH" \
    --student "$STUDENT_PATH" \
    --data-path "$PROJECTOR_DATA" \
    --device "$PROJECTOR_DEVICE" \
    --teacher-device "$PROJECTOR_TEACHER_DEVICE" \
    --teacher-gpus "$PROJECTOR_TEACHER_GPUS" \
    --max-prompt-len "$PROJECTOR_MAX_PROMPT" \
    --max-answer-len "$PROJECTOR_MAX_ANSWER" \
    --holdout "$PROJECTOR_HOLDOUT" \
    --steps "$PROJECTOR_STEPS" \
    --grad-accum "$PROJECTOR_GRAD_ACCUM" \
    --lr "$PROJECTOR_LR" \
    --warmup-steps 1000 \
    --anneal-steps "$PROJECTOR_STEPS" \
    --eval-every "$PROJECTOR_EVAL_EVERY" \
    --save-every "$PROJECTOR_SAVE_EVERY" \
    --projector-hidden 1024 \
    --projector-inter 1024 \
    --projector-layers 3 \
    --gate-init 1.0 \
    --gate-lr-mult 20.0 \
    --per-layer \
    --zero-init \
    --layer-mapping last_aligned \
    --seed "$PROJECTOR_SEED" \
    --out-dir "$PROJECTOR_OUT"
