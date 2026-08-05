#!/usr/bin/env bash
set -euo pipefail

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
: "${EVAL_DATA:?set EVAL_DATA in .env}"
: "${DEVICE:?set DEVICE in .env}"
: "${OUT_ROOT:?set OUT_ROOT in .env}"

checkpoint=${1:-$STUDENT}
tag=${2:-base}
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_ROOT/rollout_gain"

exec "$PYTHON_BIN" -m cache_eopd.eval_math_acc \
    --student "$checkpoint" --teacher "$TEACHER" --fuser-dir "$FUSER_DIR" \
    --data-path "$EVAL_DATA" --device "$DEVICE" --teacher-device "$DEVICE" \
    --teacher-gpus "${TEACHER_GPUS:-${DEVICE#cuda:}}" --num-samples 500 \
    --max-prompt-len "${MAX_PROMPT_LEN:-384}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-512}" --layer-mapping last_aligned \
    --sanity 5 --out "$OUT_ROOT/rollout_gain/${tag}_plain_vs_fused.jsonl"
