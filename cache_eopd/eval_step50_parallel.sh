#!/bin/bash
# 【阶段七·过拟合诊断】step50 加速：跑在 GPU4/5，与主链(GPU1)并行，
# 尽早拿到用户最关心的 step50 数据。teacher 在本评测(baseline 臂)不被前向，
# 仅占位，故 GPU4/5 余量足够。
set -e
PY=/home/knhdu/anaconda3/envs/rosetta/bin/python
export PYTHONPATH=/home/kejiechen/CacheEOPD
TEACHER=/home/kejiechen/taopd-baseline/modelweights/Qwen3-4B
DATA=/home/kejiechen/taopd-baseline/data/GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl
LOGDIR=/home/kejiechen/CacheEOPD/logs
BASE=/home/kejiechen/CacheEOPD

run_one () {
  local name=$1; local student=$2; local dev=$3; local tg=$4
  echo "===== $name 开始 $(date) (device=$dev teacher=$tg) ====="
  $PY -m cache_eopd.eval_math_acc \
    --teacher $TEACHER --student $student --data-path $DATA \
    --device $dev --teacher-device auto --teacher-gpus $tg \
    --num-samples 300 --max-new-tokens 512 --sanity 0 \
    --out $LOGDIR/eval_${name}.jsonl 2>&1 | tee $LOGDIR/eval_${name}.log
  echo "===== $name 结束 $(date) ====="
}

run_one fused_step50  $BASE/ckpt_student_fused/student_step50  cuda:4 4,5
run_one plain_step50  $BASE/ckpt_student_plain/student_step50  cuda:4 4,5
echo "STEP50 PARALLEL DONE $(date)"
