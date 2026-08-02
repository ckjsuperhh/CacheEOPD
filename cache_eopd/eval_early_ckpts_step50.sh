#!/bin/bash
# 【阶段七·过拟合诊断】补充 step50：fused / plain 学生在 step50 的 standalone GSM8K 评测。
# 用于把最优点卡在 <100 步区间（若 step100>step200>step300，最优点可能在 step50 甚至更早）。
set -e
PY=/home/knhdu/anaconda3/envs/rosetta/bin/python
export PYTHONPATH=/home/kejiechen/CacheEOPD
TEACHER=/home/kejiechen/taopd-baseline/modelweights/Qwen3-4B
DATA=/home/kejiechen/taopd-baseline/data/GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl
LOGDIR=/home/kejiechen/CacheEOPD/logs
BASE=/home/kejiechen/CacheEOPD

run_one () {
  local name=$1; local student=$2
  echo "===== $name 开始 $(date) ====="
  $PY -m cache_eopd.eval_math_acc \
    --teacher $TEACHER --student $student --data-path $DATA \
    --device cuda:1 --teacher-device auto --teacher-gpus 2,3 \
    --num-samples 300 --max-new-tokens 512 --sanity 0 \
    --out $LOGDIR/eval_${name}.jsonl 2>&1 | tee $LOGDIR/eval_${name}.log
  echo "===== $name 结束 $(date) ====="
}

run_one fused_step50  $BASE/ckpt_student_fused/student_step50
run_one plain_step50  $BASE/ckpt_student_plain/student_step50
echo "ALL STEP50 EVAL DONE $(date)"
