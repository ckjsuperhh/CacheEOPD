#!/bin/bash
# 【阶段七】早停/过拟合诊断：把 fused 与 plain 学生更早的 checkpoint
# (step100 / step200) 单独拿出来做 GSM8K 评测，看是否比 step300 更好。
# 注意：eval_math_acc 在不带 --projector-path 时只跑 baseline 臂（学生自身 KV），
# 所以这里测的是「该 checkpoint 本身的泛化能力」，与训练时是否融合 teacher KV 无关。
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

run_one fused_step100  $BASE/ckpt_student_fused/student_step100
run_one fused_step200  $BASE/ckpt_student_fused/student_step200
run_one plain_step100  $BASE/ckpt_student_plain/student_step100
run_one plain_step200  $BASE/ckpt_student_plain/student_step200
echo "ALL EARLY-CKPT EVAL DONE $(date)"
