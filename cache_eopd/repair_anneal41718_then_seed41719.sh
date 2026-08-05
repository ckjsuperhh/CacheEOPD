#!/usr/bin/env bash
set -uo pipefail

python_bin=/home/knhdu/anaconda3/envs/rosetta/bin/python
root=/home/kejiechen/CacheEOPD
run_root=/dev/shm/cacheeopd/correct_eopd_cacheeopd_20260805
log_root=$root/logs/correct_eopd_cacheeopd_20260805
student=/home/kejiechen/taopd-baseline/modelweights/Qwen3-0.6B
teacher=/home/kejiechen/taopd-baseline/modelweights/Qwen3-4B
fuser=/home/kejiechen/taopd-baseline/modelweights/qwen3_0.6b+qwen3_4b_base_Fuser/final
trajectory=$root/data/teacher_traj_gsm8k1500.jsonl
eval_data=$root/data/gsm8k_cot_test_500.jsonl
tag=cacheeopd_anneal_seed41718
dir=$run_root/$tag

cd "$root"
export PYTHONPATH=$root
printf '%s\t%s\t%s\n' "$(date -Is)" "$tag gpu=2" repair_started >> "$log_root/status.tsv"
"$python_bin" -m cache_eopd.train_eopd_cacheeopd_hf \
    --teacher "$teacher" --student "$student" --data-path "$trajectory" --fuser-dir "$fuser" \
    --device cuda:2 --teacher-gpus 2 --proj-device cuda:2 --mode anneal \
    --anneal-start-prob 1.0 --anneal-end-prob 0.0 --anneal-steps 300 --anneal-schedule linear \
    --max-prompt-len 384 --max-new-tokens 384 --steps 300 --lr 1e-6 --topk 16 \
    --entropy-threshold 0.8 --soft-kd-coef 1.0 --clip-ratio 0.2 --save-every 50 --log-every 10 \
    --out-dir "$dir" --seed 41718 > "$log_root/${tag}_repair_gpu2.train.log" 2>&1
if [[ $? -ne 0 ]]; then
    printf '%s\t%s\t%s\n' "$(date -Is)" "$tag gpu=2" repair_failed >> "$log_root/status.tsv"
    exit 1
fi
for step in 50 100 150 200 250 300; do
    final="$log_root/${tag}_step${step}.jsonl"
    partial="$final.partial"
    [[ -s "$final" ]] && continue
    "$python_bin" -m cache_eopd.eval_student_batch --student "$dir/step${step}" \
        --data-path "$eval_data" --device cuda:2 --num-samples 500 --batch-size 8 \
        --max-new-tokens 512 --out "$partial" > "$final.log" 2>&1 || exit 1
    mv "$partial" "$final"
done
touch "$dir/DONE"
printf '%s\t%s\t%s\n' "$(date -Is)" "$tag gpu=2" repaired_done >> "$log_root/status.tsv"
cd "$root"
nohup bash cache_eopd/run_seed41719_all.sh > logs/seed41719_all_20260805/runner.log 2>&1 &
echo "seed41719 runner started: $!" >> logs/seed41719_all_20260805/runner.log
