#!/usr/bin/env bash
set -uo pipefail

python_bin=/home/knhdu/anaconda3/envs/rosetta/bin/python
project_root=/home/kejiechen/CacheEOPD
student=/home/kejiechen/taopd-baseline/modelweights/Qwen3-0.6B
teacher=/home/kejiechen/taopd-baseline/modelweights/Qwen3-4B
fuser=/home/kejiechen/taopd-baseline/modelweights/qwen3_0.6b+qwen3_4b_base_Fuser/final
trajectory=/home/kejiechen/CacheEOPD/data/teacher_traj_gsm8k1500.jsonl
eval_data=/home/kejiechen/CacheEOPD/data/gsm8k_cot_test_500.jsonl
run_root=/dev/shm/cacheeopd/multiseed_official_20260804
log_root=/home/kejiechen/CacheEOPD/logs/multiseed_official_20260804
seeds=(41717 41718 41719)

export PYTHONPATH="$project_root"
mkdir -p "$run_root" "$log_root"
printf 'started_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"

record_status() {
    printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$log_root/status.tsv"
}

evaluate_checkpoints() {
    local method=$1
    local seed=$2
    local output_dir=$3
    local checkpoint_name
    local checkpoint_dir
    local evaluation_path
    for checkpoint_name in student_step50 student_step100 student_step150 student_step200 student_step250 student_step300 step50 step100 step150 step200 step250 step300; do
        checkpoint_dir="$output_dir/$checkpoint_name"
        if [[ ! -d "$checkpoint_dir" ]]; then
            continue
        fi
        evaluation_path="$log_root/${method}_seed${seed}_${checkpoint_name}.jsonl"
        if [[ -s "$evaluation_path" ]]; then
            continue
        fi
        "$python_bin" -m cache_eopd.eval_student_batch \
            --student "$checkpoint_dir" --data-path "$eval_data" --device cuda:1 \
            --num-samples 500 --batch-size 8 --max-new-tokens 512 \
            --out "$evaluation_path" > "$evaluation_path.log" 2>&1
    done
}

run_student_distill() {
    local method=$1
    local seed=$2
    local mode=$3
    local schedule=$4
    local output_dir="$run_root/${method}_seed${seed}"
    local train_log="$log_root/${method}_seed${seed}.train.log"
    if [[ -f "$output_dir/DONE" ]]; then
        record_status "$method seed=$seed" "already_done"
        return 0
    fi
    mkdir -p "$output_dir"
    record_status "$method seed=$seed" "started"
    if [[ "$mode" == plain ]]; then
        "$python_bin" -m cache_eopd.train_student_distill \
            --teacher "$teacher" --student "$student" --data-path "$trajectory" \
            --device cuda:1 --teacher-device auto --teacher-gpus 2,3 --mode plain \
            --steps 300 --grad-accum 8 --lr 1e-5 --save-every 50 --eval-every 50 \
            --log-every 10 --out-dir "$output_dir" --seed "$seed" \
            > "$train_log" 2>&1
    else
        "$python_bin" -m cache_eopd.train_student_distill \
            --teacher "$teacher" --student "$student" --data-path "$trajectory" \
            --fuser-dir "$fuser" --proj-device cuda:4 --device cuda:1 \
            --teacher-device auto --teacher-gpus 2,3 --mode "$mode" --fused-prob 0.5 \
            --anneal-start-prob 1.0 --anneal-end-prob 0.0 --anneal-steps 300 \
            --anneal-schedule "$schedule" --steps 300 --grad-accum 8 --lr 1e-5 \
            --save-every 50 --eval-every 50 --log-every 10 --out-dir "$output_dir" \
            --seed "$seed" > "$train_log" 2>&1
    fi
    if [[ $? -ne 0 ]]; then
        record_status "$method seed=$seed" "train_failed"
        return 1
    fi
    evaluate_checkpoints "$method" "$seed" "$output_dir"
    touch "$output_dir/DONE"
    record_status "$method seed=$seed" "done"
}

run_eopd() {
    local seed=$1
    local method=eopd
    local output_dir="$run_root/${method}_seed${seed}"
    local train_log="$log_root/${method}_seed${seed}.train.log"
    if [[ -f "$output_dir/DONE" ]]; then
        record_status "$method seed=$seed" "already_done"
        return 0
    fi
    mkdir -p "$output_dir"
    record_status "$method seed=$seed" "started"
    "$python_bin" -m cache_eopd.train_eopd_hf \
        --teacher "$teacher" --student "$student" --data-path "$trajectory" \
        --device cuda:1 --teacher-gpus 2,3 --max-prompt-len 384 --max-new-tokens 384 \
        --steps 300 --lr 1e-6 --topk 16 --entropy-threshold 0.8 --soft-kd-coef 1.0 \
        --save-every 50 --log-every 10 --out-dir "$output_dir" --seed "$seed" \
        > "$train_log" 2>&1
    if [[ $? -ne 0 ]]; then
        record_status "$method seed=$seed" "train_failed"
        return 1
    fi
    evaluate_checkpoints "$method" "$seed" "$output_dir"
    touch "$output_dir/DONE"
    record_status "$method seed=$seed" "done"
}

for seed in "${seeds[@]}"; do
    run_student_distill plain "$seed" plain linear || true
    run_eopd "$seed" || true
    run_student_distill mixed "$seed" mixed linear || true
    run_student_distill anneal_linear "$seed" anneal linear || true
    run_student_distill anneal_quadratic "$seed" anneal quadratic || true
    run_student_distill anneal_sqrt "$seed" anneal sqrt || true
done

printf 'finished_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"
