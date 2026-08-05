#!/usr/bin/env bash
set -uo pipefail

python_bin=/home/knhdu/anaconda3/envs/rosetta/bin/python
project_root=/home/kejiechen/CacheEOPD
student=/home/kejiechen/taopd-baseline/modelweights/Qwen3-0.6B
teacher=/home/kejiechen/taopd-baseline/modelweights/Qwen3-4B
fuser=/home/kejiechen/taopd-baseline/modelweights/qwen3_0.6b+qwen3_4b_base_Fuser/final
trajectory=/home/kejiechen/CacheEOPD/data/teacher_traj_gsm8k1500.jsonl
eval_data=/home/kejiechen/CacheEOPD/data/gsm8k_cot_test_500.jsonl
run_root=/dev/shm/cacheeopd/correct_eopd_cacheeopd_v2_20260805
log_root=/home/kejiechen/CacheEOPD/logs/correct_eopd_cacheeopd_v2_20260805

export PYTHONPATH="$project_root"
mkdir -p "$run_root" "$log_root"
printf 'started_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"

record_status() {
    printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$log_root/status.tsv"
}

evaluate_checkpoints() {
    local tag=$1
    local gpu=$2
    local output_dir=$3
    local step checkpoint_dir final_path partial_path log_path
    for step in 50 100 150 200 250 300; do
        checkpoint_dir="$output_dir/step${step}"
        final_path="$log_root/${tag}_step${step}.jsonl"
        partial_path="$final_path.partial"
        log_path="$final_path.log"
        if [[ ! -d "$checkpoint_dir" || -s "$final_path" ]]; then
            continue
        fi
        while [[ -e "$partial_path" ]]; do
            sleep 5
        done
        "$python_bin" -m cache_eopd.eval_student_batch \
            --student "$checkpoint_dir" --data-path "$eval_data" --device "cuda:$gpu" \
            --num-samples 500 --batch-size 8 --max-new-tokens 512 --out "$partial_path" \
            > "$log_path" 2>&1
        if [[ $? -eq 0 && -s "$partial_path" ]]; then
            mv "$partial_path" "$final_path"
        else
            record_status "$tag step=$step" "eval_failed"
            return 1
        fi
    done
}

run_one() {
    local method=$1
    local seed=$2
    local gpu=$3
    local tag="cacheeopd_${method}_seed${seed}"
    local output_dir="$run_root/$tag"
    local train_log="$log_root/${tag}_gpu${gpu}.train.log"
    if [[ -f "$output_dir/DONE" ]]; then
        record_status "$tag gpu=$gpu" "already_done"
        return 0
    fi
    mkdir -p "$output_dir"
    record_status "$tag gpu=$gpu" "started"
    local strategy_args=()
    case "$method" in
        fused)
            strategy_args=(--mode fused)
            ;;
        mixed)
            strategy_args=(--mode mixed --fused-prob 0.5)
            ;;
        anneal)
            strategy_args=(--mode anneal --anneal-start-prob 1.0 --anneal-end-prob 0.0 \
                --anneal-steps 300 --anneal-schedule linear)
            ;;
        *)
            record_status "$tag gpu=$gpu" "invalid_method"
            return 1
            ;;
    esac
    "$python_bin" -m cache_eopd.train_eopd_cacheeopd_hf \
        --teacher "$teacher" --student "$student" --data-path "$trajectory" \
        --fuser-dir "$fuser" --device "cuda:$gpu" --teacher-gpus "$gpu" \
        --proj-device "cuda:$gpu" "${strategy_args[@]}" \
        --max-prompt-len 384 --max-new-tokens 384 --steps 300 --lr 1e-6 \
        --topk 16 --entropy-threshold 0.8 --soft-kd-coef 1.0 --clip-ratio 0.2 \
        --save-every 50 --log-every 10 --out-dir "$output_dir" --seed "$seed" \
        > "$train_log" 2>&1
    if [[ $? -ne 0 ]]; then
        record_status "$tag gpu=$gpu" "train_failed"
        return 1
    fi
    evaluate_checkpoints "$tag" "$gpu" "$output_dir" || return 1
    touch "$output_dir/DONE"
    record_status "$tag gpu=$gpu" "done"
}

run_one fused 41717 1 & p1=$!
run_one mixed 41717 2 & p2=$!
run_one anneal 41717 4 & p3=$!
run_one fused 41718 5 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

run_one mixed 41718 1 & p5=$!
run_one anneal 41718 2 & p6=$!
wait "$p5" "$p6"
printf 'finished_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"
