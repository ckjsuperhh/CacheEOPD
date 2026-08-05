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

export PYTHONPATH="$project_root"
mkdir -p "$run_root" "$log_root"
printf 'parallel_started_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"

record_status() {
    printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$log_root/status.tsv"
}

evaluate_checkpoints() {
    local method=$1
    local seed=$2
    local gpu=$3
    local output_dir=$4
    local checkpoint_dir
    local final_path
    local partial_path
    local log_path
    local step
    for step in 50 100 150 200 250 300; do
        checkpoint_dir="$output_dir/student_step${step}"
        final_path="$log_root/${method}_seed${seed}_student_step${step}.jsonl"
        partial_path="$final_path.partial"
        log_path="$final_path.log"
        if [[ ! -d "$checkpoint_dir" ]]; then
            continue
        fi
        if [[ -s "$final_path" ]]; then
            continue
        fi
        while [[ -e "$partial_path" ]]; do
            sleep 5
        done
        PYTHONPATH="$project_root" "$python_bin" -m cache_eopd.eval_student_batch \
            --student "$checkpoint_dir" --data-path "$eval_data" --device "cuda:$gpu" \
            --num-samples 500 --batch-size 8 --max-new-tokens 512 --out "$partial_path" \
            > "$log_path" 2>&1
        if [[ $? -eq 0 && -s "$partial_path" ]]; then
            mv "$partial_path" "$final_path"
        else
            record_status "$method seed=$seed step=$step" "eval_failed"
            return 1
        fi
    done
}

run_one() {
    local method=$1
    local seed=$2
    local gpu=$3
    local mode=$4
    local schedule=$5
    local output_dir="$run_root/${method}_seed${seed}"
    local train_log="$log_root/${method}_seed${seed}_gpu${gpu}.train.log"
    if [[ -f "$output_dir/DONE" ]]; then
        record_status "$method seed=$seed gpu=$gpu" "already_done"
        return 0
    fi
    mkdir -p "$output_dir"
    record_status "$method seed=$seed gpu=$gpu" "started"
    if [[ ! -d "$output_dir/student_step300" ]]; then
        if [[ "$mode" == plain ]]; then
            "$python_bin" -m cache_eopd.train_student_distill \
                --teacher "$teacher" --student "$student" --data-path "$trajectory" \
                --device "cuda:$gpu" --teacher-device "cuda:$gpu" --teacher-gpus "$gpu" \
                --mode plain --steps 300 --grad-accum 8 --lr 1e-5 \
                --save-every 50 --eval-every 50 --log-every 10 \
                --out-dir "$output_dir" --seed "$seed" > "$train_log" 2>&1
        else
            "$python_bin" -m cache_eopd.train_student_distill \
                --teacher "$teacher" --student "$student" --data-path "$trajectory" \
                --fuser-dir "$fuser" --proj-device "cuda:$gpu" --device "cuda:$gpu" \
                --teacher-device "cuda:$gpu" --teacher-gpus "$gpu" --mode "$mode" \
                --fused-prob 0.5 --anneal-start-prob 1.0 --anneal-end-prob 0.0 \
                --anneal-steps 300 --anneal-schedule "$schedule" --steps 300 \
                --grad-accum 8 --lr 1e-5 --save-every 50 --eval-every 50 \
                --log-every 10 --out-dir "$output_dir" --seed "$seed" \
                > "$train_log" 2>&1
        fi
    else
        record_status "$method seed=$seed gpu=$gpu" "training_checkpoint_exists"
    fi
    if [[ "$mode" == eopd ]]; then
        true
    fi
    if [[ "$mode" == eopd ]]; then
        record_status "$method seed=$seed gpu=$gpu" "unsupported_mode"
        return 1
    fi
    evaluate_checkpoints "$method" "$seed" "$gpu" "$output_dir"
    touch "$output_dir/DONE"
    record_status "$method seed=$seed gpu=$gpu" "done"
}

run_eopd() {
    local seed=$1
    local gpu=$2
    local method=eopd
    local output_dir="$run_root/${method}_seed${seed}"
    local train_log="$log_root/${method}_seed${seed}_gpu${gpu}.train.log"
    if [[ -f "$output_dir/DONE" ]]; then
        record_status "$method seed=$seed gpu=$gpu" "already_done"
        return 0
    fi
    mkdir -p "$output_dir"
    record_status "$method seed=$seed gpu=$gpu" "started"
    "$python_bin" -m cache_eopd.train_eopd_hf \
        --teacher "$teacher" --student "$student" --data-path "$trajectory" \
        --device "cuda:$gpu" --teacher-gpus "$gpu" --max-prompt-len 384 \
        --max-new-tokens 384 --steps 300 --lr 1e-6 --topk 16 \
        --entropy-threshold 0.8 --soft-kd-coef 1.0 --save-every 50 \
        --log-every 10 --out-dir "$output_dir" --seed "$seed" \
        > "$train_log" 2>&1
    if [[ $? -ne 0 ]]; then
        record_status "$method seed=$seed gpu=$gpu" "train_failed"
        return 1
    fi
    local checkpoint_dir
    local final_path
    local partial_path
    local step
    for step in 50 100 150 200 250 300; do
        checkpoint_dir="$output_dir/step${step}"
        final_path="$log_root/${method}_seed${seed}_step${step}.jsonl"
        partial_path="$final_path.partial"
        if [[ ! -d "$checkpoint_dir" || -s "$final_path" ]]; then
            continue
        fi
        PYTHONPATH="$project_root" "$python_bin" -m cache_eopd.eval_student_batch \
            --student "$checkpoint_dir" --data-path "$eval_data" --device "cuda:$gpu" \
            --num-samples 500 --batch-size 8 --max-new-tokens 512 --out "$partial_path" \
            > "$final_path.log" 2>&1
        if [[ $? -eq 0 && -s "$partial_path" ]]; then
            mv "$partial_path" "$final_path"
        else
            record_status "$method seed=$seed step=$step" "eval_failed"
            return 1
        fi
    done
    touch "$output_dir/DONE"
    record_status "$method seed=$seed gpu=$gpu" "done"
}

run_task() {
    if [[ "$1" == eopd ]]; then
        run_eopd "$2" "$3"
    else
        run_one "$1" "$2" "$3" "$4" "$5"
    fi
}

run_wave() {
    run_task "$1" "$2" "$3" "$4" "$5" &
    local first_pid=$!
    run_task "$6" "$7" "$8" "$9" "${10}" &
    local second_pid=$!
    run_task "${11}" "${12}" "${13}" "${14}" "${15}" &
    local third_pid=$!
    run_task "${16}" "${17}" "${18}" "${19}" "${20}" &
    local fourth_pid=$!
    wait "$first_pid" "$second_pid" "$third_pid" "$fourth_pid"
}

run_wave plain 41717 1 plain linear eopd 41717 2 eopd linear mixed 41717 4 mixed linear anneal_linear 41717 5 anneal linear
run_wave anneal_quadratic 41717 1 anneal quadratic anneal_sqrt 41717 2 anneal sqrt plain 41718 4 plain linear eopd 41718 5 eopd linear
run_wave mixed 41718 1 mixed linear anneal_linear 41718 2 anneal linear anneal_quadratic 41718 4 anneal quadratic anneal_sqrt 41718 5 anneal sqrt
run_wave plain 41719 1 plain linear eopd 41719 2 eopd linear mixed 41719 4 mixed linear anneal_linear 41719 5 anneal linear
run_task anneal_quadratic 41719 1 anneal quadratic &
run_task anneal_sqrt 41719 2 anneal sqrt &
wait

printf 'parallel_finished_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"
