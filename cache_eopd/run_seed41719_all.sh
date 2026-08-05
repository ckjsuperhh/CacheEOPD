#!/usr/bin/env bash
set -uo pipefail

python_bin=/home/knhdu/anaconda3/envs/rosetta/bin/python
project_root=/home/kejiechen/CacheEOPD
student=/home/kejiechen/taopd-baseline/modelweights/Qwen3-0.6B
teacher=/home/kejiechen/taopd-baseline/modelweights/Qwen3-4B
fuser=/home/kejiechen/taopd-baseline/modelweights/qwen3_0.6b+qwen3_4b_base_Fuser/final
trajectory=/home/kejiechen/CacheEOPD/data/teacher_traj_gsm8k1500.jsonl
eval_data=/home/kejiechen/CacheEOPD/data/gsm8k_cot_test_500.jsonl
run_root=/dev/shm/cacheeopd/seed41719_all_v2_20260805
log_root=/home/kejiechen/CacheEOPD/logs/seed41719_all_v2_20260805
export PYTHONPATH="$project_root"
mkdir -p "$run_root" "$log_root"
printf 'started_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"

record_status() { printf '%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" >> "$log_root/status.tsv"; }

eval_student_steps() {
    local tag=$1 gpu=$2 dir=$3 step final partial
    for step in 50 100 150 200 250 300; do
        final="$log_root/${tag}_step${step}.jsonl"
        partial="$final.partial"
        [[ -d "$dir/step${step}" && ! -s "$final" ]] || continue
        while [[ -e "$partial" ]]; do sleep 5; done
        "$python_bin" -m cache_eopd.eval_student_batch \
            --student "$dir/step${step}" --data-path "$eval_data" --device "cuda:$gpu" \
            --num-samples 500 --batch-size 8 --max-new-tokens 512 --out "$partial" \
            > "$final.log" 2>&1
        if [[ $? -eq 0 && -s "$partial" ]]; then mv "$partial" "$final"; else return 1; fi
    done
}

run_plain() {
    local gpu=5 tag=plain_seed41719 dir="$run_root/plain_seed41719"
    record_status "$tag gpu=$gpu" started
    "$python_bin" -m cache_eopd.train_student_distill \
        --teacher "$teacher" --student "$student" --data-path "$trajectory" \
        --device cuda:$gpu --teacher-device cuda:$gpu --teacher-gpus "$gpu" --mode plain \
        --steps 300 --grad-accum 8 --lr 1e-5 --save-every 50 --eval-every 50 \
        --log-every 10 --out-dir "$dir" --seed 41719 > "$log_root/${tag}_gpu${gpu}.train.log" 2>&1 || return 1
    eval_student_steps "$tag" "$gpu" "$dir" || return 1
    touch "$dir/DONE"; record_status "$tag gpu=$gpu" done
}

run_eopd() {
    local gpu=1 tag=eopd_seed41719 dir="$run_root/eopd_seed41719" step final partial
    record_status "$tag gpu=$gpu" started
    "$python_bin" -m cache_eopd.train_eopd_hf \
        --teacher "$teacher" --student "$student" --data-path "$trajectory" \
        --device cuda:$gpu --teacher-gpus "$gpu" --max-prompt-len 384 --max-new-tokens 384 \
        --steps 300 --lr 1e-6 --topk 16 --entropy-threshold 0.8 --soft-kd-coef 1.0 \
        --save-every 50 --log-every 10 --out-dir "$dir" --seed 41719 \
        > "$log_root/${tag}_gpu${gpu}.train.log" 2>&1 || return 1
    for step in 50 100 150 200 250 300; do
        final="$log_root/${tag}_step${step}.jsonl"; partial="$final.partial"
        [[ -d "$dir/step${step}" && ! -s "$final" ]] || continue
        "$python_bin" -m cache_eopd.eval_student_batch --student "$dir/step${step}" \
            --data-path "$eval_data" --device cuda:$gpu --num-samples 500 --batch-size 8 \
            --max-new-tokens 512 --out "$partial" > "$final.log" 2>&1 || return 1
        mv "$partial" "$final"
    done
    touch "$dir/DONE"; record_status "$tag gpu=$gpu" done
}

run_cache() {
    local method=$1
    local gpu=$2
    local tag="cacheeopd_${method}_seed41719"
    local dir="$run_root/$tag"
    local args=(--mode "$method")
    [[ "$method" == mixed ]] && args+=(--fused-prob 0.5)
    [[ "$method" == anneal ]] && args+=(--anneal-start-prob 1.0 --anneal-end-prob 0.0 --anneal-steps 300 --anneal-schedule linear)
    record_status "$tag gpu=$gpu" started
    "$python_bin" -m cache_eopd.train_eopd_cacheeopd_hf \
        --teacher "$teacher" --student "$student" --data-path "$trajectory" --fuser-dir "$fuser" \
        --device cuda:$gpu --teacher-gpus "$gpu" --proj-device cuda:$gpu "${args[@]}" \
        --max-prompt-len 384 --max-new-tokens 384 --steps 300 --lr 1e-6 --topk 16 \
        --entropy-threshold 0.8 --soft-kd-coef 1.0 --clip-ratio 0.2 --save-every 50 \
        --log-every 10 --out-dir "$dir" --seed 41719 > "$log_root/${tag}_gpu${gpu}.train.log" 2>&1 || return 1
    eval_student_steps "$tag" "$gpu" "$dir" || return 1
    touch "$dir/DONE"; record_status "$tag gpu=$gpu" done
}

run_cache fused 1 & p1=$!
run_cache mixed 2 & p2=$!
run_cache anneal 4 & p3=$!
run_plain & p4=$!
wait "$p1" "$p2" "$p3" "$p4"
run_eopd
printf 'finished_at=%s\n' "$(date -Is)" >> "$log_root/status.tsv"
