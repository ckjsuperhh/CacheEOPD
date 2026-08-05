#!/usr/bin/env bash
set -uo pipefail

project_root=/home/kejiechen/CacheEOPD
source_status=/home/kejiechen/CacheEOPD/logs/correct_eopd_cacheeopd_v2_20260805/status.tsv
while ! grep -q '^finished_at=' "$source_status" 2>/dev/null; do
    sleep 30
done
cd "$project_root"
mkdir -p logs/seed41719_all_v2_20260805
nohup bash cache_eopd/run_seed41719_all.sh > logs/seed41719_all_v2_20260805/runner.log 2>&1 &
echo "seed41719 runner started: $!" >> logs/seed41719_all_v2_20260805/runner.log
