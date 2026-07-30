#!/usr/bin/env bash
# =============================================================================
# DAPO 训练数据准备脚本
#
# 用途：从 HuggingFace 下载 DAPO 算法所需的训练和测试数据集。
#
# 下载的数据集：
#   - 训练集：DAPO-Math-17k（17k 道数学题，用于 RL 训练）
#   - 测试集：AIME-2024（AIME 2024 竞赛题，用于验证）
#
# 关键参数（可通过环境变量覆盖）：
#   VERL_HOME: verl 数据根目录，默认 ~/verl
#   TRAIN_FILE: 训练数据文件路径
#   TEST_FILE: 测试数据文件路径
#   OVERWRITE: 设为 1 可强制覆盖已有文件
#
# 使用方式：bash prepare_dapo_data.sh
# =============================================================================
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"${HOME}/verl"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi
