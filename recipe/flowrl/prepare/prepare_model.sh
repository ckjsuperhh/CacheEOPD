#!/bin/bash
# ============================================================
# 脚本说明: 下载 FlowRL 训练所需的预训练模型（默认 Qwen2.5-7B）。
# ============================================================

MODEL_NAME=Qwen/Qwen2.5-7B

huggingface-cli download $MODEL_NAME \
  --repo-type model \
  --resume-download \
  --local-dir downloads/models/$MODEL_NAME \
  --local-dir-use-symlinks False \
  --exclude *.pth