#!/bin/bash
# CollabLLM 监督微调（SFT）训练脚本
#
# 用途：对 CollabLLM 进行监督微调（Supervised Fine-Tuning），
# 使用预先收集的高质量多轮对话数据训练模型。
# 启动的训练任务：通过 verl 框架的 fsdp_sft_trainer 入口启动 FSDP 分布式 SFT 训练。
#
# 关键参数含义：
# - nproc_per_node: 每个节点的 GPU 数量（第一个参数）
# - data.train_files/data.val_files: 训练/验证数据文件路径（parquet 格式）
# - data.multiturn.enable=true: 启用多轮对话模式
# - model.partial_pretrain: 预训练模型路径
# - trainer.total_epochs: 训练总轮数
#
# 使用方式：bash train_sft_collabllm.sh <nproc_per_node> [其他配置...]

set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: sft_train_collabllm.sh [<nproc_per_node> other_configs...]"
    exit 1
fi

nproc_per_node=$1

# Shift the arguments so $@ refers to the rest
shift 1

DATASET=math-hard-large

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/collabllm-$DATASET/sft_train.parquet \
    data.val_files=$HOME/data/collabllm-$DATASET/sft_validation.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    optim.lr=1e-6 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=8196 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.project_name=collabllm-sft-$DATASET \
    trainer.experiment_name=collabllm-sft-qwen2.5-7B-$DATASET \
    trainer.logger=console \
    trainer.total_epochs=3 $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true $@