# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Megatron 后端的词汇并行 KL 散度损失计算模块。

该模块实现了在张量并行（Tensor Parallel）环境下的 KL 散度损失计算，
用于 GKD（知识蒸馏）中学生模型对教师模型 top-k 分布的学习。

核心实现：
- _VocabParallelKLDivergence: 自定义 autograd Function，支持前向/反向传播
- vocab_parallel_kl_divergence: 对外接口函数
- KL(P||Q) 中 P 为教师分布（target），Q 为学生分布（source）
- 使用稀疏 top-k 表示教师分布以减少通信和计算开销
"""

import torch
from megatron.core.fusions.fused_cross_entropy import calculate_logits_max
from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.utils import VocabUtility


def normalize(logps):
    """对对数概率进行归一化处理。
    将 logps 转换为概率后重新归一化，再转回对数概率，
    确保概率分布之和为 1。
    """
    probs = torch.exp(logps)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    normalized_logps = torch.log(probs)
    return normalized_logps


def mylog(message):
    """调试用日志函数，将消息写入 kl_loss.log 文件。
    记录当前数据并行秩和张量并行秩信息。
    """
    with open("kl_loss.log", "a") as f:
        f.write(f"({get_data_parallel_rank()}, {get_tensor_model_parallel_rank()}): {message}\n")


class _VocabParallelKLDivergence(torch.autograd.Function):
    """词汇并行 KL 散度的自定义 autograd 函数。

    在张量并行环境下计算学生模型（source）与教师模型（target）之间的 KL 散度。
    前向传播：计算 KL(P||Q) = sum(P * (logP - logQ))，其中 P 为教师 top-k 分布，Q 为学生分布。
    反向传播：梯度为 (student_softmax - teacher_sparse_probs) * grad_output。

    教师分布使用稀疏 top-k 表示，仅传递概率最高的 k 个 token 的 logps 和索引，
    以减少跨进程通信量。
    """
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_topk_logps, target_topk_indices):
        # 前向传播：计算词汇并行的 KL 散度损失
        # 1. 计算学生模型的 softmax 概率（数值稳定版本：减去最大值）
        # 2. 跨 TP 组 all-reduce 得到全局最大值和 exp 之和
        # 3. 提取当前 TP 分区对应的教师 top-k 概率和索引
        # 4. 计算 KL(P||Q)，P 为教师分布，Q 为学生分布
        # seq_len, batch_size, top_k = target_topk_logps.size()
        # target_topk_logps = normalize(target_topk_logps)
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        partition_vocab_size = vocab_parallel_logits.size(-1)

        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
        )

        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        vocab_parallel_logits.exp_()
        exp_logits = vocab_parallel_logits
        sum_exp_logits = exp_logits.sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Get the partition's vocab indices
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        topk_indices_in_vocab_mask = (target_topk_indices >= vocab_start_index) & (
            target_topk_indices < vocab_end_index
        )

        vocab_parallel_target_topk_indices = target_topk_indices - vocab_start_index
        vocab_parallel_target_topk_indices[~topk_indices_in_vocab_mask] = 0
        vocab_parallel_target_topk_probs = torch.exp(target_topk_logps)
        vocab_parallel_target_topk_probs[~topk_indices_in_vocab_mask] = 0
        vocab_parallel_target_topk_logps = torch.empty_like(target_topk_logps)
        vocab_parallel_target_topk_logps[...] = target_topk_logps[...]
        vocab_parallel_target_topk_logps[~topk_indices_in_vocab_mask] = 0
        # assert ((0 <= target_topk_indices) & (target_topk_indices < partition_vocab_size)).all()

        # bs, sl, topk = target_topk_indices.shape
        target_topk_logps_origin_shape = target_topk_indices.shape
        topk = target_topk_indices.size(-1)

        vocab_parallel_source_probs = exp_logits
        vocab_parallel_source_probs.div_(sum_exp_logits.unsqueeze(-1))
        vocab_parallel_source_probs_2d = vocab_parallel_source_probs.view(-1, partition_vocab_size)  # (b*s, h/tp)

        arange_1d = torch.arange(
            start=0, end=vocab_parallel_source_probs_2d.size(0), device=vocab_parallel_source_probs_2d.device
        )  # (b*s, )
        vocab_parallel_source_topk_probs_2d = vocab_parallel_source_probs_2d[
            arange_1d.unsqueeze(-1), vocab_parallel_target_topk_indices.view(-1, topk)
        ]  # (b*s, topk)
        vocab_parallel_source_topk_probs = vocab_parallel_source_topk_probs_2d.view(
            target_topk_logps_origin_shape
        )  # (b, s, topk)
        vocab_parallel_source_topk_logps = torch.log(1e-20 + vocab_parallel_source_topk_probs)
        vocab_parallel_source_topk_logps[~topk_indices_in_vocab_mask] = 0

        # KL(P||Q)会强制 Q 覆盖 P 的所有模式（避免漏峰）
        # KL(Q||P)会鼓励 Q 聚焦于 P 的一个模式（避免多峰)
        # 这里使用 KL(P||Q)，其中P为target，Q为source，鼓励source学习target的所有模式

        per_token_kl_loss = torch.sum(
            vocab_parallel_target_topk_probs * (vocab_parallel_target_topk_logps - vocab_parallel_source_topk_logps),
            dim=-1,
        )  # (b, s)

        # if torch.isinf(per_token_kl_loss).any() or torch.isnan(per_token_kl_loss).any():
        #     breakpoint()

        torch.distributed.all_reduce(
            per_token_kl_loss,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        ctx.save_for_backward(
            vocab_parallel_source_probs, vocab_parallel_target_topk_probs, vocab_parallel_target_topk_indices
        )
        # if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 1:
        #     import ipdb; ipdb.set_trace()
        # torch.distributed.barrier()
        return per_token_kl_loss

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：计算 KL 散度对学生 logits 的梯度。
        梯度公式为：(student_softmax_probs - teacher_sparse_probs) * grad_output
        其中 teacher_sparse_probs 仅在 top-k 位置非零。
        """
        vocab_parallel_source_probs, vocab_parallel_target_topk_probs, vocab_parallel_target_topk_indices = (
            ctx.saved_tensors
        )
        # source_probs, target_probs = ctx.saved_tensors
        # KL 散度对 vocab_parallel_logits 的梯度为: (student_softmax_logits - valid_target_logits)
        grad_input = vocab_parallel_source_probs  # shape: [seq_len, batch_size, vocab_parition_size]

        topk = vocab_parallel_target_topk_indices.size(-1)
        grad_input_2d = grad_input.view(-1, grad_input.size(-1))
        arange_1d = torch.arange(start=0, end=grad_input_2d.size(0), device=grad_input_2d.device)  # (b*s, )
        grad_input_2d[arange_1d.unsqueeze(-1), vocab_parallel_target_topk_indices.view(-1, topk)] -= (
            vocab_parallel_target_topk_probs.view(-1, topk)
        )

        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None  # 返回给第一个输入 vocab_parallel_logits 的梯度


def vocab_parallel_kl_divergence(vocab_parallel_logits, target_topk_logps, target_topk_indices):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks.
    在张量并行环境下计算 KL 散度损失。

    当 logits 被分片到不同的张量并行 rank 上时，使用此函数计算
    学生模型与教师模型（top-k 稀疏表示）之间的 KL 散度。

    Args:
        vocab_parallel_logits: 分布在各个 TP rank 上的学生 logits,
                               形状为 [sequence_length, batch_size, vocab_size_per_partition]
        target_topk_logps: 教师模型的 top-k 对数概率,
                           形状为 [sequence_length, batch_size, top_k]
        target_topk_indices: 教师模型 top-k 对应的 token 索引,
                             形状为 [sequence_length, batch_size, top_k]

    Returns:
        loss: 每个 token 的 KL 散度损失，标量张量
    """
    return _VocabParallelKLDivergence.apply(vocab_parallel_logits, target_topk_logps, target_topk_indices)
