# Copyright 2024 PRIME team and/or its affiliates
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
PRIME 核心算法模块。

本模块实现了 PRIME 方法中的核心算法，包括：
  - RLOO（Reinforcement Learning with Leave-One-Out）优势函数估计
  - 多种 DPO 风格的奖励模型损失函数（CE、Detach-DPO、BoN）
  - DPO 准确率评估指标

这些算法在 PRIME 框架中用于：
  1. 计算在线奖励模型的过程奖励信号
  2. 通过 RLOO 方法估计策略梯度中的优势函数
  3. 使用 DPO 风格损失在线更新奖励模型
"""

import torch

import verl
import verl.utils.torch_functional as verl_F


def compute_rloo_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples, config):
    """
    使用 RLOO（Leave-One-Out）方法计算优势函数和返回值。

    RLOO 的核心思想：对于同一 prompt 生成的 n 个样本，每个样本的基线
    由其他 n-1 个样本的平均奖励构成，从而降低方差。

    参数:
        data: 包含 rm_scores（奖励模型分数）和 acc（准确率）的数据
        response_mask: 响应部分的注意力掩码
        n_samples: 每个 prompt 对应的样本数
        config: 配置对象，包含奖励系数等参数

    返回:
        advantages: 经过白化处理的优势函数
        returns: 返回值（累积奖励）
    """
    # 计算 RLOO 奖励：对不同奖励源分别应用 RLOO，再求和
    def masked_rloo(reward_tensor_original, mask_tensor):
        """带掩码的 RLOO 计算：每个样本的基线是其他 n-1 个样本的均值"""
        reward_tensor = reward_tensor_original.clone()
        reward_tensor[~mask_tensor] = 0
        for start_pos in range(0, reward_tensor.shape[0], n_samples):
            # 计算每个样本的奖励均值
            cur_rewards_mean = torch.cat(
                [
                    reward_tensor[pos : pos + 1][mask_tensor[pos : pos + 1]].mean(dim=0, keepdim=True)
                    for pos in range(start_pos, start_pos + n_samples)
                ],
                dim=0,
            )
            cur_rewards_sum = cur_rewards_mean.sum()
            # Leave-One-Out 基线：排除自身后的其他样本均值
            cur_reward_baseline = cur_rewards_sum / (n_samples - 1)
            # 应用 RLOO 变换：缩放并减去基线
            reward_tensor[start_pos : start_pos + n_samples][mask_tensor[start_pos : start_pos + n_samples]] = (
                reward_tensor[start_pos : start_pos + n_samples][mask_tensor[start_pos : start_pos + n_samples]]
                * (n_samples / (n_samples - 1))
                - cur_reward_baseline
            )

        return reward_tensor

    reward_tensors = []

    with torch.no_grad():
        # 处理奖励模型分数（RM scores）的贡献
        if "rm_scores" in data.batch.keys() and config.algorithm.reward_dpo_coef != 0.0:
            reward_tensor = data.batch["rm_scores"]
            reward_mask = response_mask.bool()

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_dpo_coef)

        # 处理准确率（accuracy/outcome reward）的贡献
        if "acc" in data.batch.keys() and config.algorithm.reward_gt_coef != 0.0:
            reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
            reward_mask = torch.zeros_like(response_mask, dtype=torch.bool)

            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            # 计算每个响应的有效长度
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(-1)

            # 将 outcome reward 放在响应的最后一个有效 token 位置
            reward_mask[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1,
            ] = True
            reward_tensor[
                torch.arange(0, valid_response_length.shape[0], dtype=torch.long, device=valid_response_length.device),
                valid_response_length - 1,
            ] = data.batch["acc"]

            reward_tensors.append(masked_rloo(reward_tensor, reward_mask) * config.algorithm.reward_gt_coef)

        # 合并所有奖励源的贡献
        final_reward_tensor = sum(reward_tensors)

        # 计算返回值：从后向前累积奖励
        returns = (final_reward_tensor * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # 优势函数等于返回值（无 Critic 时的简化形式）
        advantages = returns.clone()
        # 对优势函数进行掩码白化处理（均值为0、方差为1）
        advantages = verl_F.masked_whiten(advantages, response_mask)

        return advantages, returns


def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
    """
    交叉熵（CE）风格的 DPO 奖励模型损失。

    将奖励模型的分数经过 sigmoid 后，与真实的准确率标签做二元交叉熵损失。

    参数:
        token_level_scores: token 级别的奖励分数
        acc: 每个样本的准确率标签（0 或 1）
        response_mask: 响应部分的掩码
        beta: 温度系数，控制分数的缩放

    返回:
        cur_dpo_loss: 标量损失值
    """
    cur_scores = ((token_level_scores * response_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc)
    return cur_dpo_loss


def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"):
    """
    分离式（Detached）DPO 奖励模型损失。

    利用 Best-of-N (BoN) 采样中的对比信息：对于每个样本，选择准确率
    更高/更低的其他样本作为正/负例，构造 DPO 损失。

    参数:
        token_level_scores: token 级别的奖励分数
        acc: 每个样本的准确率标签
        Q_bc: BoN 采样中所有样本的 Q 值矩阵
        acc_bc: BoN 采样中所有样本的准确率矩阵
        response_mask: 响应部分的掩码
        beta: 温度系数
        bon_mode: BoN 模式（"none"/"bon_rm"/"bon_acc"）

    返回:
        dpo_loss: 标量损失值
    """
    # 假设 BoN 大小等于 n_samples
    cur_Q = (token_level_scores * response_mask).sum(dim=1) * beta
    other_Q = torch.zeros_like(cur_Q)
    for i in range(token_level_scores.shape[0]):
        # 根据当前样本的准确率选择对比样本
        Q_chosen = Q_bc[i][acc_bc[i] < acc[i]] if acc[i] > 0 else Q_bc[i][acc_bc[i] > acc[i]]
        if len(Q_chosen) > 0:
            other_Q[i] = Q_chosen.mean() * beta
        else:
            other_Q[i] = 0
    # 计算 DPO 损失：正负样本之间的排序损失
    dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * ((acc > 0).float() * 2 - 1)))
    if bon_mode == "none":
        # 普通模式：直接取均值
        dpo_loss = dpo_loss.mean()
    else:
        # BoN 模式：使用顺序统计量加权
        weight = torch.zeros_like(dpo_loss)
        n_samples = acc_bc.shape[1]
        if bon_mode == "bon_rm":
            # 按奖励模型分数计算 BoN 权重
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((Q_bc[i] * beta <= cur_Q[i]).float().mean(), n_samples - 1)
        elif bon_mode == "bon_acc":
            # 按准确率计算 BoN 权重
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((acc_bc[i] <= acc[i]).float().mean(), n_samples - 1)
        else:
            raise NotImplementedError
        dpo_loss = (dpo_loss * weight).sum()

    return dpo_loss


def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):
    """
    计算 DPO 排序准确率：衡量奖励模型分数排序与真实准确率排序的一致性。

    对于同一 prompt 的 n 个样本，比较所有样本对 (i, j)：
    如果 acc[i] > acc[j]，则期望 score[i] > score[j]。

    参数:
        token_level_scores: token 级别的奖励分数
        acc: 准确率标签
        response_mask: 响应掩码
        n_samples: 每个 prompt 的样本数

    返回:
        标量，DPO 排序准确率（加权）
    """
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        # 计算每个样本的总分数
        cur_scores = (
            token_level_scores[start_id : start_id + n_samples] * response_mask[start_id : start_id + n_samples]
        ).sum(dim=1)

        def get_upper_triangle(tensor_x):
            """获取上三角部分的成对差值"""
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id : start_id + n_samples])  # 范围 [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # 实数范围
        cur_score_prediction = (cur_score_diff > 0).float()  # 范围 [0,1]
        if cur_acc_diff.abs().sum() == 0:
            # 如果所有样本准确率相同，准确率为 0.5（随机猜测）
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            # 加权准确率：按准确率差异的绝对值加权
            cur_acc = (
                ((cur_score_diff > 0) == (cur_acc_diff > 0)).float() * cur_acc_diff.abs()
            ).sum() / cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()


def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):
    """
    计算绝对 DPO 准确率：奖励分数的符号是否与准确率标签的符号一致。

    参数:
        token_level_scores: token 级别的奖励分数
        acc: 准确率标签（0 或 1）
        response_mask: 响应掩码
        n_samples: 每个 prompt 的样本数

    返回:
        标量，绝对 DPO 准确率
    """
    return (torch.sign((token_level_scores * response_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()
