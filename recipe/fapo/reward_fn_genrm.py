# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

"""
FAPO GenRM 奖励函数模块（用于训练 GenRM）。

该模块实现了用于训练生成式奖励模型（Generative Reward Model, GenRM）的奖励函数。
在 FAPO 框架中，GenRM 被训练来判断数学解题步骤中首次出现错误的位置。

核心函数：
- compute_score_fapo_genrm：评估 GenRM 对错误步骤定位的准确性，
  通过与标准答案对比计算奖励分数。
"""


from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed


def parse_ans(
    solution_str: str,
    total_steps: int,
) -> tuple[bool, str]:
    """
    从解题字符串中提取 GenRM 预测的错误步骤索引。

    从解题字符串的最后 300 个字符中提取 \\boxed{} 中的答案，
    并将其转换为整数索引。-1 表示无错误，其他值表示首次出错的步骤索引。

    Args:
        solution_str: 模型的解题字符串。
        total_steps: 解题过程的总步骤数。

    Returns:
        int 或 None: 预测的错误步骤索引，-1 表示无错误，None 表示解析失败。
    """
    try:
        boxed_answer = last_boxed_only_string(solution_str[-300:])  # 从末尾提取 boxed 答案
        extracted_answer = int(remove_boxed(boxed_answer))
        # 有效范围：-1（无错误）或 [0, total_steps)（错误步骤索引）
        if extracted_answer == -1 or 0 <= extracted_answer < total_steps:
            return extracted_answer
        else:
            return None
    except Exception:
        return None


def compute_score_fapo_genrm(
    solution_str: str,
    ground_truth: int,
    extra_info: dict,
    **kwargs,
) -> float:
    """
    计算 GenRM 训练时的奖励分数。

    评估 GenRM 对错误步骤定位的准确性：
    - 如果标准答案标记为 -1（即解题无错误），GenRM 也应返回 -1
    - 如果标准答案给出了具体的错误步骤，GenRM 应尽量定位到相近的步骤
    - 解析失败（无效输出）给予 -1.0 惩罚

    Args:
        solution_str: 模型的解题字符串。
        ground_truth: 标准答案（-1 表示无错误，其他值为首次出错的步骤索引）。
        extra_info: 额外信息，包含 total_steps（总步骤数）。

    Returns:
        dict: 包含 score（奖励分数）、acc（是否准确）、pred（预测值）、gt（标准答案）。
    """
    # 验证解题过程
    total_steps = extra_info["total_steps"]
    extracted_answer = parse_ans(solution_str, total_steps)
    gt = "correct" if ground_truth == -1 else "incorrect"  # 标准答案是否正确（无错误/有错误）
    pred = "correct" if extracted_answer == -1 else "incorrect"  # 预测是否正确
    if extracted_answer is None:
        pred = "[INVALID]"  # 标记为无效输出
    acc = gt == pred  # 是否匹配
    # 奖励计算逻辑
    if extracted_answer is None:
        reward = -1.0  # 无效输出惩罚
    elif ground_truth == -1:
        reward = 1.0 if extracted_answer == -1 else -1.0  # 无错误场景：正确返回 -1 得正奖励
    else:
        # ground truth != -1，有错误场景
        if extracted_answer == -1:
            reward = -1.0  # 应该检测到错误但返回了 -1
        else:
            # 都检测到了错误，根据距离计算奖励（距离越近奖励越高）
            reward = 1.0
            reward -= abs(extracted_answer - ground_truth) / total_steps

    return {
        "score": reward,
        "acc": acc,
        "pred": extracted_answer,
        "gt": ground_truth,
    }
