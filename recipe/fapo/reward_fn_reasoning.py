# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FAPO 推理奖励函数模块（单控制器模式）。

该模块实现了 FAPO（Flawed-Aware Policy Optimization）的奖励计算逻辑，
在 verl 单控制器模式下运行。

FAPO 的核心思想是检测"有缺陷的正面样本"（Flawed Positive）：
即模型给出了正确答案但推理过程存在错误的情况。
通过 GenRM（生成式奖励模型）检测此类情况，并给予惩罚，
从而训练出推理过程更可靠的模型。

核心函数：
- verify：验证模型答案是否正确
- compute_score_baseline：基线奖励计算（仅检查答案正确性）
- compute_score_fapo：FAPO 奖励计算（额外通过 GenRM 检测有缺陷的正面样本）
"""

import json
import logging
import os

import aiohttp
from transformers import PreTrainedTokenizer

from verl.utils.ray_utils import get_event_loop
from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def verify(
    solution_str: str,
    gt: str,
) -> tuple[bool, str]:
    """
    验证数学解题答案是否正确。

    从解题字符串末尾（最后 300 字符）提取 \\boxed{} 中的答案，
    并与标准答案进行标准化比较。

    Args:
        solution_str: 模型的解题字符串。
        gt: 标准答案。

    Returns:
        tuple: (是否正确, 提取的预测答案)
    """
    solution_str = solution_str[-300:]  # 仅检查末尾部分以提高效率
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is not None:
        extracted_answer = remove_boxed(boxed_answer)
    else:
        extracted_answer = "[INVALID]"

    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


async def compute_score_baseline(
    solution_str: str,
    ground_truth: str,
    **kwargs,
):
    """
    基线奖励计算（仅检查答案正确性）。

    正确答案获得 +1.0 奖励，错误答案获得 -1.0 惩罚。

    Args:
        solution_str: 模型的解题字符串。
        ground_truth: 标准答案。

    Returns:
        dict: 包含 score、acc、pred。
    """
    loop = get_event_loop()
    """Compute the reward score for Baseline."""
    correct, pred = await loop.run_in_executor(None, lambda: verify(solution_str, ground_truth))
    reward_score = 1.0 if correct else -1.0
    return {"score": reward_score, "acc": correct, "pred": pred}


# FAPO Hyper-parameters
FAPO_GENRM_TEMPLATE = (
    "The following is a math problem with its ground truth answer, along with an AI solution (split into steps):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Ground Truth]\n\n"
    "{ground_truth}\n\n"
    "[AI Solution]\n\n"
    "{solution}\n\n"
    "Your task is to review and critique the solution step by step. "
    "Once you identify an error in a step, return the index of the step where the earliest error occurs. "
    "Otherwise, return the index of -1 (which typically denotes 'not found').\n\n"
    "Please reason step by step, put your final answer (i.e., the index) in \\boxed{{}}."
)
GRM_SAMPLING_PARAMS = {
    "max_new_tokens": 16384,
}
FLAWED_REWARD_PENALTY = 1.0


async def generate_aiohttp(router_address: str, prompt_ids: list[int], sampling_params: dict):
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
    }
    url = f"http://{router_address}/generate"
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with session.post(url, json=payload) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                return output
            except Exception:
                logger.error(f"Failed to parse JSON response: {output}")
                return {}
    finally:
        await session.close()


async def compute_score_fapo(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    """Compute the reward score for FAPO.

    计算 FAPO 奖励分数（单控制器模式）。

    核心流程：
    1. 首先验证模型答案是否正确（baseline 奖励）
    2. 对于答案正确且属于训练集的样本，调用 GenRM 检测是否存在"有缺陷的正面样本"
    3. 如果 GenRM 检测到错误步骤（flawed positive），给予额外惩罚

    Args:
        data_source: 数据来源标识。
        solution_str: 模型的解题字符串。
        ground_truth: 标准答案。
        extra_info: 额外信息，包含 question 和 split。
        reward_router_address: GenRM 服务路由地址。
        reward_model_tokenizer: GenRM 的分词器。

    Returns:
        dict: 包含 score、acc、pred、is_flawed_positive。
    """
    loop = get_event_loop()

    question, split = extra_info["question"], extra_info["split"]
    correct, pred = await loop.run_in_executor(None, lambda: verify(solution_str, ground_truth))
    reward_score = 1.0 if correct else -1.0
    is_flawed_positive = False

    # 对于测试集或答案错误的样本，直接返回基线奖励
    if split == "test" or not correct:
        return {"score": reward_score, "acc": correct, "pred": pred, "is_flawed_positive": is_flawed_positive}

    # 构建 GenRM 提示：让 GenRM 判断解题步骤中首次出现错误的位置
    grm_prompt = FAPO_GENRM_TEMPLATE.format(
        problem=question,
        ground_truth=ground_truth,
        solution=solution_str,
    )
    grm_prompt_ids = await loop.run_in_executor(
        None,
        lambda: reward_model_tokenizer.apply_chat_template(
            [{"role": "user", "content": grm_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        ),
    )
    grm_outputs = await generate_aiohttp(
        router_address=reward_router_address,
        prompt_ids=grm_prompt_ids,
        sampling_params=GRM_SAMPLING_PARAMS,
    )
    grm_response_ids = grm_outputs.get("output_ids", None)
    if grm_response_ids is not None:
        grm_response = await loop.run_in_executor(
            None, lambda: reward_model_tokenizer.decode(grm_response_ids, skip_special_tokens=True)
        )
        try:
            err_location = remove_boxed(last_boxed_only_string(grm_response))
            is_flawed_positive = int(err_location) != -1
        except Exception:
            is_flawed_positive = False

        if is_flawed_positive:
            reward_score -= FLAWED_REWARD_PENALTY

    return {"score": reward_score, "acc": correct, "pred": pred, "is_flawed_positive": is_flawed_positive}
