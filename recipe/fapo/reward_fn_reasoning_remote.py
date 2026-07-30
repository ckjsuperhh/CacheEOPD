# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
FAPO 推理奖励函数模块（远程服务模式）。

该模块实现了 FAPO 的奖励计算逻辑，在外部远程 GenRM 服务模式下运行。
与 reward_fn_reasoning.py 的区别在于，该模块通过 HTTP API 调用外部的
GenRM 推理服务（而非在 verl 内部部署 GenRM）。

核心函数：
- verify：验证模型答案是否正确
- compute_score_baseline：基线奖励计算（仅检查答案正确性）
- compute_score_fapo：FAPO 奖励计算（通过远程 GenRM 服务检测有缺陷的正面样本）
- chat_completions_aiohttp：通过 aiohttp 发送异步 HTTP 请求到 GenRM 服务
- judge_fp_process：解析 GenRM 返回的错误步骤索引
"""

import json

import aiohttp

from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed


def verify(
    solution_str: str,
    gt: str,
) -> tuple[bool, str]:
    """
    验证数学解题答案是否正确。

    Args:
        solution_str: 模型的解题字符串。
        gt: 标准答案。

    Returns:
        tuple: (是否正确, 提取的预测答案)
    """
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is not None:
        extracted_answer = remove_boxed(boxed_answer)
    else:
        extracted_answer = "[INVALID]"

    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


def compute_score_baseline(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> float:
    """
    基线奖励计算（仅检查答案正确性）。

    正确答案 +1.0，错误答案 -1.0。仅检查解题末尾 300 字符。

    Returns:
        dict: 包含 score、acc、pred。
    """
    # 限制解题长度以提高效率（MATH-500 中最长的答案为 159 字符）
    solution_str = solution_str[-300:]

    # 验证答案
    correct, pred = verify(solution_str, ground_truth)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }


ADDRESS = "xx.xx.xx.xx:xxxx"
MODEL_NAME = "FAPO-4B-GenRM"
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


async def chat_completions_aiohttp(address, **chat_complete_request):
    """
    通过 aiohttp 异步发送 HTTP 请求到远程 GenRM 服务。

    使用 OpenAI 兼容的 chat/completions 接口调用远程 GenRM 推理服务。

    Args:
        address: 远程服务的地址（IP:端口格式）。
        **chat_complete_request: 传递给 chat/completions API 的请求参数。

    Returns:
        str: GenRM 的文本回复内容。
    """
    try:
        request_url = f"http://{address}/v1/chat/completions"
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=request_url,
            json=chat_complete_request,
        ) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                return output["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error: {e}. Output: {output}")
                return ""
    finally:
        await session.close()


def judge_fp_process(response, return_err_step=False):
    """
    解析 GenRM 返回的错误步骤索引，判断是否为"有缺陷的正面样本"。

    从 GenRM 回复中提取 \\boxed{} 中的步骤索引：
    - 返回 -1 表示无错误（非 flawed positive）
    - 返回其他值表示检测到错误步骤（flawed positive）

    Args:
        response: GenRM 的文本回复。
        return_err_step: 是否同时返回错误步骤索引。

    Returns:
        bool 或 tuple: 是否为 flawed positive；可选返回错误步骤索引。
    """
    try:
        boxed_result = last_boxed_only_string(response)
        result = remove_boxed(boxed_result)
        reward_score = int(eval(result)) != -1  # -1 表示无错误，其他值表示有缺陷
        if return_err_step:
            return reward_score, int(result)
        return reward_score
    except Exception:
        if return_err_step:
            return None, None
        return None


async def compute_score_fapo(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    """
    计算 FAPO 奖励分数（远程服务模式）。

    核心流程：
    1. 首先验证模型答案是否正确（baseline 奖励）
    2. 对于答案正确且属于训练集的样本，通过远程 GenRM 服务检测 flawed positive
    3. 如果检测到 flawed positive，将奖励分数设为 0（惩罚）

    Args:
        data_source: 数据来源标识。
        solution_str: 模型的解题字符串。
        ground_truth: 标准答案。
        extra_info: 额外信息，包含 question 和 split。
        keep_genrm_critics: 是否保留 GenRM 的详细评判文本。

    Returns:
        dict: 包含 score、acc、pred、flawed_positive 等字段。
    """
    question, split = extra_info["question"], extra_info["split"]
    result = compute_score_baseline(solution_str, ground_truth)
    result["flawed_positive"] = False

    # 测试集或答案错误的样本直接返回
    if split == "test" or result["acc"] == 0:
        if keep_genrm_critics:
            result["genrm_critics"] = ""
        return result
    else:
        # 构建 GenRM 提示并调用远程服务
        prompt = FAPO_GENRM_TEMPLATE.format(problem=question, ground_truth=ground_truth, solution=solution_str)
        messages = [{"role": "user", "content": prompt}]
        response = await chat_completions_aiohttp(
            ADDRESS,
            messages=messages,
            model=MODEL_NAME,
            max_tokens=16384,
        )
        if response is not None and judge_fp_process(response):  # 检测到 flawed positive
            result["score"] = 0.0  # 惩罚：将奖励设为 0
            result["flawed_positive"] = True

        if keep_genrm_critics and response is not None:
            result["genrm_critics"] = response

    return result
