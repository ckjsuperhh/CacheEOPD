# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

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
准确性（Accuracy）评估指标模块。

该模块通过 LLM-as-Judge 方式评估策略模型在多轮对话中最终回复的准确性。
它会将被评估的对话（包含目标问题和标准答案）发送给评判 LLM，
由评判 LLM 判断模型回复是否与标准答案一致，输出二元评分（0 或 1）。

该指标是 CollabLLM 多任务奖励（Multiturn-aware Reward）的组成部分之一。
"""

from recipe.collabllm.utils import extract_json, parse_messages

ACCURACY_PROMPT = '''You are a helpful and meticulous evaluator. Your task is to \
evaluate the *accuracy* of an AI model's answer to a target question. \
You will be given the target question, the ground truth answer, and the conversation between the AI and the user.

Provided Information:

<|The Start of Target Question and Ground Truth Answer|>
Target Question: {single_turn_prompt}
Ground Truth Answer: {ground_truth}
<|The End of Target Question and Ground Truth Answer|>

<|The Start of The Conversation|>
{chat_history}
<|The End of The Conversation|>

You should determine whether the model's final response to the target question is \
factually correct and consistent with the provided ground truth.

Rating criteria (binary):
  • 1 = Correct   — the response matches the ground truth.
  • 0 = Incorrect — the response contradicts or misses the ground truth.

Output format (JSON):
{{
    "thought": "<your reasoning here>",
    "accuracy": <0 or 1>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. \
Use " or """ to wrap up the thought and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''


async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    """
    计算准确性评分。

    使用 LLM 评判策略模型在多轮对话中的最终回复是否与标准答案一致。

    Args:
        data_source: 数据来源标识。
        messages: 对话消息列表（Message 对象列表）。
        ground_truth: 标准答案。
        extra_info: 额外信息字典，包含交互参数（如 single_turn_prompt）。
        **kwargs: 传递给 LLM 调用的额外参数（如 model、max_tokens、temperature）。

    Returns:
        float: 准确性评分，1.0 表示正确，0.0 表示错误。
    """
    # 检查 litellm 是否可用，不可用时回退到 openai 库
    try:
        import litellm

        use_litellm = True
    except ImportError:
        # litellm 未安装，回退到 openai
        import openai

        use_litellm = False

    # 将消息列表格式化为对话文本
    chat_history = parse_messages(messages, strip_sys_prompt=True)
    # 构建评判提示：包含目标问题、标准答案和对话历史
    prompt = ACCURACY_PROMPT.format(
        single_turn_prompt=extra_info["interaction_kwargs"]["single_turn_prompt"],
        ground_truth=ground_truth,
        chat_history=chat_history,
    )

    if use_litellm:
        full_response = (
            (
                await litellm.acompletion(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )
    else:
        client = openai.AsyncOpenAI()  # Assumes API key is set in environment
        full_response = (
            (
                await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )

    full_response = extract_json(full_response)

    assert isinstance(full_response, dict), f"Expected a dict, got {type(full_response)}"
    assert {"accuracy", "thought"}.issubset(full_response.keys()), (
        f"Expected keys not found from {full_response.keys()}"
    )

    accuracy = full_response.pop("accuracy")
    return float(accuracy)
