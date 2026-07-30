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
代码通过率（Pass Rate）评估指标模块。

该模块用于评估策略模型在代码生成类任务中的表现。
首先使用 LLM 从多轮对话中提取最终完整版本的代码，
然后通过 BigCodeBench 的 untrusted_check 工具执行单元测试来验证代码正确性。

该指标适用于 bigcodebench-large 等代码生成数据集，
是 CollabLLM 多任务奖励（Multiturn-aware Reward）在编程任务中的评估方式。
"""

from bigcodebench.eval import untrusted_check

from recipe.collabllm.utils import extract_json, parse_messages

EXTRACT_MULTITURN_COMPLETION_PROMPT = '''You are a thorough and diligent conversation analyzer. \
Your task is to extract the final and complete version of a code function {entry_point} that was generated \
during a multiturn conversation between a user and a chat assistant. \
The extracted content should reflect the final and comprehensive response provided by the \
assistant based on the user’s request.

You will be provided with the task and the conversation:

<|The Start of The Task|>
{single_turn_prompt}
<|The End of The Task|>

<|The Start of The Conversation|>
{chat_history}
<|The End of The Conversation|>

Instructions for Extraction:

1. Identify the Most Update-to-Date Contents: Review the entire conversation to identify the most updated parts of \
the content provided by the assistant. This may include:
   - Different parts of the code snippet, function, class, or script.

2. Integrate Revisions: If the assistant made revisions, updates, or added sections throughout the conversation, \
ensure that these changes are fully integrated into the final content. The goal is to extract a single, cohesive \
output that incorporates all modifications and additions made during the conversation. For example, if the assistant \
writes a function at the beginning and changes a part, the final output should take the modification into account.

3. Focus on Completeness:
   - For code: Extract a complete and functional code snippet, including all necessary components such as imports, \
     functions, classes, and any other essential elements. The code should be runnable, but you do not need to \
     include any testing examples including the contents after `if __name__ == "__main__":`. Only the function code \
     is required. 

You should output a JSON object with two entries:
- "thought" (str): Output your thought process when extracting the final content. 
   1. How do different parts of the conversation contribute to the final output?
   2. How do you make sure you included the most updated and complete information?
   3. How do you make sure you did not include any information that is not necessary?
- "final_completion" (str): The final and complete version of the code extracted from the conversation. \
Rename main function name for the task to {entry_point} if needed. Remove any comments wrapped by """.

Note: 
1. If there are multiple lines, you should use triple quotes (""") to wrap the content. For example, \
   "final_completion": """first line. 
   second line.""" or "thought": """first line;
   second line.""". You should not use other triple quotes inside. 
2. In the "final_completion" entry, replace all double quotes (") with single quotes (') to prevent JSON formatting \
   issues. For example, you can output "final_completion": "'Hello World' is a common phrase." 

Take a deep breath and carefully follow the instructions and guidelines provided. 
'''


async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    """
    计算代码通过率评分。

    首先使用 LLM 从多轮对话中提取最终代码，然后通过单元测试验证代码正确性。

    Args:
        data_source: 数据来源标识。
        messages: 对话消息列表。
        ground_truth: 标准答案（该指标中不直接使用，改用 single_turn_metadata 中的测试用例）。
        extra_info: 额外信息字典，包含交互参数和 single_turn_metadata（入口函数名和测试用例）。
        **kwargs: 传递给 LLM 调用的额外参数。

    Returns:
        float: 1.0 表示代码通过测试，0.0 表示未通过。
    """
    # 检查 litellm 是否可用，不可用时回退到 openai 库
    try:
        import litellm

        use_litellm = True
    except ImportError:
        import openai

        use_litellm = False

    chat_history = parse_messages(messages, strip_sys_prompt=True)

    prompt = EXTRACT_MULTITURN_COMPLETION_PROMPT.format(
        chat_history=chat_history,
        single_turn_prompt=extra_info["interaction_kwargs"]["single_turn_prompt"],
        entry_point=extra_info["single_turn_metadata"]["entry_point"],
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
    assert {"final_completion", "thought"}.issubset(full_response.keys()), (
        f"Expected keys not found from {full_response.keys()}"
    )

    final_completion = full_response.pop("final_completion")
    metadata = extra_info["single_turn_metadata"]
    res = untrusted_check(
        final_completion,
        metadata["test"],
        metadata["entry_point"],
        max_as_limit=300 * 1024,
        max_data_limit=300 * 1024,
        max_stack_limit=300 * 1024,
        min_time_limit=60,
        gt_time_limit=60,
    )
    passed = res[0] == "pass"

    # info = res[1] # for printing extra info
    return float(passed)
