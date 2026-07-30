# Copyright 2024 CollabLLM Ltd. and/or its affiliates
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
CollabLLM 交互模块（Interaction）。

该模块实现了 CollabLLM 的用户模拟器交互（CollabLLMInteraction），
使用大语言模型（如 GPT-4o-mini）模拟用户与策略模型进行多轮对话。
在训练过程中，用户模拟器会根据当前对话历史生成逼真的用户回复，
从而让策略模型在丰富的多轮交互环境中学习协作能力。

此外还包含一个自定义的 JSON 解析器 extract_json，用于从 LLM 输出中提取结构化响应。
"""

import asyncio
import copy
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from recipe.collabllm.utils import remove_think_block
from verl.interactions.base import BaseInteraction
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"
USER_PROMPT_TEMPLATE = """You are role-playing as a human USER interacting with an AI collaborator to complete a specific task. Your goal is to generate realistic, natural responses that a user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- Complete Prompt or Reference Goal: This field may include the complete user request/query or a reference answer to user's request. Use this field to understand the user's intent, requirements, or what would count as a satisfactory outcome.
- Chat History: The ongoing conversation between you (as the user) and the AI

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
{task_desc}
<|The End of Task Description|>

<|The Start of Complete Prompt or Reference Goal (Not visible to the AI)|>
{single_turn_prompt}
<|The End of Complete Prompt or Reference Goal|>

<|The Start of Chat History|>
{chat_history}
<|The End of Chat History|>


## Guidelines:
- Stay in Character: Role-play as a human USER. You are NOT an AI. Maintain a consistent personality throughout the chat.
- Minimize Effort: IMPORTANT! As a user, avoid being too detailed in your responses. Provide vague or incomplete demands in the early stages of the conversation to minimize your effort. Let the AI ask for clarification rather than providing everything upfront.
- Knowledge Background: Reflect the user's knowledge level in the role-playing. If the user is less knowledgeable about a task, they might not notice incorrect statements. Ask questions that demonstrate your current understanding and areas of confusion.
- Occasionally Make Mistakes: Real-world users might misspell words, provide incorrect dates, give wrong information, or ask unclear questions. Simulate this behavior to reflect natural interactions.
- Mention Personal Preferences: Include preferences or constraints that might influence your requests or responses. For example, "I prefer short answers," "I need this done quickly," or "I like detailed comments in code."
- Goal-Oriented: Keep the chat focused on your intent. Avoid small talk or digressions. Redirect the chat back to the main objective if it starts to stray.

## Output Format:
You should output a JSON object with three entries:
- "current_answer" (str): Briefly summerize the AI's current solution to the task.
- "thought" (str): Output your thought process as a user deciding what to say next. Consider:
1. Have you obtained a satisfactory solution from the AI? If yes, you can terminate this chat.
2. If not, what specific part of the problem or solution are you struggling with?
3. Has the AI asked you to perform a task or answer a question? If so, how should you approach it?
4. Are you noticing any patterns or potential misunderstandings that need clarification?
5. If you're stuck, how can you phrase your question to get the most helpful response while demonstrating your current understanding?
- "response" (str): Based on your thought process, respond to the AI as the user you are role-playing. Stop immediately when the user's response is completed.

## Important Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Conversation Flow: If "Current Chat History" is empty, start the conversation from scratch with an initial request. Otherwise, continue based on the existing conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries or any provided information directly in your responses.
- Completion Signal: Use "{termination_signal}" as your response when you believe your goal has been solved or if you determine the AI cannot help further.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.

Remember to stay in character as a user throughout your response, and follow the instructions and guidelines carefully."""  # noqa: E501


class CollabLLMInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of CollabLLM.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the assistant.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.

    CollabLLM 交互类，用于在多轮对话中模拟用户行为。

    该类通过调用外部 LLM（如 GPT-4o-mini）来扮演用户角色，与策略模型进行对话。
    主要功能包括：
    - `start_interaction`：为一条轨迹启动一个交互实例，初始化对话上下文。
    - `generate_response`：根据当前对话历史，调用用户模型生成下一轮用户回复。
    - `finalize_interaction`：结束并清理交互实例。
    """

    def __init__(self, config: dict):
        """
        初始化 CollabLLM 交互实例。

        Args:
            config: 配置字典，包含用户模型名称、重试次数、日志开关等参数。
        """
        super().__init__(config)
        _config = copy.deepcopy(config)

        _config.pop("enable_log", None)

        self.name = _config.pop("name")  # 交互名称标识
        self.user_model = _config.pop("user_model")  # 用于模拟用户的 LLM 模型

        self.termination_signal = _config.pop("termination_signal", TERMINATION_SIGNAL)  # 对话终止信号
        self.num_retries = _config.pop("num_retries", 3)  # API 调用失败时的最大重试次数

        # 剩余的 config 参数作为用户模型的调用参数（如 max_tokens、temperature 等）
        self.user_model_kwargs = _config

        self._instance_dict = {}  # 存储活跃的交互实例

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        """
        启动一次交互会话。

        为指定的轨迹创建一个新的交互实例，保存标准答案（ground_truth）和交互参数。

        Args:
            instance_id: 实例 ID，如果为 None 则自动生成 UUID。
            ground_truth: 标准答案，用于后续奖励计算。
            **kwargs: 额外参数，必须包含 single_turn_prompt（单轮提示/任务目标）。

        Returns:
            str: 交互实例的唯一标识 ID。
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        self.interaction_kwargs = kwargs
        assert "single_turn_prompt" in kwargs, "single_turn_prompt is required in interaction_kwargs"
        return instance_id

    @rollout_trace_op
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        """
        根据当前对话历史生成用户模拟器回复。

        调用外部 LLM（如 GPT-4o-mini）来模拟用户回复，包含重试机制以应对 API 限流。
        如果用户模型输出中包含终止信号，则标记对话应终止。

        Args:
            instance_id: 交互实例 ID。
            messages: 当前对话消息列表，最后一条必须是 system 或 assistant 角色的消息。
            **kwargs: 额外参数。

        Returns:
            tuple: (是否应终止, 用户回复文本, 奖励值(此处为0), 额外信息字典)
        """
        assert messages[-1]["role"] in ["system", "assistant"], (
            "Last message input to the user model must be from system or assistant role"
        )

        import litellm

        # 将消息列表解析为可读的对话文本格式
        chat_history = self._parse_messages(messages, strip_sys_prompt=True)
        # 使用用户模拟器提示模板填充任务描述、单轮提示和对话历史
        prompt = USER_PROMPT_TEMPLATE.format(
            task_desc=self.interaction_kwargs.get("task_desc", "general assistance task"),
            single_turn_prompt=self.interaction_kwargs["single_turn_prompt"],
            chat_history=chat_history,
            termination_signal=self.termination_signal,
        )
        response = ""
        # 重试循环：尝试调用用户模型获取回复
        for i in range(self.num_retries):
            try:
                full_response = (
                    (
                        await litellm.acompletion(
                            model=self.user_model,
                            messages=[{"role": "user", "content": prompt}],
                            **self.user_model_kwargs,
                        )
                    )
                    .choices[0]
                    .message.content
                )
            except litellm.RateLimitError as e:
                logger.warning(f"[CollabLLMInteraction] hit RateLimitError: {e}. Retrying...")
                await asyncio.sleep(max(2**i, 60))  # 指数退避等待
                continue
            except Exception as e:
                logger.exception(f"An unexpected error occurred in CollabLLMAgentLoop: {e}")
                continue

            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)  # 从 LLM 输出中提取 JSON
            except Exception as e:
                logger.warning(f"[CollabLLMInteraction] Error extracting JSON: {e}. Retrying...")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                # 检查是否包含所需的三个字段：当前答案摘要、思考过程、回复内容
                if {"current_answer", "thought", "response"}.issubset(keys):
                    response = full_response.pop("response")
                    if isinstance(response, str):
                        break  # 成功获取有效回复，退出重试循环
                    else:
                        logger.warning(
                            f"[CollabLLMInteraction] got an invalid response {response} full_response {full_response}. \
                                Retrying..."
                        )
                        continue
                else:
                    logger.warning(f"[CollabLLMInteraction] Keys {keys} do not match expected keys. Retrying...")
                    continue

        self._instance_dict[instance_id]["response"] = response
        logger.debug(f"[CollabLLMInteraction] User: {response}")
        # 检查回复中是否包含终止信号，决定是否结束对话
        should_terminate_sequence = self.termination_signal in response
        reward = 0.0  # 交互阶段不计算奖励，奖励由奖励管理器后续计算

        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """结束并清理指定的交互实例，释放相关资源。"""
        del self._instance_dict[instance_id]

    def _parse_messages(self, messages, strip_sys_prompt=True):
        """
        将消息列表格式化为可读的对话文本。

        Args:
            messages: 消息字典列表，每个字典包含 'role' 和 'content' 字段。
            strip_sys_prompt: 是否去除系统提示消息，默认为 True。

        Returns:
            str: 格式化后的对话文本，格式为 "**角色**: 内容"。
        """
        if messages is None:
            return ""

        if strip_sys_prompt:
            messages = [msg for msg in messages if msg["role"] != "system"]

        messages = [remove_think_block(msg) for msg in messages]  # 去除 <think> 块

        chat = "\n".join(f"**{m['role'].capitalize()}**: {m['content']}" for m in messages)

        return chat


def extract_json(s):
    """
    自定义 JSON 解析器，从字符串中提取 JSON 对象。

    与标准 json.loads 相比，该解析器更加宽松，支持：
    - 单引号和三引号字符串
    - 自动类型转换（true/false/null/数字）
    - 从混合文本中定位 JSON 对象（通过查找首个 { 和末尾 }）

    主要用于解析 LLM 输出中的结构化 JSON 响应。

    Args:
        s: 包含 JSON 内容的字符串。

    Returns:
        解析后的 Python 对象（通常是字典）。
    """

    def convert_value(value):
        true_values = {"true": True, "false": False, "null": None}
        value_lower = value.lower()
        if value_lower in true_values:
            return true_values[value_lower]
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # Return as string if not a number

    def parse_number(s, pos):
        start = pos
        while pos < len(s) and s[pos] in "-+0123456789.eE":
            pos += 1
        num_str = s[start:pos]
        try:
            if "." in num_str or "e" in num_str.lower():
                return float(num_str), pos
            else:
                return int(num_str), pos
        except ValueError:
            logger.error(f"Invalid number at position {start}: {num_str}")
            raise

    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
        return pos

    def parse_string(s, pos):
        quote_char = s[pos]
        assert quote_char in ('"', "'")
        pos += 1
        result = ""
        while pos < len(s):
            c = s[pos]
            if c == "\\":
                pos += 1
                if pos >= len(s):
                    raise ValueError("Invalid escape sequence")
                c = s[pos]
                escape_sequences = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", quote_char: quote_char}
                result += escape_sequences.get(c, c)
            elif c == quote_char:
                pos += 1
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += c
            pos += 1
        raise ValueError("Unterminated string")

    def parse_key(s, pos):
        pos = skip_whitespace(s, pos)
        if s[pos] in ('"', "'"):
            key, pos = parse_string(s, pos)
            return key, pos
        else:
            raise ValueError(f"Expected string for key at position {pos}")

    def parse_object(s, pos):
        obj = {}
        assert s[pos] == "{"
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "}":
            pos = skip_whitespace(s, pos)
            key, pos = parse_key(s, pos)
            pos = skip_whitespace(s, pos)
            if pos >= len(s) or s[pos] != ":":
                raise ValueError(f'Expected ":" at position {pos}')
            pos += 1
            pos = skip_whitespace(s, pos)
            value, pos = parse_value(s, pos)
            obj[key] = value
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "}":
                break
            elif pos < len(s) and s[pos] != "}":
                raise ValueError(f'Expected "," or "}}" at position {pos}')
        if pos >= len(s) or s[pos] != "}":
            raise ValueError(f'Expected "}}" at position {pos}')
        pos += 1
        return obj, pos

    def parse_array(s, pos):
        lst = []
        assert s[pos] == "["
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "]":
            value, pos = parse_value(s, pos)
            lst.append(value)
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "]":
                break
            elif pos < len(s) and s[pos] != "]":
                raise ValueError(f'Expected "," or "]" at position {pos}')
        if pos >= len(s) or s[pos] != "]":
            raise ValueError(f'Expected "]" at position {pos}')
        pos += 1
        return lst, pos

    def parse_triple_quoted_string(s, pos):
        if s[pos : pos + 3] == "'''":
            quote_str = "'''"
        elif s[pos : pos + 3] == '"""':
            quote_str = '"""'
        else:
            raise ValueError(f"Expected triple quotes at position {pos}")
        pos += 3
        result = ""
        while pos < len(s):
            if s[pos : pos + 3] == quote_str:
                pos += 3
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += s[pos]
                pos += 1
        raise ValueError("Unterminated triple-quoted string")

    def parse_value(s, pos):
        pos = skip_whitespace(s, pos)
        if pos >= len(s):
            raise ValueError("Unexpected end of input")
        if s[pos] == "{":
            return parse_object(s, pos)
        elif s[pos] == "[":
            return parse_array(s, pos)
        elif s[pos : pos + 3] in ("'''", '"""'):
            return parse_triple_quoted_string(s, pos)
        elif s[pos] in ('"', "'"):
            return parse_string(s, pos)
        elif s[pos : pos + 4].lower() == "true":
            return True, pos + 4
        elif s[pos : pos + 5].lower() == "false":
            return False, pos + 5
        elif s[pos : pos + 4].lower() == "null":
            return None, pos + 4
        elif s[pos] in "-+0123456789.":
            return parse_number(s, pos)
        else:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")

    json_start = s.index("{")
    json_end = s.rfind("}")
    s = s[json_start : json_end + 1]

    s = s.strip()
    result, pos = parse_value(s, 0)
    pos = skip_whitespace(s, pos)
    if pos != len(s):
        raise ValueError(f"Unexpected content at position {pos}")
    return result
