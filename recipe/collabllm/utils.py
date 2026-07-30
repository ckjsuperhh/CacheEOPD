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
CollabLLM 工具函数模块。

提供 CollabLLM recipe 中使用的各种工具函数，包括：
- parse_messages：将消息列表格式化为可读对话文本
- strip_system_prompt：去除消息中的系统提示
- extract_json：自定义 JSON 解析器（从 LLM 输出中提取 JSON）
- remove_think_block：去除消息内容中的 <think> 块
- is_valid_messages：验证消息格式的有效性（检查 <think> 标签配对等）
"""

import logging
import os
import re

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def parse_messages(messages, strip_sys_prompt=True):
    """
    将消息列表格式化为可读的对话文本。

    Args:
        messages: List[dict]
            消息对象列表，每个对象具有 'role' 和 'content' 属性。
            示例: messages = [{'role': 'user', 'content': 'Hello!'},
                             {'role': 'assistant', 'content': 'Hi!'}, ...]
        strip_sys_prompt: 是否去除系统提示消息，默认为 True。

    Returns:
        str: 格式化后的对话文本，格式为 "**角色**: 内容"，每行一条消息。
    """
    if messages is None:
        return ""

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    chat = "\n".join(f"**{m.role.capitalize()}**: {m.content}" for m in messages)

    return chat


def strip_system_prompt(messages):
    """
    去除消息列表中的系统提示（system prompt）消息。

    Args:
        messages: 消息对象列表。

    Returns:
        过滤后的消息列表，不包含 role="system" 的消息。
    """
    return [msg for msg in messages if msg.role != "system"]


def extract_json(s):
    """
    自定义 JSON 解析器，从字符串中提取 JSON 对象。

    支持单引号、三引号字符串和自动类型转换，主要用于解析 LLM 输出。

    Args:
        s: 包含 JSON 内容的字符串。

    Returns:
        解析后的 Python 对象。
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


def remove_think_block(msg: dict):
    """
    从消息内容中移除 <think>...</think> 块。

    Args:
        msg: 消息字典，包含 'content' 字段。

    Returns:
        dict: 处理后的消息字典，<think> 块已被移除。
    """
    if "content" in msg and isinstance(msg["content"], str):
        msg["content"] = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL).strip()
    return msg


def is_valid_messages(msg: dict) -> bool:
    """
    检查消息是否有效，验证 <think> 标签的正确性。

    检查规则包括：
    1. <think> 标签必须与 </think> 标签配对
    2. <think> 标签内外不能为空
    3. 不允许嵌套 <think> 标签，最多只能有一个 <think> 块
    4. 去除末尾的 <|im_end|> 后内容不能为空

    Args:
        msg: 消息字典，包含 'content' 字段。

    Returns:
        bool: 消息是否有效。
    """
    content = msg.get("content")
    if not isinstance(content, str):
        return True

    # Base case: empty or whitespace-only content is invalid.
    if not content.strip():
        return False

    num_think_open = content.count("<think>")
    num_think_close = content.count("</think>")

    # Rule 1: Check for paired tags.
    if num_think_open != num_think_close:
        return False

    # Rule 3: Allow at most one think block.
    if num_think_open > 1:
        return False

    # Case 1: No <think> blocks.
    if num_think_open == 0:
        visible_content = content
    # Case 2: Exactly one <think> block.
    else:
        # Rule 2: Check for empty content inside the think block.
        match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not match or not match.group(1).strip():
            return False

        # The "visible" content is what's outside the think block.
        visible_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    visible_content = visible_content.strip()

    # Rule 4 & 2 (outside): Check if visible content is empty after handling <|im_end|>.
    if visible_content.endswith("<|im_end|>"):
        visible_content = visible_content[: -len("<|im_end|>")]

    if not visible_content.strip():
        return False

    return True
