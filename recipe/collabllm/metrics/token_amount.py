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
Token 数量（Token Amount）指标模块。

该模块计算多轮对话中策略模型生成的总 token 数（通过单词数近似估计）。
该指标用作奖励函数中的惩罚项，鼓励模型用更少的交互轮次和更短的回复完成任务，
提升交互效率。

在 CollabLLM 奖励函数中通常以负权重使用（如 -0.0001），
即对话越长，惩罚越大。
"""


def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    """
    计算多轮对话中未来对话部分的 token 数量。

    通过简单的单词数估计来近似计算 token 数量，
    仅统计原始提示之后的多轮交互部分的消息。

    Args:
        data_source: 数据来源标识。
        messages: 对话消息列表（Message 对象列表）。
        ground_truth: 标准答案（该指标中不使用）。
        extra_info: 额外信息字典，包含原始 prompt 用于确定未来对话的起始位置。
        **kwargs: 额外参数（该指标中不使用）。

    Returns:
        int: 未来对话部分的估计 token（单词）总数。
    """
    prompt = extra_info["prompt"]

    # 获取原始提示之后的多轮对话部分（即策略模型与用户模拟器的后续交互）
    future_conv = messages[len(prompt) :]

    # 简单的 token 数量估计：对所有消息内容按空格分词后统计词数
    total_tokens = sum(len(m.content.split()) for m in future_conv)

    return total_tokens
