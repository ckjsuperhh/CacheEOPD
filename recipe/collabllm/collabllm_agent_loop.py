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
CollabLLM Agent Loop 模块。

该模块定义了 CollabLLM 的智能体循环（Agent Loop），继承自 verl 框架的 ToolAgentLoop。
它负责在多轮对话中管理策略模型（policy model）的推理、工具调用和交互过程，
是 CollabLLM recipe 中 rollout 阶段的核心组件。
"""

import logging
import os
from copy import deepcopy
from typing import Any
from uuid import uuid4

from recipe.collabllm.utils import is_valid_messages
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.schemas import Message

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CollabLLMAgentLoop(ToolAgentLoop):
    """
    CollabLLM 智能体循环类。

    继承自 verl 框架的 ToolAgentLoop，负责在多轮对话场景中协调以下流程：
    1. 策略模型生成初始回复
    2. 通过用户模拟器（interaction）生成后续多轮对话
    3. 收集 rollout 数据用于强化学习训练
    """

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        执行一次完整的 CollabLLM 智能体循环。

        主要流程：先让策略模型生成初始回复，然后通过用户模拟器进行多轮交互采样，
        最终收集所有对话数据用于后续奖励计算和策略更新。

        Args:
            sampling_params: 采样参数字典，控制模型生成的温度、top_p 等参数。
            **kwargs: 额外参数，包括原始提示、多模态数据、工具参数等。

        Returns:
            AgentLoopOutput: 包含提示 ID、回复 ID、回复掩码、多模态数据、
            对数概率、轮次数、指标和额外字段（如消息列表）的输出对象。
        """
        messages = list(kwargs["raw_prompt"])
        image_data = deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # 初始化交互模块（如果配置了的话）
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            # 启动交互实例，注册该请求的交互会话
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # 创建 AgentData 实例，封装所有状态信息
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        # 对于 CollabLLM，首先让策略模型生成初始回复（PENDING -> GENERATING）
        await self._handle_pending_state(agent_data, sampling_params)

        status = await self._handle_generating_state(agent_data, sampling_params)

        if status == AgentState.TERMINATED:
            # 如果模型提前终止，告知奖励管理器打 -1 分并跳过后续交互
            # 避免不完整消息导致的奖励欺骗（reward hacking）
            num_repeats = 0
        else:
            # 收集多轮交互的 rollout 数据，num_repeat_rollouts 控制重复采样次数
            num_repeats = self.config.actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts

        # 为每次重复采样创建独立的数据副本
        interaction_requests = [deepcopy(agent_data) for _ in range(num_repeats)]

        # messages 仅在 CollabLLM 奖励管理器中使用
        messages_lst = []
        for _agent_data in interaction_requests:
            if not is_valid_messages(_agent_data.messages[-1]):
                break

            prev_msg_len = len(_agent_data.messages)
            # 运行交互循环，让用户模拟器与策略模型进行多轮对话
            await self.run_agent_data_loop(_agent_data, sampling_params, AgentState.INTERACTING)
            messages_lst.append([Message(**msg) for msg in _agent_data.messages])

            if interaction.config.get("enable_log"):
                print(f"Assistant: ...{messages_lst[-1][prev_msg_len - 1].content[-100:]}")
                print(f"User:      {messages_lst[-1][prev_msg_len].content[:100]}...")

        # 组装最终输出：分离提示和回复的 token ID
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={
                "turn_scores": agent_data.turn_scores,
                "messages": {"messages": messages_lst},  # 与 sglang 交互模式兼容的消息格式
            },
        )
        return output

    async def run_agent_data_loop(self, agent_data: AgentData, sampling_params: dict[str, Any], state: AgentState):
        """
        运行智能体数据循环，处理智能体数据的各种状态转换。

        根据当前状态执行对应的处理逻辑，包括等待（PENDING）、生成（GENERATING）、
        工具处理（PROCESSING_TOOLS）和交互（INTERACTING）等状态。

        Args:
            agent_data (AgentData): 要处理的智能体数据，包含消息、指标等状态。
            sampling_params (dict[str, Any]): 采样参数。
            state (AgentState): 智能体的初始状态，默认为 None。
        """

        # 状态机循环：根据当前状态分派到对应的处理函数
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
