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
CollabLLM 奖励函数模块。

该模块实现了 CollabLLM 的对话级奖励函数（conversation-level reward function）
和自定义奖励管理器（CollabLLMRewardManager）。

核心组件：
1. conversation_level_reward_func：异步对话级奖励函数，动态加载 metrics 目录下的
   评估指标文件，对多轮对话进行打分，支持多种评估维度（准确性、交互性、token 数量等）。
2. CollabLLMRewardManager：verl 框架的奖励管理器，负责将对话级奖励聚合为
   训练所需的奖励张量（reward tensor），支持多次重复 rollout 的奖励聚合。

在 EOPD/verl 框架中，该模块通过 custom_reward_function 配置项注册到训练流程中，
是 CollabLLM 强化学习训练中奖励计算的核心组件。
"""

import asyncio
import importlib.util
import os
import sys
from typing import Any, Callable, Optional

import litellm
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"


async def conversation_level_reward_func(
    data_source, messages, ground_truth, extra_info, metrics, **kwargs
) -> torch.Tensor:
    """
    Async version of conversation-level reward function.

    Apply conversation-level reward function to the future interactions between the user simulator
    and policy model, which are generated from `verl/interactions/collabllm_interation.py`

    对话级奖励函数的异步版本。

    动态加载 metrics 目录下的评估指标模块，对策略模型与用户模拟器之间的多轮对话进行打分。
    每个指标文件需实现一个 compute_score 函数，支持异步调用。包含指数退避的重试机制
    以应对 API 限流等异常情况。

    Args:
        data_source: 数据来源标识。
        messages: 对话消息列表。
        ground_truth: 标准答案。
        extra_info: 额外信息字典。
        metrics: 要计算的指标名称列表（对应 metrics/ 目录下的 .py 文件名）。
        **kwargs: 传递给各指标计算函数的额外参数（如 LLM 评判模型的配置）。

    Returns:
        dict: 键为指标名称，值为对应的 torch.Tensor 评分。
    """
    num_retries = kwargs.get("num_retries", 6)

    rewards = {}
    for metric in metrics:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metric_file_path = os.path.join(current_dir, f"metrics/{metric}.py")

        if not os.path.exists(metric_file_path):
            print(f"Error: Metric file '{metric_file_path}' not found. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue

        spec = importlib.util.spec_from_file_location(f"metric_{metric}", metric_file_path)
        if spec is None:
            print(f"Error: Could not create spec for metric '{metric}'. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue

        module = importlib.util.module_from_spec(spec)

        try:
            sys.modules[f"metric_{metric}"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error loading metric module from '{metric_file_path}': {e}. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue

        # Assume each metric file has a compute_score function
        if not hasattr(module, "compute_score"):
            print(
                f"Error: Function 'compute_score' not found in '{metric_file_path}'. Assigning 0 to metric '{metric}'."
            )
            rewards[metric] = 0.0
            continue

        compute_score_fn = module.compute_score

        # Retry mechanism for calling the metric function
        for attempt in range(num_retries):
            try:
                # Call the metric function (await if it's async)
                if asyncio.iscoroutinefunction(compute_score_fn):
                    rewards[metric] = await compute_score_fn(data_source, messages, ground_truth, extra_info, **kwargs)
                else:
                    rewards[metric] = compute_score_fn(data_source, messages, ground_truth, extra_info, **kwargs)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == num_retries - 1:  # Last attempt
                    print(
                        f"Error: Failed to compute metric '{metric}' after {num_retries} attempts. "
                        f"Last error: {e}. Assigning 0 to metric '{metric}'."
                    )
                    rewards[metric] = 0.0
                else:
                    print(f"Attempt {attempt + 1} failed for metric '{metric}': {e}. Retrying...")
                    if isinstance(e, litellm.RateLimitError):
                        await asyncio.sleep(max(2**attempt, 60))  # Exponential backoff

    # Return dict with metric names as keys
    return {metric: torch.tensor(reward, dtype=torch.float32) for metric, reward in rewards.items()}


@register("collabllm")
class CollabLLMRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/Wuyxin/collabllm/

    CollabLLM 奖励管理器，注册名称为 "collabllm"。

    该类继承自 verl 框架的 AbstractRewardManager，负责：
    1. 接收训练循环传来的 rollout 数据（DataProto）
    2. 调用 conversation_level_reward_func 对多轮对话进行多维评分
    3. 将多次重复 rollout 的评分进行加权聚合
    4. 输出最终的奖励张量（reward tensor），用于策略梯度更新

    支持的配置参数：
    - metric_weights：各评估指标的权重字典（如 accuracy=1, interactivity=1, token_amount=-0.0001）
    - llm_judge_kwargs：传递给 LLM 评判模型的参数（如 model、max_tokens、temperature）
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        metric_weights: dict,
        llm_judge_kwargs: dict,
        reward_fn_key: str = "data_source",
        compute_score: Optional[Callable] = None,
        normalize_by_data_source=False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        self.metric_weights = metric_weights
        self.llm_judge_kwargs = llm_judge_kwargs
        self.normalize_by_data_source = normalize_by_data_source

        self.metrics = list(self.metric_weights.keys())

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        计算并返回奖励张量。

        如果数据中已包含 rm_scores（奖励模型分数），则直接返回；
        否则通过异步方式调用对话级奖励函数进行计算。

        Args:
            data: 包含 rollout 数据的 DataProto 对象。
            return_dict: 是否以字典形式返回（包含 "reward_tensor" 键）。

        Returns:
            torch.Tensor 或 dict: 奖励张量或包含奖励张量的字典。
        """
        # 如果数据中已有奖励模型分数，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        # 使用线程兼容的事件循环管理异步计算（避免与 asyncio.run() 冲突）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._compute_rewards_async(data, return_dict))
        finally:
            loop.close()

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """
        异步计算奖励的内部方法。

        核心流程：
        1. 从 DataProto 中提取提示、回复和消息数据
        2. 对多次重复 rollout 的消息进行批量评分
        3. 将各指标的评分按权重加权聚合
        4. 将最终奖励放置在回复序列的最后一个有效 token 位置

        Args:
            data: 包含 rollout 数据的 DataProto 对象。
            return_dict: 是否以字典形式返回。

        Returns:
            torch.Tensor 或 dict: 奖励张量或包含奖励张量的字典。
        """
        # 批量评分：获取提示长度和有效回复长度
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        data_source = data.non_tensor_batch["data_source"]
        ground_truth = data.non_tensor_batch["ground_truth"]
        extra_info = data.non_tensor_batch["extra_info"]
        message_lst = data.non_tensor_batch["messages"]

        # 将消息按重复 rollout 次数进行分组
        num_repeat_rollouts = len(message_lst[0]["messages"])  # 每次 rollout 重复采样的对话数
        batch_size = len(data_source)

        grouped_messages = [
            [message_lst[i]["messages"][j] for i in range(len(message_lst))] for j in range(num_repeat_rollouts)
        ]

        # 将所有批次数据按 rollout 次数展平，便于并行评分
        flattened_data_sources = [data_source[i] for _ in range(num_repeat_rollouts) for i in range(batch_size)]
        flattened_ground_truths = [ground_truth[i] for _ in range(num_repeat_rollouts) for i in range(batch_size)]
        flattened_extra_infos = [extra_info[i] for _ in range(num_repeat_rollouts) for i in range(batch_size)]
        flattened_messages = [grouped_messages[j][i] for j in range(num_repeat_rollouts) for i in range(batch_size)]

        if num_repeat_rollouts > 0:
            # 并行调用奖励函数对所有展平后的对话进行评分
            tasks = [
                self.compute_score(
                    flattened_data_sources[i],
                    flattened_messages[i],
                    flattened_ground_truths[i],
                    flattened_extra_infos[i],
                    self.metrics,
                    **self.llm_judge_kwargs,
                )
                for i in range(len(flattened_data_sources))
            ]
            score_dicts = await asyncio.gather(*tasks)

            # 将每个指标在多次重复 rollout 中的评分进行聚合（按 rollout 次数重塑后求和）
            scores_by_metrics = {
                metric: torch.stack([score_dict[metric] for score_dict in score_dicts])
                .view(num_repeat_rollouts, -1)
                .sum(dim=0)
                for metric in self.metrics
            }

            # 对各指标应用对应的权重系数，并截断到 [-1.0, 1.0] 范围
            weighted_scores_by_metrics = {
                metric: torch.clamp(
                    scores_by_metrics[metric] * self.metric_weights[metric] / num_repeat_rollouts,
                    min=-1.0,
                    max=1.0,
                )
                for metric in self.metrics
            }
            # Compute mean of weighted scores for each metric
            mean_weighted_scores_by_metrics = {
                metric: weighted_scores_by_metrics[metric].mean(dim=0) for metric in self.metrics
            }

            # Combine weighted scores from all metrics into a single tensor
            scores = torch.stack([weighted_scores_by_metrics[metric] for metric in self.metrics]).sum(dim=0)
        else:
            score_dicts = []
            scores = torch.full((batch_size,), 0.0, dtype=torch.float32, device=prompt_ids.device)
            mean_weighted_scores_by_metrics = {metric: 0.0 for metric in self.metrics}

        print("Scores:", scores, mean_weighted_scores_by_metrics)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor
