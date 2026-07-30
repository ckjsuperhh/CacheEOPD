# Copyright 2025 Meituan Ltd. and/or its affiliates
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
One-Step-Off Policy 异步 Agent Loop 管理模块。

该模块实现了 OneStepOffAgentLoopManager，继承自 AgentLoopManager，
负责将输入批次分片并分发给多个 agent loop worker 进行异步序列生成。
使用 asyncio.gather 并行等待所有 worker 完成，并合并结果。
"""

import asyncio
import logging
import os

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OneStepOffAgentLoopManager(AgentLoopManager):
    """One-Step-Off Policy 的异步 Agent Loop 管理器。

    继承自 AgentLoopManager，提供异步序列生成能力：
    - generate_sequences_async: 将输入分片分发到多个 worker 并行生成，使用 asyncio.gather 等待
    - wake_up / sleep: 控制 rollout replica 的唤醒和休眠
    - clear_kv_cache: 清理 KV 缓存
    """
    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version).
        异步地将输入批次分片并分发给 agent loop workers 进行序列生成。

        使用 asyncio.gather + asyncio.to_thread 避免阻塞事件循环，
        每个 worker 处理一个分片，最终合并所有结果。

        Args:
            prompts (DataProto): 输入批次数据。

        Returns:
            DataProto: 合并后的生成结果，包含 timing 性能指标。
        """

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        # Use asyncio.gather with ray.get wrapped in asyncio.to_thread to avoid blocking
        import asyncio

        outputs = await asyncio.gather(
            *[
                asyncio.to_thread(ray.get, worker.generate_sequences.remote(chunk))
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])
