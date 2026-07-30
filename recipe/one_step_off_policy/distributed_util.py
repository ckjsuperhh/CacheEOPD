# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
分布式通信工具模块（One-Step-Off Policy）。

提供 vLLM StatelessProcessGroup 初始化功能，用于在 Actor 和 Rollout worker 之间
建立 NCCL 通信组，实现权重同步。支持 CUDA 和 NPU（华为昇腾）两种后端。
"""

from verl.utils.device import is_npu_available


def vllm_stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    使用 vLLM 的 StatelessProcessGroup 创建无状态进程组。

    该函数在训练进程和 vLLM worker 之间建立 NCCL 数据面通信，
    用于 Actor 到 Rollout 的权重同步。

    Args:
        master_address: 主节点地址
        master_port: 主节点端口
        rank: 当前进程的全局秩
        world_size: 进程组总大小
        device: 设备（GPU/NPU）

    Returns:
        PyNcclCommunicator: NCCL 通信器实例
    """
    # NOTE: If it is necessary to support weight synchronization with the sglang backend in the future,
    # the following can be used:
    # from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    # from sglang.srt.distributed.utils import statelessprocessgroup
    if is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
