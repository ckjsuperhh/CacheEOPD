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

# ---------------------------------------------------------------------------
# 模块说明：
#   本文件是 verl 框架中 "rollout"（采样 / 生成）模块的接口契约与插件入口。
#   - BaseRollout：所有推理后端（vLLM / SGLang 等）的 rollout 实现都必须继承的抽象基类。
#   - get_rollout_class：根据后端名称与运行模式，动态加载并返回对应的实现类（工厂函数）。
#   与 CacheOPD / EOPD 主题最相关的部分，是基类定义的权重与 KV cache 显存生命周期接口
#   （resume / update_weights / release），真正落地逻辑在 vllm_rollout、sglang_rollout 子类中。
# ---------------------------------------------------------------------------

import importlib
from abc import ABC, abstractmethod
from typing import Generator

import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

# 该模块对外暴露的符号
__all__ = ["BaseRollout"]


class BaseRollout(ABC):
    """Base class for rollout.

    抽象基类：统一不同推理后端的接口。所有具体 rollout 实现都继承它。
    子类需实现三个异步协程（resume / update_weights / release），用于管理
    推理引擎中的权重与 KV cache 在 GPU 显存中的装载、热更新与释放。
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        # 把 Hydra/OmegaConf 的 DictConfig 转换成普通 dataclass，便于类型检查与访问
        self.config = omega_conf_to_dataclass(config)
        # 模型结构配置（如 hidden_size、num_layers 等），明确指定为 HFModelConfig 类型
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        # 分布式拓扑信息（DeviceMesh），描述本 rollout 工作在哪些 GPU / 通信组上
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        将权重或 KV cache 重新装载（恢复）到 GPU 显存中，供推理使用。
        这与 CacheOPD 中“复用上一轮缓存”的思路直接相关。

        Args:
            tags: weights or kv_cache.  例如 ["weights"] 或 ["kv_cache"]，指定要恢复的内容。
        """
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update the weights of the rollout model.

        用训练侧更新后的权重，热更新推理引擎里的模型权重，避免每步都重新加载整个模型。
        以生成器（generator）形式逐个产出 (权重名, 张量)，支持流式、低显存峰值更新。

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        pass

    @abstractmethod
    async def release(self):
        """Release weights and kv cache in GPU memory.

        释放 GPU 显存中的权重与 KV cache，腾出空间给训练侧或其它阶段使用。
        """
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        【同步模式】批量生成序列。注意这不是抽象方法：默认直接抛 NotImplementedError，
        只有走“同步 rollout”的后端才需要重写它；异步 server 模式走上面的 resume/update_weights/release。

        Args:
            prompts: The input prompts.

        Returns:
            The output sequences.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Rollout 注册表：把 (后端名, 模式) 映射到具体的“模块路径.类名”（全限定名 FQDN）。
# 这里目前只注册了 "async"（server 模式）的两种后端。
# ---------------------------------------------------------------------------
_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
}


def get_rollout_class(rollout_name: str, mode: str = "async") -> type[BaseRollout]:
    """Get the rollout class by name.

    工厂函数：根据后端名称与运行模式，动态查找并返回对应的 Rollout 实现类
    （返回的是“类本身”，而非实例；实例化的时机由调用方决定）。

    Args:
        rollout_name: The name of the rollout.  例如 "vllm" 或 "sglang"。
        mode: The mode of the rollout, async: server mode.  默认 "async"（server 模式）。

    Returns:
        The rollout class.
    """
    # 校验 (后端名, 模式) 是否在注册表中，否则给出清晰报错
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, f"Rollout {rollout_name} with mode {mode} not found"
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    # 把 "a.b.c.ClassName" 拆成模块名 "a.b.c" 与类名 "ClassName"
    module_name, class_name = fqdn.rsplit(".", 1)
    # 动态导入模块（import 该后端实现文件）
    rollout_module = importlib.import_module(module_name)
    # 从模块里取出具体类并返回（策略/插件模式的典型写法）
    return getattr(rollout_module, class_name)
