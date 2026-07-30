# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

"""
One-Step-Off Policy PPO 训练入口模块。

该模块是 EOPD/verl 框架中 one_step_off_policy recipe 的主入口文件，负责：
1. 使用 Hydra 加载训练配置
2. 创建 Actor/Rollout/Critic 等角色的 Worker 映射
3. 配置资源池管理器（Actor 和 Rollout 使用独立 GPU 资源池）
4. 启动 OneStepTaskRunner 执行完整的 PPO 训练流程

核心特点：
- Actor 和 Rollout 分离到不同的 GPU 资源池
- 支持 FSDP 和 Megatron 两种后端策略
- 使用异步 rollout 管理器实现一步超前（one-step-off）的流水线训练
"""

import asyncio
import os
import socket

import hydra
import ray

from recipe.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from recipe.one_step_off_policy.utils import need_critic
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    Create resource pool manager
    创建资源池管理器。

    根据角色列表创建两个独立的 GPU 资源池：
    - trainer_pool: 用于 Actor/Critic/RefPolicy/RewardModel（共享）
    - rollout_pool: 用于 Rollout（独立资源池）

    Args:
        config: 配置对象
        roles: 需要创建资源池的角色列表

    Returns:
        ResourcePoolManager: 资源池管理器实例
    """
    resource_pool_spec = {}
    mapping = {}

    # Actor/Critic resource pool
    if any(role in roles for role in [Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        # Map training-related roles to the same resource pool
        for role in [Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]:
            if role in roles:
                mapping[role] = "trainer_pool"

    # Rollout resource pool
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
        resource_pool_spec["rollout_pool"] = rollout_pool
        mapping[Role.Rollout] = "rollout_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config):
    """
    Create mapping from roles to worker classes
    创建角色到 Worker 类的映射。

    根据配置的后端策略（fsdp/megatron）选择对应的 Worker 实现：
    - DetachActorWorker: 分离式 Actor Worker（仅训练）
    - DetachAsyncRolloutWorker: 分离式异步 Rollout Worker（仅推理）
    - CriticWorker: Critic Worker（价值函数训练）

    Args:
        config: 配置对象

    Returns:
        dict: 角色到 Worker 类的映射
    """
    # Select worker class based on strategy
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from recipe.one_step_off_policy.fsdp_workers import (
            CriticWorker,
            DetachActorWorker,
            DetachAsyncRolloutWorker,
        )
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.critic.strategy == "megatron"
        from recipe.one_step_off_policy.megatron_workers import (
            CriticWorker,
            DetachActorWorker,
            DetachAsyncRolloutWorker,
        )
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

    role_worker_mapping = {
        Role.Actor: ray.remote(DetachActorWorker),
        Role.Rollout: ray.remote(DetachAsyncRolloutWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    if config.reward_model.enable:
        if config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(f"Unsupported reward model strategy: {config.reward_model.strategy}")

        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    # Add reference policy (if KL loss or reward is required)
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls


@ray.remote(num_cpus=10, max_concurrency=100)  # please make sure main_task is not scheduled on head
class OneStepTaskRunner:
    """One-Step-Off Policy 的 Ray 远程任务执行器。

    负责在独立 CPU 节点上执行完整的 PPO 训练流程：
    1. 解析配置并初始化 Worker 映射
    2. 下载模型检查点、实例化 tokenizer
    3. 创建数据集和奖励函数
    4. 初始化 OneStepOffRayTrainer 并启动训练
    """
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(role_worker_mapping),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, max_samples=config.data.get("val_max_samples", -1)
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = OneStepOffRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="one_step_off_ppo_trainer", version_base=None)
def main(config):
    from time import time

    from verl.trainer.main_ppo import run_ppo

    start_time = time()

    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    run_ppo(config, task_runner_class=OneStepTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
