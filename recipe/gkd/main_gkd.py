# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, furunding
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
GKD（On-Policy Knowledge Distillation，在线策略知识蒸馏）训练入口模块。

该模块是 EOPD/verl 框架中 GKD recipe 的主入口文件，负责：
1. 使用 Hydra 加载训练配置
2. 初始化 Ray 分布式集群
3. 创建并启动 TaskRunner 远程任务执行器，驱动整个知识蒸馏训练流程

注意：该文件没有与 ray_trainer.py 合并，因为 ray_trainer 可能被其他 main 入口复用。
"""

"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.gkd.ray_trainer import OnPolicyDistillTrainer

RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "false",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


@hydra.main(config_path="config", config_name="on_policy_distill_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.
    训练主入口函数，使用 Hydra 管理配置并启动在线策略蒸馏训练流程。

    Args:
        config: Hydra 配置字典，包含所有训练参数。
    """
    run_on_policy_distill(config)


# Define a function to run the PPO-like training process
def run_on_policy_distill(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.
    初始化 Ray 集群并运行分布式知识蒸馏训练流程。

    该函数负责：
    1. 初始化 Ray 运行时环境（配置环境变量、CPU 数量等）
    2. 创建 TaskRunner 远程 Actor 实例
    3. 支持 nsys 性能分析配置
    4. 等待训练任务完成并可选地导出时间线分析文件

    Args:
        config: 训练配置对象，包含 Ray 初始化设置、模型路径、训练超参数等。
    """
    # Check if Ray is not initialized

    if not ray.is_initialized():
        # 初始化 Ray 本地集群
        # 在运行时环境中设置环境变量以控制 tokenizer 并行度、
        # NCCL 调试级别、VLLM 日志级别等
        ray.init(
            runtime_env=RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    # 创建 TaskRunner 远程实例
    # 如果配置了 nsys 性能分析工具且有指定的分析步数，则附加 nsight 运行时选项
    if (
        config.global_profiler.tool == "nsys"
        and OmegaConf.select(config.global_profiler, "steps") is not None
        and len(OmegaConf.select(config.global_profiler, "steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.
    Ray 远程任务执行器，封装了分布式知识蒸馏训练的核心逻辑。

    该类作为 Ray Remote Actor 运行在独立的 CPU 节点上（避免与 head 节点竞争资源），
    负责初始化 tokenizer、数据集、worker 组、trainer，并启动完整的训练流程。

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.
        执行主训练工作流。

        该方法负责：
        1. 解析并打印配置信息
        2. 下载模型检查点到本地
        3. 实例化 tokenizer 和处理器
        4. 根据后端策略（Megatron）创建对应的 Actor/Rollout Worker
        5. 配置资源池和角色映射
        6. 创建训练/验证数据集
        7. 初始化 OnPolicyDistillTrainer 并启动训练

        Args:
            config: 训练配置对象，包含设置和运行知识蒸馏训练所需的所有参数。
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Megatron-only workers, split into rollout and actor
        if config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray import RayWorkerGroup

            from .megatron_workers import (
                MegatronOnPolicyDistillActorWorker,
                MegatronOnPolicyDistillRolloutWorker,
            )

            rollout_cls = MegatronOnPolicyDistillRolloutWorker
            actor_cls = MegatronOnPolicyDistillActorWorker
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        # Worker mapping and resource pools
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.Rollout: ray.remote(rollout_cls),
            Role.Actor: ray.remote(actor_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        actor_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes

        resource_pool_spec = {
            "rollout_pool": rollout_pool,
            "actor_pool": actor_pool,
        }
        mapping = {
            Role.Rollout: "rollout_pool",
            Role.Actor: "actor_pool",
        }
        print(f"resource_pool_spec: {resource_pool_spec}")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # Create training and validation datasets.
        train_dataset = RLHFDataset(config.data.train_files, tokenizer, config.data, None)

        if config.data.val_files:
            val_dataset = RLHFDataset(config.data.val_files, tokenizer, config.data, None)
        else:
            val_dataset = None

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = OnPolicyDistillTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    main()
