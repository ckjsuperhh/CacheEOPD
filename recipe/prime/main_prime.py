# Copyright 2024 PRIME team and/or its affiliates
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

"""
PRIME 训练主入口模块。

该模块是 PRIME（Process Reward Model for Implicit Process Reward）recipe 的启动入口。
它使用 Hydra 配置框架加载 YAML 配置，通过 Ray 分布式框架启动训练任务。
主要职责：
  1. 初始化 Ray 集群
  2. 根据配置选择 Actor/Critic 后端（FSDP 或 Megatron）
  3. 构建角色到 Worker 的映射关系
  4. 创建奖励函数（Reward Manager）
  5. 实例化 RayPRIMETrainer 并启动训练
"""

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import validate_config

from .prime_ray_trainer import RayPRIMETrainer


@hydra.main(config_path="config", config_name="prime_trainer", version_base=None)
def main(config):
    """Hydra 入口函数，加载配置后调用 run_prime 启动训练。"""
    run_prime(config)


def run_prime(config, compute_score=None):
    """
    初始化 Ray 集群并提交主训练任务。

    参数:
        config: OmegaConf 配置对象，包含所有训练超参数
        compute_score: 可选的自定义评分函数，传递给 RewardManager
    """
    if not ray.is_initialized():
        # 设置 Ray 运行时环境变量，启用 tokenizer 并行和 NCCL 日志
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        # 初始化本地 Ray 集群
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # 将主训练任务提交到 Ray 集群远程执行
    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # 确保 main_task 不会被调度到 head 节点上
def main_task(config, compute_score=None):
    """
    Ray 远程执行的主训练任务。

    负责：
      1. 打印和解析配置
      2. 根据策略类型（FSDP/Megatron）选择 Worker 类
      3. 构建角色-Worker 映射和资源池
      4. 加载 tokenizer 和奖励管理器
      5. 创建并启动 RayPRIMETrainer

    参数:
        config: OmegaConf 配置对象
        compute_score: 可选的自定义评分函数
    """
    # 打印初始配置
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True 会解析符号引用
    OmegaConf.resolve(config)

    # 根据 actor 策略类型定义 Worker 类
    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        assert config.critic.strategy in {"fsdp", "fsdp2"}
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # 角色到 Worker 的映射：ActorRollout 角色使用 ActorRolloutRefWorker
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }

    # 定义全局资源池，分配所有 GPU
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }

    # 如果需要 KL 散度约束，则添加参考策略模型
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    # 如果启用奖励模型，添加 PRIME 专用的奖励模型 Worker
    if config.reward_model.enable:
        from .prime_fsdp_workers import PRIMERewardModelWorker

        role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # 验证配置的合法性
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(role_worker_mapping),
        use_critic=False,  # PRIME 不使用 Critic 模型
    )

    # 从 HDFS 下载模型检查点到本地
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # 实例化 tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)
    # 根据配置选择奖励管理器类型（naive 或 prime）
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError
    # 创建训练用奖励函数（num_examine=0 表示不做检查）
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # 创建验证用奖励函数（num_examine=1 表示进行检查）
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    # 初始化资源池管理器
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # 创建 PRIME 训练器并启动训练
    trainer = RayPRIMETrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
