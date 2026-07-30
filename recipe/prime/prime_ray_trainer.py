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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.

PRIME 专用 Ray 训练器模块。

本模块扩展了 verl 框架的 RayPPOTrainer，实现了 PRIME 算法的完整训练循环：
  1. 不使用 Critic 模型（与标准 PPO 的关键区别）
  2. 使用在线奖励模型（PRM）提供过程奖励信号
  3. 支持多种奖励模型更新策略（before/after/reverse）
  4. 支持 RLOO 优势函数估计
  5. 支持基于准确率和截断长度的数据过滤与下采样

核心类 RayPRIMETrainer 继承自 RayPPOTrainer，重写了：
  - fit(): PRIME 特有的训练循环
  - compute_reward(): 奖励模型的前向/更新逻辑
  - _save_checkpoint/_load_checkpoint: 额外的奖励模型检查点管理
  - filter_and_downsample: 数据过滤和下采样策略
"""

import os
import statistics
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import _compute_response_info
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import simple_timer

from . import prime_core_algos


def compute_advantage(data: DataProto, adv_estimator, config):
    """
    计算优势函数和返回值。

    参数:
        data: 包含响应和奖励信息的数据
        adv_estimator: 优势估计算法名称（目前仅支持 "rloo"）
        config: 配置对象
    返回:
        添加了 advantages 和 returns 字段的 data
    """
    if adv_estimator == "rloo":
        # 使用 RLOO（Leave-One-Out）方法计算优势函数
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = prime_core_algos.compute_rloo_advantage_return(
            data, response_mask, config.actor_rollout_ref.rollout.n, config
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def compute_data_metrics(batch, use_critic=True):
    """
    计算数据相关的训练指标。

    包括优势函数/返回值/价值函数的统计量、响应长度和 prompt 长度的统计量。

    参数:
        batch: 训练数据批次
        use_critic: 是否使用 Critic（PRIME 中为 False）
    返回:
        包含各项指标的字典
    """
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_response_mask(data: DataProto):
    """提取响应部分的注意力掩码，用于计算响应相关的损失。"""
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_timing_metrics(batch, timing_raw):
    """
    计算计时性能指标。

    将各阶段的耗时按对应阶段的 token 数量归一化，
    得到每 token 的处理时间（毫秒）。

    参数:
        batch: 训练数据批次
        timing_raw: 各阶段的原始耗时（秒）
    返回:
        包含原始耗时和每 token 耗时的字典
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


class RayPRIMETrainer(RayPPOTrainer):
    """
    PRIME 算法的 Ray 分布式训练器。

    继承自 RayPPOTrainer，运行在驱动进程（driver process）上。
    与标准 PPO 训练器的主要区别：
      - 不使用 Critic 模型（self.use_critic = False）
      - 使用在线奖励模型（PRM）代替固定的奖励函数
      - 奖励模型在每个训练步骤中都会更新
      - 使用 RLOO 优势估计而非 GAE

    训练流程：
      生成响应 → 验证奖励 → 过滤数据 → 计算 log-prob →
      计算奖励模型分数 → 更新奖励模型 → 计算优势 → 更新策略
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        device_name="cuda",
    ):
        # assert get_torch_device().is_available(), 'cuda must be available on driver'

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=device_name,
        )

        self.use_critic = False

    def _create_dataloader(self, *args, **kwargs):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            data_files=self.config.data.train_files, tokenizer=self.tokenizer, config=self.config.data
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            seed = self.config.data.get("seed")
            if seed is not None:
                train_dataloader_generator.manual_seed(seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=int(self.config.data.train_batch_size * self.config.data.oversample_factor),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            data_files=self.config.data.val_files, tokenizer=self.tokenizer, config=self.config.data
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _save_checkpoint(self):
        """
        保存训练检查点。

        保存内容包括：
          - Actor 模型检查点
          - 奖励模型检查点（如果使用）
          - 数据加载器状态（用于恢复训练进度）
          - 最新检查点步数追踪文件
        """
        # 路径格式：given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
        )

        if self.use_rm:
            reward_local_path = os.path.join(local_global_step_folder, "reward")
            reward_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "reward")
            )
            self.rm_wg.save_checkpoint(
                reward_local_path,
                reward_remote_path,
                self.global_steps,
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        import dill

        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """
        从检查点恢复训练状态。

        支持三种恢复模式：
          - "disable": 不恢复，从头训练
          - "auto": 自动查找最新的检查点恢复
          - "resume_path": 从指定路径恢复

        恢复内容包括 Actor 模型、奖励模型和数据加载器状态。
        """
        if self.config.trainer.resume_mode == "disable":
            return 0

        # 从 HDFS 加载（尚未实现）
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        reward_path = os.path.join(global_step_folder, "reward")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load rm
        if self.use_rm:
            self.rm_wg.load_checkpoint(reward_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def compute_reward(self, batch: DataProto, n_samples: int):
        """
        计算奖励模型分数，支持多种更新策略。

        根据配置的 update_style 决定奖励模型的行为：
          - "none": 仅执行前向推理（不更新 RM）
          - "after": 更新 RM 并直接返回奖励
          - "before": 先更新 RM，再用更新后的 RM 前向推理
          - "reverse": 先前向推理获取统计量（Q_bc, acc_bc），再更新 RM

        参数:
            batch: 训练数据批次
            n_samples: 每个 prompt 的样本数
        返回:
            reward_output: 奖励模型的输出
            reward_output_metrics: 奖励模型的训练指标
        """
        update_style = self.config.reward_model.model.get("update", "none")
        reward_output_metrics = {}
        if update_style == "none":  # 仅执行前向推理
            reward_output = self.rm_wg.compute_rm_score(batch)
        elif update_style == "after":  # 更新 RM 并直接返回奖励
            reward_output = self.rm_wg.update_rm(batch)
        elif update_style == "before":  # 先更新 RM，再用更新后的 RM 前向推理
            reward_output = self.rm_wg.update_rm(batch)
            if "metrics" in reward_output.meta_info.keys():
                reward_output_metrics = reduce_metrics(reward_output.meta_info["metrics"])

            reward_output = self.rm_wg.compute_rm_score(batch)
        elif update_style == "reverse":  # run forward to calculate statistics, then update reward model
            reward_output = self.rm_wg.compute_rm_score(batch)

            # broadcast q and acc tensor to each result
            bc_td = DataProto.from_dict(
                tensors={
                    "Q_bc": reward_output.batch["q"]
                    .sum(dim=-1)
                    .view(-1, n_samples)
                    .unsqueeze(1)
                    .expand(-1, n_samples, -1)
                    .reshape(-1, n_samples),
                    "acc_bc": batch.batch["acc"]
                    .view(-1, n_samples)
                    .unsqueeze(1)
                    .expand(-1, n_samples, -1)
                    .reshape(-1, n_samples),
                }
            )
            batch = batch.union(bc_td)
            reward_output = self.rm_wg.update_rm(batch)
        else:
            raise NotImplementedError

        return reward_output, reward_output_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to
        construct the PPO dataflow. The light-weight advantage computation is done on the driver process.

        PRIME 训练循环。驱动进程通过 RPC 调用 Worker 组的计算函数来构建完整的数据流。
        轻量级的优势函数计算在驱动进程上执行。

        训练流程：
          1. 加载检查点并执行初始验证
          2. 对每个 epoch 和每个 batch：
             a. 生成响应（可能含 ReMax 基线）
             b. 验证奖励并过滤/下采样数据
             c. 计算旧 log-prob 和参考 log-prob
             d. 计算奖励模型分数并更新奖励模型
             e. 计算优势函数（RLOO）
             f. 更新 Actor 策略
             g. 定期验证和保存检查点
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                with simple_timer("step", timing_raw):
                    # generate a batch
                    with simple_timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == "remax":
                        with simple_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            rm_scores, _ = self.compute_reward(batch, 1)
                            reward_baseline_tensor = rm_scores.batch.get(
                                "rm_scores", rm_scores.batch.get("acc_bc", None)
                            )
                            if reward_baseline_tensor is None:
                                raise ValueError(
                                    "Neither 'rm_scores' nor 'acc_bc' found in reward model output for baseline."
                                )
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # verify
                    with simple_timer("verify", timing_raw):
                        scores = self.reward_fn.verify(batch)
                        metrics["acc"] = statistics.mean(scores)

                    # filter the batch. 1/oversample_factor samples will be kept.
                    # If there is a filter, prompts passing it will be prioritized.

                    batch = self.filter_and_downsample(scores, batch)
                    batch.meta_info["n"] = self.config.actor_rollout_ref.rollout.n
                    n_samples = self.config.actor_rollout_ref.rollout.n

                    # recompute old_log_probs
                    with simple_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = compute_response_mask(batch)
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with simple_timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with simple_timer("adv", timing_raw):
                        if self.use_rm:
                            reward_output, reward_output_metrics = self.compute_reward(batch, n_samples)
                            batch = batch.union(reward_output)
                            if "metrics" in reward_output.meta_info.keys():
                                reward_output_metrics.update(reduce_metrics(reward_output.meta_info["metrics"]))
                            metrics.update(reward_output_metrics)

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch, adv_estimator=self.config.algorithm.adv_estimator, config=self.config
                        )

                    # update actor
                    with simple_timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with simple_timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with simple_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    if (
                        self.config.trainer.save_freq > 0
                        and (self.global_steps - 1) % self.config.trainer.save_freq != 0
                    ):
                        with simple_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                    return

    def filter_and_downsample(self, scores, batch: DataProto):
        """
        downsample the batch according to oversample_factor
        samples passing the filters will be prioritized

        根据 oversample_factor 对批次进行下采样。
        通过过滤器的样本会被优先保留。

        过滤策略：
          - 准确率过滤：过滤掉平均准确率过高或过低的 prompt（太难或太简单）
          - 截断过滤：过滤掉响应被截断（达到最大长度）的 prompt

        参数:
            scores: 每个样本的奖励分数
            batch: 数据批次
        返回:
            下采样后的数据批次（原地操作）
        """
        n_samples = int(self.config.actor_rollout_ref.rollout.n)
        reward_matrix = torch.tensor(scores).reshape(-1, n_samples)

        filter_mask = torch.ones((reward_matrix.shape[0]), dtype=torch.bool)

        # 准确率过滤：保留中等难度的样本
        if self.config.data.filter_accuracy:
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            filter_mask[
                (acc_tensor > self.config.data.accuracy_upper_bound)
                | (acc_tensor < self.config.data.accuracy_lower_bound)
            ] = False

        # 截断过滤：过滤掉达到最大长度的样本
        if self.config.data.filter_truncate:
            length_matrix = (
                batch.batch["attention_mask"][:, -batch.batch["responses"].shape[-1] :]
                .sum(dim=-1)
                .reshape(-1, n_samples)
            )
            length_tensor = torch.max(length_matrix, dim=-1)[0]
            filter_mask[length_tensor >= self.config.data.max_response_length - 1] = False

        reorder_index = torch.argsort(filter_mask, descending=True)
        reorder_index = (reorder_index.unsqueeze(-1) * n_samples + torch.arange(0, n_samples).unsqueeze(0)).view(-1)
        batch.reorder(
            reorder_index[: int(len(batch) // self.config.data.oversample_factor)]
        )  # this operation is inplace

        return batch
