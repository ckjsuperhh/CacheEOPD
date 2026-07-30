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
Implement a multiprocess PPOCritic.
PRIME 奖励模型的数据并行实现模块。

本模块实现了 DataParallelPRIMERewardModel 类，负责：
  1. 使用因果语言模型（CausalLM）作为过程奖励模型（PRM），计算 token 级别的奖励
  2. 通过对比当前模型与参考模型的 log-prob 差异来估计 Q 值
  3. 支持 remove_padding 和 Ulysses 序列并行优化
  4. 支持 token 级别或 whole-response 级别的奖励分配
  5. 使用 DPO 风格的损失函数在线更新奖励模型
"""

import itertools

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.device import get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

from .prime_core_algos import compute_ce_dpo_loss_rm, compute_detach_dpo_loss_rm

__all__ = ["DataParallelPRIMERewardModel"]


class DataParallelPRIMERewardModel:
    """
    PRIME 奖励模型的数据并行实现。

    使用因果语言模型（CausalLM）作为过程奖励模型（PRM），通过计算
    当前模型与参考模型的 log-probability 差异来产生 token 级别的奖励信号。
    支持 FSDP 数据并行、remove_padding 优化以及 Ulysses 序列并行。

    属性:
        config: 奖励模型配置
        reward_module: 可训练的奖励模型（CausalLM）
        ref_module: 冻结的参考模型（用于对比）
        reward_optimizer: 奖励模型的优化器
    """

    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        """
        初始化 PRIME 奖励模型。

        参数:
            config: 配置对象
            reward_module: 可训练的奖励模型（CausalLM）
            ref_module: 冻结的参考模型，若为 None 则使用 old_log_probs
            reward_optimizer: 优化器
        """
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        print(f"Reward model use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

    def _forward_micro_batch(self, micro_batch, prompt_length):
        """
        对单个微批次执行前向计算，生成 token 级别的奖励分数。

        流程：
          1. 使用奖励模型和参考模型分别计算 log-probability
          2. 计算两者的差值 q = log_prob_rm - log_prob_ref
          3. 根据 q 和 outcome reward 计算过程奖励 r
          4. 按配置的粒度（token/whole）分配奖励到 token 级别

        参数:
            micro_batch: 微批次数据字典
            prompt_length: prompt 的 token 长度

        返回:
            token_level_score: token 级别的奖励分数
            q: 奖励模型与参考模型的 log-prob 差值（即 Q 值估计）
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        num_actions = micro_batch["input_ids"].shape[-1] - prompt_length
        max_positions = micro_batch["attention_mask"][:, prompt_length:].sum(-1)

        if self.use_remove_padding:
            # 移除 padding token，将变长序列压缩为紧凑表示，提高计算效率
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # 移除 padding 后的 position_ids，用于对齐旋转位置编码
            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
            ).transpose(0, 1)

            # 将 input_ids 左移一位，用于计算下一个 token 的 log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # 如果使用序列并行（sp > 1），对输入进行 padding 和切分
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            output = self.reward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
                return_dict=self.use_fused_kernels,
            )

            if self.use_fused_kernels:
                rm_log_labels = output.log_probs.squeeze(0)  # (total_nnz,)
                rm_log_labels = rm_log_labels.to(torch.float32)

            else:
                rm_output_logits = output.logits.squeeze(0)
                rm_log_labels = verl_F.logprobs_from_logits(
                    logits=rm_output_logits,
                    labels=input_ids_rmpad_rolled,
                )

            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outputs_and_unpad(
                    rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )
            rm_log_labels = pad_input(
                hidden_states=rm_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            ).squeeze(-1)[:, -num_actions - 1 : -1]

        else:
            output = self.reward_module(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
                use_cache=False,
                return_dict=self.use_fused_kernels,
            )

            if self.use_fused_kernels:
                rm_log_labels = output.log_probs[:, :-1]  # (bsz, seq_length)
                rm_log_labels = rm_log_labels.to(torch.float32)

            else:
                rm_output_logits = output.logits
                rm_log_prob = torch.nn.functional.log_softmax(
                    rm_output_logits[:, :-1, :], dim=-1
                )  # (batch_size, seq_length, vocab_size)
                rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(
                    -1
                )  # (batch, seq_length)

        if self.ref_module is not None:
            # do not have to pad again
            with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output = self.ref_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                    )

                    if self.use_fused_kernels:
                        ref_log_labels = ref_output.log_probs.squeeze(0)  # (total_nnz,)
                        ref_log_labels = ref_log_labels.to(torch.float32)

                    else:
                        ref_output_logits = ref_output.logits.squeeze(0)
                        ref_log_labels = verl_F.logprobs_from_logits(
                            logits=ref_output_logits, labels=input_ids_rmpad_rolled
                        )

                    ref_log_labels = gather_outputs_and_unpad(
                        ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    ref_log_labels = pad_input(
                        hidden_states=ref_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                    ).squeeze(-1)[:, -num_actions - 1 : -1]
                else:
                    ref_output = self.ref_module(
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch["attention_mask"],
                        position_ids=micro_batch["position_ids"],
                        use_cache=False,
                    )

                    if self.use_fused_kernels:
                        ref_log_labels = ref_output.log_probs[:, :-1]  # (batch_size, seq_length)
                        ref_log_labels = ref_log_labels.to(torch.float32)

                    else:
                        ref_output_logits = ref_output.logits
                        ref_log_prob = torch.nn.functional.log_softmax(
                            ref_output_logits[:, :-1, :], dim=-1
                        )  # (batch_size, seq_length, vocab_size)
                        ref_log_labels = ref_log_prob.gather(
                            dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)
                        ).squeeze(-1)  # (batch, seq_length)

        else:
            ref_log_labels = micro_batch["old_log_probs"]

        ref_log_labels.to(rm_log_labels.dtype)
        # 计算 Q 值：奖励模型与参考模型的 log-prob 差值（即隐式过程奖励）
        q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]

        # 裁剪掉超出有效响应长度的 log-prob 值
        for i in range(micro_batch["input_ids"].shape[0]):
            q[i, max_positions[i] :] = 0

        # 奖励计算不需要梯度，只有 q 需要梯度（用于反向传播更新 RM）
        with torch.no_grad():
            # r 是策略模型的过程奖励信号，或奖励模型的优势函数
            lam = self.config.get("lambda", 0.0)  # GAE lambda 参数
            beta = self.config.model.get("beta_train", 0.05)  # 温度系数
            if lam == 0.0:
                # 不使用 GAE 时，过程奖励直接等于 Q * beta
                r = q * beta
            else:
                # 使用 GAE（广义优势估计）计算过程奖励
                acc = micro_batch["acc"]
                q_ = q * beta
                r = torch.zeros_like(q)
                lastgaelam = 0
                # 如果使用 GT（ground truth），调整最后一个 token 使得总和等于准确率
                # outcome reward to calculate V
                for i in range(q.shape[0]):
                    if self.config.prime_use_gt:
                        q_[i, max_positions[i] - 1] = acc[i] - q_[i, : max_positions[i] - 1].sum()
                    q_[i, max_positions[i] :] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r[:, t] = lastgaelam

            token_level_score = torch.zeros_like(q)

            # 按配置的粒度分配过程奖励到 token 级别
            if self.config.prime_granularity == "token":
                # token 粒度：将过程奖励分配到每个非最后一个 token
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, : max_positions[i] - 1] = r[i, : max_positions[i] - 1]
            elif self.config.prime_granularity == "whole":
                # whole 粒度：将整个过程奖励集中到最后一个有效 token
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, max_positions[i] - 1] = r[i, : max_positions[i]]
            else:
                raise NotImplementedError

        return token_level_score, q

    def _optimizer_step(self):
        """执行优化器步骤，包括梯度裁剪。返回梯度范数。"""
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip
            )
        self.reward_optimizer.step()
        return grad_norm

    def prime_norm(self, token_level_scores):
        """
        PRIME 特有的奖励归一化。

        使用反向累积和的最大绝对值进行批量归一化，稳定训练过程。

        参数:
            token_level_scores: token 级别的奖励分数
        返回:
            归一化后的分数
        """
        if self.config.prime_norm == "batch_norm":
            # 计算反向累积和，然后用其最大绝对值归一化
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        """
        仅执行前向推理，计算奖励模型分数（不更新模型参数）。

        用于训练循环中获取奖励信号，或验证阶段评估奖励模型质量。

        参数:
            data: 包含 input_ids、attention_mask 等的数据
        返回:
            rm_scores: 归一化后的 token 级别奖励分数
            q: 原始 Q 值（detach）
            metrics: 记录奖励模型统计指标的字典
        """
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "acc"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)
        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        return (
            rm_scores,
            q.detach(),
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )

    def update_rm(self, data: DataProto):
        """
        使用 DPO 风格损失更新奖励模型参数。

        流程：
          1. 将数据分成 mini-batch 和 micro-batch
          2. 对每个 micro-batch 计算前向传播和 DPO 损失
          3. 累积梯度并执行优化器步骤
          4. 返回更新后的奖励分数和训练指标

        参数:
            data: 包含训练数据（可能包含 BoN 对比信息 Q_bc、acc_bc）
        返回:
            rm_scores: 更新后的归一化奖励分数
            metrics: 包含 DPO 损失、梯度范数等指标的字典
        """
        # 确保处于训练模式
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts"]

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()

            for data in micro_batches:
                data = data.to(get_device_name())
                attention_mask = data["attention_mask"]
                acc = data["acc"]

                prompt_ids = data["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_mask = attention_mask[:, prompt_length:]

                rm_score, q = self._forward_micro_batch(data, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "dpo":
                    # the implementation of dpo is actually detached, which means we have to know the average
                    # value of w/l reward before the update.
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q, acc, Q_bc=data["Q_bc"], acc_bc=data["acc_bc"], response_mask=response_mask, beta=beta
                    )
                elif self.config.model.loss_type == "bon_acc":
                    # change the original distribution of each sample to BoN distribution, then update reward model
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_acc",
                    )
                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=data["Q_bc"],
                        acc_bc=data["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_rm",
                    )
                else:
                    raise NotImplementedError

                data = {"reward_model/dpo_loss": dpo_loss.detach().item()}

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = dpo_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {"reward_model/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.prime_norm(rm_scores)

        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
