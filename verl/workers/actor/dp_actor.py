# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
【EOPD 核心】数据并行 PPO Actor 实现。

本文件实现 DataParallelPPOActor 类，基于 FSDP（Fully Sharded Data Parallel）的
数据并行策略来执行 Actor（策略模型）或 Reference Policy 的前向/反向传播。

核心功能：
1. _forward_micro_batch(): 微批次前向传播，计算 log_prob、entropy、top-k logits
2. compute_log_prob(): 批量计算 log 概率（无梯度，用于 rollout 阶段）
3. update_policy(): 策略更新（带梯度），实现 PPO mini-batch 训练循环

【EOPD 相关改动】：
- _forward_micro_batch() 支持 teacher_topk_indices 参数，提取学生模型在教师 top-k 位置
  的 log_prob，用于计算 Soft KD 损失
- update_policy() 检测 on_policy_distill 模式，传递 ref_entropy、ref_topk_log_probs、
  student_topk_log_probs 等额外参数给损失函数
- 支持 student_generation_mask 和 partial_solution_mask 用于部分解蒸馏

与其他模块的关系：
- 被 fsdp_workers.py 中的 AsyncActorRolloutRefWorker 实例化和调用
- 调用 core_algos.py 中的 policy_loss_fn 计算策略损失
- 依赖 verl.utils.attention_utils 进行 padding/unpadding 操作
"""

import logging
import os
import sys
import json

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        tokenizer=None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        self.debug_file_path = f"actor_debug_rank_{torch.distributed.get_rank()}.jsonl"
        self.debug_file = open(self.debug_file_path, "w")
        print(f"DEBUG: Logging to {self.debug_file_path}")

    def __del__(self):
        """Cleanup: close debug file if it exists"""
        if hasattr(self, 'debug_file') and self.debug_file is not None:
            try:
                self.debug_file.close()
            except:
                pass

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, topk=None, teacher_topk_indices=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple | None]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            full_log_probs: # (bs, seqlen)
            topk_info: # ((bs, seqlen, topk), (bs, seqlen, topk)) or None - (student_topk_log_probs, teacher_topk_indices)
                If teacher_topk_indices is provided, student log probs are extracted at those indices.
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    topk_log_probs_rmpad = None
                    topk_indices_rmpad = None

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # Extract top-k using teacher indices if provided, otherwise use student's own top-k
                    if teacher_topk_indices is not None:
                        # teacher_topk_indices: (bsz, seqlen, topk) - need to unpad to (total_nnz, topk)
                        # First pad back to get full sequence indices, then extract at teacher indices
                        # Actually, we need to work with unpadded indices
                        # For now, let's pad back first, then extract
                        # But we need to handle the unpadded case...
                        # Let's extract at teacher indices after padding back
                        topk_log_probs_rmpad = None
                        topk_indices_rmpad = None
                    elif topk is not None:
                        # Extract student's own top-k
                        topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, k=topk, dim=-1)  # (total_nnz, topk)
                        import torch.nn.functional as F
                        topk_log_probs_rmpad = F.log_softmax(topk_logits_rmpad, dim=-1)  # (total_nnz, topk)
                    else:
                        topk_log_probs_rmpad = None
                        topk_indices_rmpad = None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or topk is not None:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if topk_log_probs_rmpad is not None:
                        topk_log_probs_rmpad = gather_outputs_and_unpad(
                            topk_log_probs_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        topk_indices_rmpad = gather_outputs_and_unpad(
                            topk_indices_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if topk_log_probs_rmpad is not None:
                        topk_log_probs_rmpad = topk_log_probs_rmpad[:0]
                        topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                # pad back top-k if needed
                topk_info = None
                if teacher_topk_indices is not None:
                    # teacher_topk_indices: (bsz, seqlen, topk) - already in full sequence format
                    # Need to extract student logits at teacher indices
                    # First pad back logits to full sequence
                    full_logits = pad_input(
                        hidden_states=logits_rmpad,  # (total_nnz, vocab_size)
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )  # (bsz, seqlen, vocab_size)

                    # Extract student log probs at teacher indices
                    # teacher_topk_indices: (bsz, seqlen, topk)
                    # full_logits: (bsz, seqlen, vocab_size)
                    import torch.nn.functional as F
                    use_full_vocab = getattr(self.config.policy_loss, "soft_kd_student_full_vocab", False)
                    if use_full_vocab:
                        # Use full-vocab normalization, then gather teacher indices
                        student_full_log_probs = F.log_softmax(full_logits, dim=-1)  # (bsz, seqlen, vocab_size)
                        student_topk_log_probs = torch.gather(
                            student_full_log_probs,
                            dim=-1,
                            index=teacher_topk_indices,
                        )  # (bsz, seqlen, topk)
                    else:
                        # Re-normalize within top-k only
                        student_topk_logits = torch.gather(
                            full_logits,
                            dim=-1,
                            index=teacher_topk_indices,
                        )  # (bsz, seqlen, topk)
                        student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)  # (bsz, seqlen, topk)
                    topk_info = (student_topk_log_probs, teacher_topk_indices)
                elif topk_log_probs_rmpad is not None:
                    # topk_log_probs_rmpad: (total_nnz, topk)
                    # topk_indices_rmpad: (total_nnz, topk)
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad,  # (total_nnz, topk)
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )  # (bsz, seqlen, topk)
                    full_topk_indices = pad_input(
                        hidden_states=topk_indices_rmpad,  # (total_nnz, topk)
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )  # (bsz, seqlen, topk)
                    topk_info = (full_topk_log_probs, full_topk_indices)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    full_log_probs = output.log_probs
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    topk_info = None

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    
                    # Extract top-k using teacher indices if provided, otherwise use student's own top-k
                    if teacher_topk_indices is not None:
                        # teacher_topk_indices: (bsz, seqlen, topk)
                        # logits: (bsz, seqlen, vocab_size)
                        # Extract student log probs at teacher indices
                        import torch.nn.functional as F
                        use_full_vocab = getattr(self.config.policy_loss, "soft_kd_student_full_vocab", False)
                        if use_full_vocab:
                            # Use full-vocab normalization, then gather teacher indices
                            student_full_log_probs = F.log_softmax(logits, dim=-1)  # (bsz, seqlen, vocab_size)
                            student_topk_log_probs = torch.gather(
                                student_full_log_probs,
                                dim=-1,
                                index=teacher_topk_indices,
                            )  # (bsz, seqlen, topk)
                        else:
                            # Re-normalize within top-k only
                            student_topk_logits = torch.gather(
                                logits,
                                dim=-1,
                                index=teacher_topk_indices,
                            )  # (bsz, seqlen, topk)
                            student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)  # (bsz, seqlen, topk)
                        topk_info = (student_topk_log_probs, teacher_topk_indices)
                    elif topk is not None:
                        # Extract student's own top-k
                        topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)  # (bsz, seqlen, topk)
                        import torch.nn.functional as F
                        topk_log_probs = F.log_softmax(topk_logits, dim=-1)  # (bsz, seqlen, topk)
                        topk_info = (topk_log_probs, topk_indices)
                    else:
                        topk_info = None
                    
                    labels = input_ids.roll(-1, dims=1)
                    full_log_probs = logprobs_from_logits(logits, labels)
                    
                    log_probs = full_log_probs[:, -response_length - 1 : -1]
                    
                    if calculate_entropy:
                        logits_resp = logits[:, -response_length - 1 : -1, :]
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits_resp)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits_resp)

            return entropy, log_probs, full_log_probs.squeeze(-1), topk_info

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self, 
        data: DataProto, 
        calculate_entropy=False,
        topk=None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, tuple | None]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

            calculate_entropy (bool): Whether to calculate entropy.
            topk (int, optional): If provided, return top-k log probs and indices for full sequence.

        Returns:
            tuple: (log_probs, entropys, full_log_probs, topk_info)
                - log_probs: (bsz, response_length) - response part log probs
                - entropys: (bsz, response_length) or None
                - full_log_probs: (bsz, seqlen) or None (always returned for compatibility)
                - topk_info: tuple of (topk_log_probs, topk_indices) or None
                    - topk_log_probs: (bsz, seqlen, topk)
                    - topk_indices: (bsz, seqlen, topk)
        """
        # set to eval
        self.actor_module.eval()

        if topk is not None and self.use_fused_kernels:
            raise ValueError("topk requires use_fused_kernels=False")

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        full_log_probs_lst = []
        topk_info_lst = []
        
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, full_log_probs, topk_info = self._forward_micro_batch(
                    model_inputs, 
                    temperature=temperature, 
                    calculate_entropy=calculate_entropy,
                    topk=topk
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            full_log_probs_lst.append(full_log_probs)
            if topk_info is not None:
                topk_info_lst.append(topk_info)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        full_log_probs = torch.concat(full_log_probs_lst, dim=0)
        
        topk_info_result = None
        if topk_info_lst:
            topk_log_probs = torch.concat([x[0] for x in topk_info_lst], dim=0)
            topk_indices = torch.concat([x[1] for x in topk_info_lst], dim=0)
            topk_info_result = (topk_log_probs, topk_indices)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            full_log_probs = restore_dynamic_batch(full_log_probs, batch_idx_list)
            if topk_info_result is not None:
                topk_log_probs, topk_indices = topk_info_result
                topk_log_probs = restore_dynamic_batch(topk_log_probs, batch_idx_list)
                topk_indices = restore_dynamic_batch(topk_indices, batch_idx_list)
                topk_info_result = (topk_log_probs, topk_indices)

        return log_probs, entropys, full_log_probs, topk_info_result

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        # Include ref_log_prob if using KL loss or on-policy distillation
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        is_on_policy_distill = loss_mode == "on_policy_distill"
        if self.config.use_kl_loss or is_on_policy_distill:
            select_keys.append("ref_log_prob")
            # Also include full sequence and top-k logits if available
            if "ref_full_log_probs" in data.batch.keys():
                select_keys.append("ref_full_log_probs")
            if "ref_topk_log_probs" in data.batch.keys():
                select_keys.append("ref_topk_log_probs")
            if "ref_topk_indices" in data.batch.keys():
                select_keys.append("ref_topk_indices")
            # ref_entropy is REQUIRED for on_policy_distill
            if "ref_entropy" in data.batch.keys():
                select_keys.append("ref_entropy")
            elif is_on_policy_distill:
                raise RuntimeError(
                    f"ref_entropy not found in data.batch for on_policy_distill. "
                    f"Available keys: {sorted(list(data.batch.keys()))}"
                )
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # Include partial solution and student generation masks for on-policy distillation
        if "partial_solution_mask" in data.batch.keys():
            select_keys.append("partial_solution_mask")
        if "student_generation_mask" in data.batch.keys():
            select_keys.append("student_generation_mask")
        if "partial_solution_length" in data.batch.keys():
            select_keys.append("partial_solution_length")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch_idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Get teacher top-k indices if available for soft KD
                    teacher_topk_indices = model_inputs.get("ref_topk_indices", None)
                    topk = None
                    if teacher_topk_indices is not None:
                        # Get topk from teacher indices shape
                        topk = teacher_topk_indices.shape[-1]
                    
                    # all return: (bsz, response_length)
                    # Note: gradient is connected (no torch.no_grad()) for student model
                    entropy, log_prob, full_log_probs, student_topk_info = self._forward_micro_batch(
                        model_inputs, 
                        temperature=temperature, 
                        calculate_entropy=calculate_entropy,
                        topk=topk,
                        teacher_topk_indices=teacher_topk_indices
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        is_on_policy_distill = loss_mode == "on_policy_distill"
                        
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    
                    ref_log_prob = model_inputs.get("ref_log_prob", None)

                    # Check if using partial solution (on-policy distillation with partial solution)
                    # If student_generation_mask exists, use it instead of response_mask
                    # This means we only compute loss on student generation part (not partial solution part)
                    effective_response_mask = response_mask
                    if is_on_policy_distill and "student_generation_mask" in model_inputs:
                        student_generation_mask = model_inputs["student_generation_mask"].to(bool)
                        effective_response_mask = student_generation_mask
                        # Log metrics about mask usage
                        partial_solution_mask = model_inputs.get("partial_solution_mask", None)
                        if partial_solution_mask is not None:
                            partial_solution_mask = partial_solution_mask.to(bool)
                            micro_batch_metrics["actor/partial_solution_tokens"] = partial_solution_mask.sum().item()
                            micro_batch_metrics["actor/student_generation_tokens"] = student_generation_mask.sum().item()

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    # For on_policy_distill, ref_log_prob (teacher) is passed separately
                    policy_loss_agg_mode = loss_agg_mode
                    
                    # Prepare extra args for on_policy_distill with soft KD
                    extra_loss_args = {}
                    if loss_mode == "on_policy_distill":
                        response_length = log_prob.shape[1]
                        
                        # Debug context for loss-side logging
                        extra_loss_args["debug_epoch"] = int(epoch_idx)
                        extra_loss_args["debug_batch"] = int(batch_idx)
                        extra_loss_args["debug_micro_batch"] = int(micro_batch_idx)
                        
                        # Teacher entropy is REQUIRED for on-policy soft KD gating
                        ref_entropy = model_inputs.get("ref_entropy", None)
                        if ref_entropy is None:
                            print(f"[dp_actor.update_policy] DEBUG: model_inputs keys: {sorted(list(model_inputs.keys()))}")
                            print(f"[dp_actor.update_policy] DEBUG: data.batch keys (before select): {sorted(list(data.batch.keys()) if hasattr(data, 'batch') else [])}")
                            raise RuntimeError(
                                "Missing `ref_entropy` in model_inputs for on_policy_distill. "
                                "Teacher entropy is required for entropy-gated Soft KD. "
                                f"Available keys: {sorted(list(model_inputs.keys()))}"
                            )
                        # Debug: Log entropy stats
                        if ref_entropy.numel() > 0:
                            print(f"[dp_actor.update_policy] ref_entropy shape: {ref_entropy.shape}, "
                                  f"mean: {ref_entropy.mean().item():.4f}, "
                                  f"tokens with entropy > 1.0: {(ref_entropy > 1.0).sum().item()}/{ref_entropy.numel()}")
                        # ref_entropy is already (bsz, response_length) from compute_ref_log_prob
                        extra_loss_args["teacher_entropy"] = ref_entropy
                        
                        # Prepare top-k info
                        # ref_topk_log_probs is (bsz, seqlen, topk) -> need slicing
                        ref_topk_log_probs = model_inputs.get("ref_topk_log_probs", None)
                        if ref_topk_log_probs is None:
                            raise RuntimeError(
                                "Missing `ref_topk_log_probs` in model_inputs for on_policy_distill Soft KD. "
                                "Ensure `actor_rollout_ref.ref.topk_logits` is set and teacher ref_log_prob path preserves it."
                            )
                        extra_loss_args["teacher_topk_log_probs"] = ref_topk_log_probs[:, -response_length:]
                        
                        # student_topk_info[0] is (bsz, seqlen, topk) -> need slicing
                        if student_topk_info is None:
                            raise RuntimeError(
                                "Missing `student_topk_info` for on_policy_distill Soft KD. "
                                "Expected _forward_micro_batch(... teacher_topk_indices=ref_topk_indices) to return student top-k log probs."
                            )
                        extra_loss_args["student_topk_log_probs"] = student_topk_info[0][:, -response_length:]

                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=effective_response_mask,  # Use student_generation_mask
                        loss_agg_mode=policy_loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        ref_log_prob=ref_log_prob,
                        input_ids=model_inputs["input_ids"][:, -log_prob.shape[1]:], # Sliced to response part
                        tokenizer=self.tokenizer,
                        **extra_loss_args
                    )
                    
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
