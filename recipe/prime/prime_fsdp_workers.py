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
PRIME 奖励模型的 FSDP Worker 模块。

本模块实现了 PRIMERewardModelWorker，它是 verl 框架中的分布式 Worker，
负责在 GPU 集群上管理 PRIME 奖励模型的生命周期，包括：
  1. 构建奖励模型和参考模型（使用 FSDP 分片）
  2. 计算奖励模型分数（前向推理）
  3. 更新奖励模型参数（训练）
  4. 检查点的保存和加载
  5. 支持 CPU/GPU 之间的参数和优化器状态卸载（offload）
"""
import logging
import os
import warnings

import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config.optimizer import build_optimizer
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from .prime_core_algos import compute_dpo_abs_accuracy, compute_dpo_accuracy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PRIMERewardModelWorker(Worker):
    """
    PRIME 奖励模型的 FSDP 分布式 Worker。

    在分布式训练集群中管理 PRIME 奖励模型，负责：
      - 初始化 FSDP 分片的奖励模型和参考模型
      - 处理来自 Ray 调度器的 RPC 调用
      - 支持 Ulysses 序列并行和 CPU/GPU 参数卸载

    继承自 verl 的 Worker 基类，通过 @register 装饰器暴露
    compute_rm_score、update_rm 等接口供训练器调用。
    """

    def __init__(self, config):
        """
        初始化 Worker，设置分布式环境、设备网格和序列并行。

        参数:
            config: 奖励模型的配置对象
        """
        super().__init__()
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_nccl_backend())
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # 设置 FSDP 参数和优化器状态的 CPU 卸载选项
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # 根据分布式 world_size 归一化批次大小配置
        self.config.mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
            assert self.config.mini_batch_size % self.config.micro_batch_size_per_gpu == 0

    def _build_reward_ref_model_optimizer(self, config):
        """
        构建奖励模型、参考模型和优化器。

        流程：
          1. 加载 tokenizer 和模型配置
          2. 初始化奖励模型（CausalLM）并应用 monkey patch
          3. 使用 FSDP 包装奖励模型和参考模型
          4. 构建优化器和学习率调度器

        参数:
            config: 奖励模型配置
        返回:
            reward_module, ref_module, reward_optimizer, reward_lr_scheduler
        """
        # 以下导入是必要的
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_local_path_from_hdfs(config.model.path)

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Reward model overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForCausalLM

        trust_remote_code = False
        reward_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        reward_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(use_meta_tensor=not reward_model_config.tie_word_embeddings)
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_model_config.classifier_dropout = 0.0
            reward_model_config.hidden_dropout = "0"
            reward_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=reward_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            fused_kernel_options = config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=reward_module,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_remove_padding=config.model.get("use_remove_padding", False),
                use_fused_kernels=config.model.get("use_fused_kernels", False),
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype
            reward_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if self.rank == 0:
            print_model_size(reward_module)

        self.reward_model_config = reward_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage("Before reward model FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward_model_config.classifier_dropout = 0.0
            reward_model_config.hidden_dropout = "0"
            ref_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=copy_local_path_from_hdfs(config.model.ref_path),
                torch_dtype=torch_dtype,
                config=reward_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # some parameters may not in torch_dtype
            ref_module.to(torch_dtype)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        log_gpu_memory_usage("After reward FSDP", logger=None)

        ref_module = FSDP(
            ref_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=None,
        )

        reward_optimizer = build_optimizer(reward_module.parameters(), config.model.optim)

        total_steps = config.model.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.model.optim.get("lr_warmup_steps", -1))
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.model.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup

        reward_lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=reward_optimizer, num_warmup_steps=num_warmup_steps
        )

        return reward_module, ref_module, reward_optimizer, reward_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化模型（由 Ray 调度器调用）。

        导入外部库，构建奖励模型/参考模型/优化器，
        创建 DataParallelPRIMERewardModel 实例和检查点管理器。
        如果配置了 CPU 卸载，会将模型参数从 GPU 卸载到 CPU。
        """
        # 导入外部库到 huggingface 系统中
        import_external_libs(self.config.model.get("external_lib", None))

        from .prime_dp_rm import DataParallelPRIMERewardModel

        self.reward_module, self.ref_module, self.reward_optimizer, self.reward_lr_scheduler = (
            self._build_reward_ref_model_optimizer(config=self.config)
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            offload_fsdp_model_to_cpu(self.ref_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)

        self.rm = DataParallelPRIMERewardModel(
            config=self.config,
            reward_module=self.reward_module,
            ref_module=self.ref_module,
            reward_optimizer=self.reward_optimizer,
        )

        self.flops_counter = FlopsCounter(self.reward_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.reward_module,
            optimizer=self.reward_optimizer,
            lr_scheduler=self.reward_lr_scheduler,
            tokenizer=self.tokenizer,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        """
        计算奖励模型分数（前向推理，不更新参数）。

        使用 DP_COMPUTE_PROTO 调度模式，将数据分发到所有 DP rank 并行计算。
        同时计算 DPO 排序准确率作为评估指标。

        参数:
            data: 包含 prompt/response 输入的数据
        返回:
            DataProto，包含 rm_scores 和 q 张量以及评估指标
        """
        data = data.to(get_device_name())

        # 如果参数卸载到了 CPU，需要先加载回 GPU
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)
            load_fsdp_model_to_gpu(self.ref_module)
        micro_batch_size = self.config.micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # 执行前向计算
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            rm_scores, q, metrics = self.rm.compute_rm_score(data=data)

            prompt_length = data.batch["prompts"].shape[-1]
            response_mask = data.batch["attention_mask"][:, prompt_length:]
            acc = data.batch["acc"]

            dpo_acc = compute_dpo_accuracy(rm_scores, acc, response_mask=response_mask, n_samples=data.meta_info["n"])
            dpo_acc_abs = compute_dpo_abs_accuracy(rm_scores, acc, response_mask, n_samples=data.meta_info["n"])

            metrics["reward_model/dpo_acc"] = dpo_acc.detach().item()
            metrics["reward_model/dpo_acc_abs"] = dpo_acc_abs.detach().item()

            output = DataProto.from_dict(tensors={"rm_scores": rm_scores, "q": q}, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            offload_fsdp_model_to_cpu(self.ref_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_rm(self, data: DataProto):
        """
        更新奖励模型参数（训练步骤）。

        使用 DP_COMPUTE_PROTO 调度模式，执行 DPO 风格的奖励模型更新。
        更新后计算学习率并记录 DPO 准确率指标。

        参数:
            data: 包含训练数据（可能含 BoN 对比信息）
        返回:
            DataProto，包含更新后的 rm_scores 和训练指标
        """
        data = data.to(get_device_name())
        # 加载卸载的模型和优化器状态到 GPU
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.ref_module)
            load_fsdp_model_to_gpu(self.reward_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.reward_optimizer, device_id=get_device_id())

        # 执行训练
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            rm_scores, metrics = self.rm.update_rm(data=data)

            self.reward_lr_scheduler.step()
            lr = self.reward_lr_scheduler.get_last_lr()[0]
            metrics["rm/lr"] = lr

            prompt_length = data.batch["prompts"].shape[-1]
            response_mask = data.batch["attention_mask"][:, prompt_length:]
            acc = data.batch["acc"]

            dpo_acc_before = compute_dpo_accuracy(
                rm_scores, acc, response_mask=response_mask, n_samples=data.meta_info["n"]
            )
            dpo_acc_abs = compute_dpo_abs_accuracy(rm_scores, acc, response_mask, n_samples=data.meta_info["n"])

            metrics["reward_model/dpo_acc_before"] = dpo_acc_before.detach().item()
            metrics["reward_model/dpo_acc_abs_before"] = dpo_acc_abs.detach().item()

            output = DataProto.from_dict(tensors={"rm_scores": rm_scores}, meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
            offload_fsdp_model_to_cpu(self.ref_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.reward_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        保存奖励模型检查点（支持本地和 HDFS 存储）。

        参数:
            local_path: 本地保存路径
            hdfs_path: HDFS 保存路径（可选）
            global_step: 当前全局训练步数
            max_ckpt_to_keep: 最大保留的检查点数量
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, del_local_after_load=True):
        """
        从检查点加载奖励模型参数。

        参数:
            local_path: 本地检查点路径
            del_local_after_load: 加载后是否删除本地文件
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.reward_module)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.reward_module)
