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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.

【C2C 集成点 · 路线 A 入口】这是把 C2C（Cache-to-Cache）融入学生 rollout 的**最简单路线**。
与 vLLM/SGLang server 不同，HFRollout 直接调用 HuggingFace `self.module.generate(...)`，
可以传入 `past_key_values`，因此最适合做「先融合 teacher 的 KV-Cache 再让学生生成」的实验：
  1. 对学生 prompt 做一次 teacher 前向，拿到 teacher 的 KV-Cache（sharer cache）
  2. 对学生 prompt 做一次 student 前缀前向，拿到 student 的 KV-Cache（base cache）
  3. 用 rosetta 的 Projector 把 teacher KV 投影到 student 维度，与 student KV 融合（fused cache）
  4. 把 fused cache 作为 `past_key_values` 注入 `self.module.generate(...)`，让学生从此处续写
具体要改的位置见 `generate_sequences` 与 `_generate_minibatch` 中的【C2C 集成点】标记。
参考实现：C2C 的 `rosetta/model/wrapper.py` 的 `RosettaModel.forward`（两阶段 Stage1 缓存 / Stage2 融合）。
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        # 注意：HFRollout 不走 base.py 的 __init__（那里要求 config/model_config/device_mesh），
        # 而是直接持有训练侧的模型 module 与 rollout 配置 config。
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # 【C2C 集成点】rollout 的统一入口（被 fsdp_workers.ActorRolloutRefWorker.generate_sequences 调用）。
        # 每个 minibatch 的实际生成逻辑在 _generate_minibatch 中，C2C 融合 KV 的改造也在那里。
        # 1) 取出本批样本数（batch 的第一维）
        batch_size = prompts.batch.batch_size[0]
        # 2) 按 micro_batch_size 把整批切成若干 chunk，避免一次 generate 占用过多显存
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        # 3) 逐 chunk 调用 _generate_minibatch 生成，再拼接回一个 DataProto 返回
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # -------------------------------------------------------------------
        # 第 1 步：解析采样超参（可被 prompts.meta_info 中的字段逐批覆盖）
        # -------------------------------------------------------------------
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        # 根据「是否采样 / 是否校验」决定走哪套生成参数（三种分支）：
        if not do_sample:
            # do_sample==False -> 贪心解码（greedy）
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # 校验阶段且采样 -> 用 val_kwargs 里的高阶采样参数
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # 训练阶段采样 -> 用 rollout config 里的采样参数
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                # already repeat in ray_trainer
                # https://github.com/volcengine/verl/blob/2fdfbdcba6f2e076f64bc47922d8fe6cf7dc7da5/verl/trainer/ppo/ray_trainer.py#L1117
                "num_return_sequences": 1,
            }

        # 把采样参数打包成 HF 的 GenerationConfig
        generation_config = GenerationConfig(**kwargs)

        # 从 DataProto 中取出 prompt 相关张量
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # 用于构造 attention_mask 的特殊 token
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        # -------------------------------------------------------------------
        # 第 2 步：进入生成（核心）。处理 FSDP 下取全量参数 / 自动混合精度
        # -------------------------------------------------------------------
        self.module.eval()  # 切到 eval 模式（关闭 dropout 等），保证 rollout 可复现
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # FSDP 下需要 summon_full_params 临时聚合分片参数（writeback=False 不回写，recurse=False 见官方 issue）
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            # ==================== 【C2C 集成点】Fused-KV 注入处 ====================
            # 标准 HF generate 调用就在下方。要把 C2C 融入学生 rollout，需把它替换为
            # 「先构造 fused KV、再续写」的流程（参考 rosetta/model/wrapper.py 的 forward）：
            #   (a) 教师对同 prompt 前向，取 teacher_cache = teacher_model(...).past_key_values
            #   (b) 学生对 prompt 前缀前向，取 student_cache = self.module(...).past_key_values
            #   (c) 用 Projector 把 teacher_cache 投影到 student 维度并融合：
            #         fused_cache = projector.cache_project(teacher_cache, student_cache)
            #   (d) 让学生从 fused_cache 续写：self.module.generate(..., past_key_values=fused_cache)
            # 注意：直接把完整前缀 KV 传给 generate 会与内部重新 prefill 冲突；稳妥做法是
            # 把这里改写成本文件末尾那样的「自回归 decode loop」，每步用 fused_cache 作 past_key_values。
            # 更简单的临时验证：传 past_key_values=fused_cache 同时把 input_ids 切到最后一个 token。
            # ================================================================
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
                # past_key_values=fused_cache,  # 【C2C】融合后的 KV 在这里注入（需配合把 input_ids 切到末 token）
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # -------------------------------------------------------------------
        # 第 3 步：把生成结果 pad 到固定长度（response_length），构造训练用字段
        # -------------------------------------------------------------------
        # HF generate 会在整批都到达 [EOS] 时停止，因此需要把短序列 pad 到 prompt+response 总长
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            # 用 pad_token_id 填充尾部
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # num_return_sequences > 1 时，需要把 position_ids / attention_mask 按返回序列数重复
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            # num_return_sequences > 1 时，把 position_ids / attention_mask 按返回序列数重复对齐
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        # 切分 prompt 与 response 两段
        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        # 构造 response 部分的增量 position_ids（在 prompt 最后一个位置之后递增）
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # 用 eos_token 计算 response 的 attention_mask（EOS 之后置 0）
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # 组装成 TensorDict 返回
        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # 生成结束后释放显存（old_log_prob 计算前清空 cache）
        get_torch_device().empty_cache()

        self.module.train()  # 恢复 train 模式，交还给训练流程
        return DataProto(batch=batch)
