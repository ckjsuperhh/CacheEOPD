"""
【C2C 核心】C2CHFRollout —— 把 fused-KV 注入 EOPD 学生 rollout 的 HF 后端实现（路线 A）。

继承 verl 的 HFRollout，改造点只有一处：_generate_minibatch 里把
    output = self.module.generate(...)
替换为四步：
    (a) teacher 前缀前向 → teacher KV        （FusedKVBuilder 内部）
    (b) student 前缀前向 → student KV        （FusedKVBuilder 内部）
    (c) Projector 投影融合 → fused KV        （FusedKVBuilder 内部）
    (d) 学生带 fused KV 自回归续写            （本文件 _decode_with_cache）

正确性已由 cache_eopd/prototype_generate.py 在 apex-llm 上验证：
    - self-KV 注入路径与原生 generate 逐 token 一致
    - zero_init projector 下 fused 路径与 self-KV 路径逐 token 一致
"""

import contextlib

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, DynamicCache

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.hf_rollout import HFRollout

from cache_eopd.fused_kv import (
    FusedKVBuilder,
    FusedKVConfig,
    _get_layer_kv,
    load_projector_ckpt,
)

__all__ = ["C2CHFRollout"]


class C2CHFRollout(HFRollout):
    """HFRollout + C2C fused-KV。

    额外配置（在 actor_rollout_ref.rollout 下）:
        c2c:
            teacher_path   : teacher(sharer) 模型路径；EOPD 中即 teacher_model.path
            projector_path : 可选，C2C 预训练 projector 权重(state_dict)；不给则随机初始化
            zero_init      : projector 输出层零初始化（=先等价于纯 EOPD，训练中逐步引入融合）
            enable         : 总开关，False 时行为与原版 HFRollout 完全一致
    """

    def __init__(self, module: nn.Module, config):
        # 不走 HFRollout.__init__（它调用 BaseRollout.__init__() 缺参会报错——上游遗留问题；
        # BaseRollout 新签名要求 config/model_config/device_mesh，而 HF 后端不需要它们）。
        # 直接设置 HFRollout 实际用到的两个属性。
        self.config = config
        self.module = module
        c2c_cfg = config.get("c2c", {}) or {}
        self.c2c_enabled = bool(c2c_cfg.get("enable", False))
        self._builder = None
        if self.c2c_enabled:
            teacher_path = c2c_cfg["teacher_path"]
            # 【C2C 核心】teacher 常驻 rollout worker（bf16、eager attention、eval-only）。
            # teacher_device: 缺省与 student 同卡；"auto" 用 accelerate 跨卡分片（显存紧张时）；
            # 也可指定 "cuda:N"。
            teacher_device = c2c_cfg.get("teacher_device", None) or get_device_name()
            if teacher_device == "auto":
                self.teacher_module = AutoModelForCausalLM.from_pretrained(
                    teacher_path, dtype=torch.bfloat16, attn_implementation="eager",
                    device_map="auto",
                ).eval()
            else:
                self.teacher_module = AutoModelForCausalLM.from_pretrained(
                    teacher_path, dtype=torch.bfloat16, attn_implementation="eager",
                ).to(teacher_device).eval()
            for p in self.teacher_module.parameters():
                p.requires_grad_(False)

            fkv_cfg = FusedKVConfig(
                dtype=torch.bfloat16,
                zero_init=bool(c2c_cfg.get("zero_init", False)),
                layer_mapping_strategy=c2c_cfg.get(
                    "layer_mapping", c2c_cfg.get("mapping", "relative_depth")
                ),
            )
            # module 可能是 FSDP 包装的；FusedKVBuilder 只在 summon_full_params 上下文内调用
            self._fkv_cfg = fkv_cfg
            self._projector_path = c2c_cfg.get("projector_path", None)

    # ------------------------------------------------------------------
    # BaseRollout 的三个抽象方法是给 async server 后端（vllm/sglang）用的：
    # 它们管理独立推理进程里的权重/KV 显存生命周期。HF 后端直接复用训练侧 module，
    # 权重天然同步、KV 每次 generate 后就释放，因此这里是空实现。
    async def resume(self, tags: list[str]):
        pass

    async def update_weights(self, weights, **kwargs):
        pass

    async def release(self):
        get_torch_device().empty_cache()

    # ------------------------------------------------------------------
    def _get_builder(self, student_module: nn.Module) -> FusedKVBuilder:
        """惰性构造 FusedKVBuilder（需要 student config，且 FSDP 下要在 summon 上下文内）。"""
        if self._builder is None:
            projectors = None
            if self._projector_path:
                # train_projector.py 存的是 per-layer 的 ModuleList（28 个 projector），
                # 结构必须由 ckpt 决定而不是 config，否则形状对不上会静默走随机权重。
                projectors = load_projector_ckpt(self._projector_path)
            self._builder = FusedKVBuilder.from_models(
                self.teacher_module, student_module, self._fkv_cfg, projector=projectors)
            self._builder.projectors.eval()  # 硬门控，与验收口径一致
        return self._builder

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        if not self.c2c_enabled:
            return super()._generate_minibatch(prompts)

        # ---- 采样超参解析（与父类保持一致）----
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))
        if is_validate and do_sample:
            top_p = self.config.val_kwargs.top_p
            top_k = max(0, self.config.val_kwargs.top_k)
            temperature = self.config.val_kwargs.temperature

        idx = prompts.batch["input_ids"]
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            # ==========【C2C 核心】(a)(b)(c) 构造 fused KV ==========
            builder = self._get_builder(self.module)
            fused_cache, _extras = builder.build(idx, attention_mask, position_ids)
            # ==========【C2C 核心】(d) 带 fused KV 自回归续写 ==========
            response = self._decode_with_cache(
                fused_cache, idx, attention_mask, position_ids,
                max_new_tokens=response_length,
                do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
                eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            )
        seq = torch.cat([idx, response], dim=-1)

        # ---- 以下与父类完全一致：pad 到定长 + 构造训练字段 ----
        generated_batch_size = seq.size(0)
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]
        if delta_length > 0:
            delta_tokens = torch.ones(
                size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            seq = torch.cat((seq, pad_token_id * delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:]
        resp_len = response.size(1)
        delta_position_id = torch.arange(1, resp_len + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

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
        get_torch_device().empty_cache()
        self.module.train()
        return DataProto(batch=batch)

    # ------------------------------------------------------------------
    def _decode_with_cache(
        self, cache, input_ids, attention_mask, position_ids,
        max_new_tokens, do_sample, temperature, top_p, top_k,
        eos_token_id, pad_token_id,
    ) -> torch.Tensor:
        """【C2C 核心】带外部前缀 KV 的自回归 decode loop（支持采样）。

        cache 裁掉末 token 那一格，由 loop 首步重算——与原型验证过的方案一致。
        """
        B, L = input_ids.shape
        device = input_ids.device
        cache = _crop_cache(cache, L - 1)
        cur = input_ids[:, -1:]
        cur_pos = position_ids[:, -1:]
        mask = attention_mask
        eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
        eos_tensor = torch.tensor(eos_ids, device=device)

        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            out = self.module(
                input_ids=cur, attention_mask=mask, position_ids=cur_pos,
                past_key_values=cache, use_cache=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            next_tok = _sample_token(logits, do_sample, temperature, top_p, top_k)
            next_tok = torch.where(
                finished.unsqueeze(-1), torch.full_like(next_tok, pad_token_id), next_tok)
            generated.append(next_tok)
            finished = finished | (next_tok.squeeze(-1).unsqueeze(-1) == eos_tensor).any(-1)
            if finished.all():
                break
            cur = next_tok
            cur_pos = cur_pos + 1
            mask = torch.cat([mask, (~finished).long().unsqueeze(-1)], dim=-1)
        return torch.cat(generated, dim=-1)


def _sample_token(logits, do_sample, temperature, top_p, top_k):
    """单步采样：greedy / temperature + top-k + top-p。"""
    if not do_sample:
        return logits.argmax(-1, keepdim=True)
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(-1)
        remove = cum - probs > top_p  # 保留使累计概率首次超过 top_p 的那个 token
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _crop_cache(cache, length):
    """把 DynamicCache 裁到前 length 个位置。"""
    new_cache = DynamicCache()
    idx = 0
    while True:
        try:
            k, v = _get_layer_kv(cache, idx)
        except (IndexError, AttributeError):
            break
        if k is None:
            break
        new_cache.update(k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous(), idx)
        idx += 1
        if idx > 500:
            break
    return new_cache
