"""
The ensemble of multiple standard transformers LLM models, with automatic kv-cache projection.
It shares the same interface as the standard transformers LLM models.
多模型集成封装器 —— C2C (Cache-to-Cache) 框架的核心模型封装模块。

本模块实现了 RosettaModel 类，它将多个标准 Transformer LLM 模型封装为一个统一模型，
并自动处理 KV-Cache 的投影与融合。对外提供与标准 HuggingFace LLM 相同的接口
（forward / generate），使调用者无需关心底层的 KV-Cache 投影逻辑。

核心类:
    - RosettaModel: 主模型类，封装了以下组件：
      * model_list: 多个 LLM 模型（index 0 = base/receiver, index 1+ = sharer/teacher）
      * projector_list: KV-Cache 投影器列表
      * projector_dict: 层映射配置 {target_model: {source_model: {target_layer: [(source_layer, projector_idx)]}}}
      * kv_cache_dict: 运行时 KV-Cache 存储

核心功能:
    1. 两阶段前向传播 (forward):
       - 按 kv_cache_index 将输入分段处理
       - 非最后段：base 模型 + sharer 模型分别前向传播，缓存 KV-Cache
       - 最后段：应用 Projector 投影 sharer KV → 融合到 base KV → monkey-patch 注意力层
    2. 自回归生成 (generate):
       - Prefill 阶段调用 forward 处理完整 prompt
       - Decode 阶段逐 token 生成，支持 temperature/top-p/top-k 采样
    3. 多源融合 (multi_source_fusion_mode):
       - sequential: 逐个 source 迭代更新 base cache
       - parallel: 所有 source 从干净的 base cache 投影，然后累加残差
    4. Include-Response 模式:
       - 通过 monkey-patch Qwen3 注意力层，使 prefill 阶段直接使用融合后的 KV

辅助函数:
    - clone_kv_cache: 深拷贝 DynamicCache
    - hybrid_to_dynamic: 将 HybridCache 转换为 DynamicCache

与其他模块的关系:
    - rosetta.model.projector.Projector: 投影器接口
    - rosetta.model.sampling.sample_token: Token 采样函数
    - transformers.DynamicCache: KV-Cache 容器
"""

from typing import List, Optional, Union
import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import json

from rosetta.model.projector import Projector
from rosetta.model.sampling import sample_token
from transformers.utils import ModelOutput
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except Exception:
    GreedySearchDecoderOnlyOutput = None
    SampleDecoderOnlyOutput = None

def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    """
    深拷贝 DynamicCache（KV-Cache 容器）。
    逐层克隆 key_cache 和 value_cache，使用 .clone().detach() 确保
    拷贝后的 tensor 与原始计算图断开连接，避免梯度回传干扰。

    Args:
        kv_cache: 待克隆的 DynamicCache 对象

    Returns:
        DynamicCache: 深拷贝后的新 KV-Cache 对象
    """
    new_cache = DynamicCache()  # 创建一个新的空 DynamicCache 容器
    for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
        # 逐层克隆 key 和 value tensor，.clone() 复制数据，.detach() 断开梯度计算图
        # 这样拷贝后的缓存不会干扰原始模型的梯度回传
        new_cache.key_cache.append(k.clone().detach())
        new_cache.value_cache.append(v.clone().detach())
    return new_cache

def hybrid_to_dynamic(hybrid_cache):
    """
    将 HybridCache（混合缓存）转换为 DynamicCache（动态缓存）。
    HybridCache 是 HuggingFace transformers 的新缓存格式，
    内部可能包含 sliding window cache 等多种缓存类型。
    此函数提取其中的 key/value cache 并转为统一的 DynamicCache 格式。

    Args:
        hybrid_cache: HybridCache 或 DynamicCache 或 None

    Returns:
        DynamicCache: 转换后的动态缓存，如果输入为 None 则返回 None
    """
    if hybrid_cache is None:
        return None
    # 如果已经是 DynamicCache 格式，直接返回，无需转换
    if isinstance(hybrid_cache, DynamicCache):
        return hybrid_cache

    # 手动从 HybridCache 提取：HybridCache 可能包含 sliding window cache 等混合缓存类型，
    # 需要提取其中的 key/value cache 并转为统一的 DynamicCache 格式
    if hasattr(hybrid_cache, "key_cache") and hasattr(hybrid_cache, "value_cache"):
        keys = hybrid_cache.key_cache      # 提取所有层的 key cache 列表
        values = hybrid_cache.value_cache  # 提取所有层的 value cache 列表
        assert len(keys) == len(values), "key/value 层数不一致"

        # 将 (key, value) 对打包成 legacy cache 格式，然后用 DynamicCache.from_legacy_cache 转换
        legacy_cache = [(k, v) for k, v in zip(keys, values)]
        return DynamicCache.from_legacy_cache(legacy_cache)

    raise TypeError(f"Unsupported cache type: {type(hybrid_cache)}")  # 不支持的缓存类型

class RosettaModel(nn.Module):
    """
    Drop in replacement for the standard transformers LLM models, like Qwen3ForCausalLM.
    标准 Transformer LLM 模型（如 Qwen3ForCausalLM）的直接替换封装器。

    C2C (Cache-to-Cache) 框架的核心模型类。它将多个 LLM 模型封装为一个统一模型，
    通过 KV-Cache 投影与融合实现模型间的直接通信。对外提供与标准 HuggingFace LLM
    相同的 forward/generate 接口。

    核心组件:
        - model_list: 模型列表，index 0 为 base/receiver 模型，index 1+ 为 sharer/teacher 模型
        - projector_list: KV-Cache 投影器列表，负责将 sharer 的 KV 维度投影到 receiver 兼容维度
        - projector_dict: 层映射配置字典，定义 source 层到 target 层的投影关系
        - kv_cache_dict: 运行时 KV-Cache 存储字典，缓存各模型的中间 KV 状态

    两阶段推理流程:
        - Stage 1 (非最后段): base 模型 + sharer 模型分别前向传播，缓存 KV-Cache
        - Stage 2 (最后段): 应用 Projector 投影 sharer KV → 融合到 base KV → base 模型生成输出

    多源融合模式 (multi_source_fusion_mode):
        - sequential: 逐个 source 迭代更新 base cache（每个 source 看到的是前一个 source 融合后的 cache）
        - parallel: 所有 source 从干净的 base cache 投影，然后累加残差（各 source 独立投影，互不干扰）
    """

    def __init__(self, model_list: List[PreTrainedModel], base_model_idx = 0, projector_list: List[Projector] = [], include_response: bool = False, multi_source_fusion_mode: str = "parallel"):
        """
        初始化 RosettaModel。

        Args:
            model_list: LLM 模型列表。
                - model_list[base_model_idx] 为 base/receiver 模型（接收方）
                - 其余模型为 sharer/teacher 模型（发送方，提供 KV-Cache）
            base_model_idx: base 模型在 model_list 中的索引，默认为 0
            projector_list: KV-Cache 投影器列表，每个 Projector 负责将一个 source 层的 KV
                投影到 target 层兼容的维度和语义空间
            include_response: 是否启用 Include-Response 模式。
                若为 True，在 prefill 阶段会通过 monkey-patch 注意力层，
                使 base 模型直接使用融合后的 KV 进行注意力计算（而非仅在后续 token 生效）
            multi_source_fusion_mode: 多源融合模式。
                - "sequential": 逐个 source 迭代更新 base cache
                - "parallel": 所有 source 从干净 base cache 投影后累加残差
        """
        super().__init__()
        # model list: a list of model, model 0 by default is the base model
        # 模型列表：默认 model 0 是 base/receiver 模型，其余为 sharer 模型
        # projector list: a list of projector
        # 投影器列表：每个 Projector 实现 source KV → target KV 的维度/语义投影
        # standard init with additional model list parameter
        # kv-cache dict: key (source_model_idx, target_model_idx), value (Cache), assume only convert at prefill with one type of model
        # KV-Cache 字典：存储运行时各模型的 KV 缓存，键为 (target_model_idx, source_model_idx)，值为 Cache 对象
        # projector dict: key (source_model_idx, target_model_idx) value dict(key (source_model_layer_idx, M_target value )
        # 投影器配置字典：定义 source 层到 target 层的映射关系

        self.base_model_idx = base_model_idx  # base/receiver 模型索引
        self.model_list = nn.ModuleList(model_list)  # 使用 ModuleList 包装，确保参数注册到 PyTorch

        device = model_list[base_model_idx].device   # 获取 base 模型所在设备
        dtype = model_list[base_model_idx].dtype     # 获取 base 模型数据类型
        # 将投影器列表移到与 base 模型相同的设备和数据类型
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)

        # projector_dict: 层映射配置字典，结构为:
        # {target_model_idx: {source_model_idx: {target_layer_idx: [(source_layer_idx, projector_idx), ...]}}}
        # 表示：target_model 的 target_layer 层，需要从 source_model 的 source_layer 层
        # 通过 projector_list[projector_idx] 投影得到
        self.projector_dict = {}

        # kv_cache_dict: 运行时 KV-Cache 存储，结构为:
        # {target_model_idx: {source_model_idx: DynamicCache}}
        # 用于缓存 prefill 阶段各模型产生的 KV，供后续投影和融合使用
        self.kv_cache_dict = {}

        # monkey-patch 钩子处理器列表，用于 Include-Response 模式
        self._generation_hook_handlers = []

        # Multi-source fusion mode: 
        # "sequential" (default): each source updates base cache iteratively
        # 顺序模式：每个 source 依次更新 base cache，下一个 source 看到更新后的 cache
        # "parallel": all sources project from clean base cache, then sum projections
        # 并行模式：所有 source 从同一份干净的 base cache 投影，最后累加残差
        self.include_response = include_response  # 是否在 prefill 阶段直接使用融合后的 KV
        if multi_source_fusion_mode not in ["sequential", "parallel"]:
            raise ValueError(f"multi_source_fusion_mode must be 'sequential' or 'parallel', got '{multi_source_fusion_mode}'")
        self.multi_source_fusion_mode = multi_source_fusion_mode  # 多源融合模式

    @property
    def device(self):
        """获取 base 模型所在设备（GPU/CPU）。"""
        return self.model_list[self.base_model_idx].device
    
    def to(self, device):
        """
        Move the RosettaModel and all underlying models and projectors to the specified device.
        将 RosettaModel 及其所有子模型和投影器移动到指定设备。

        Args:
            device: 目标设备（如 'cuda:0', 'cpu'）

        Returns:
            self: 移动后的 RosettaModel 实例
        """
        super().to(device)  # 调用父类 to()，移动已注册的参数
        for model in self.model_list:
            model.to(device)       # 逐个移动子模型（确保未注册的缓冲区也被移动）
        for projector in self.projector_list:
            projector.to(device)   # 逐个移动投影器
        return self
        
    # set projector 
    # 设置投影器配置：定义 source 模型某层到 target 模型某层的投影映射关系
    def set_projector_config(self, 
                        source_model_idx: int, 
                        source_model_layer_idx: int, 
                        target_model_idx: int,
                        target_model_layer_idx: int, 
                        projector_idx: int):
        """
        Set the projector configuration.
        设置投影器配置，建立 source 模型层 → target 模型层的投影映射。

        Args:
            source_model_idx: 源模型（sharer/teacher）在 model_list 中的索引
            source_model_layer_idx: 源模型中的 Transformer 层索引
            target_model_idx: 目标模型（receiver/base）在 model_list 中的索引
            target_model_layer_idx: 目标模型中的 Transformer 层索引
            projector_idx: 使用的投影器在 projector_list 中的索引

        The projector dict structure supports multiple projectors per target layer.
        projector_dict 的结构支持每个 target 层关联多个投影器：
        Structure/结构:
        {
            target_model_idx: {
                source_model_idx: {
                    target_model_layer_idx: [(source_model_layer_idx, projector_idx), ...]
                }
            }
        }
        Repeated calls for the same (target, source, target_layer) append additional pairs.
        对相同的 (target, source, target_layer) 重复调用会追加新的映射对。
        """

        # 按需创建嵌套字典结构：target_model → source_model → target_layer
        if target_model_idx not in self.projector_dict.keys():
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx].keys():
            self.projector_dict[target_model_idx][source_model_idx] = {}
        # Accumulate list of (source_layer, projector_idx) for this target layer
        # 获取当前 target 层已有的映射列表，如果没有则创建新列表
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(target_model_layer_idx)
        if layer_entry is None:
            # 首次为该 target 层设置映射
            self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx] = [(source_model_layer_idx, projector_idx)]
        else:
            # 追加新的 (source_layer, projector_idx) 对，支持一个 target 层从多个 source 层融合
            layer_entry.append((source_model_layer_idx, projector_idx))


    def load_projector(self, projector_list):
        """
        加载/替换投影器列表。

        Args:
            projector_list: 新的 Projector 列表，替换当前的 projector_list
        """
        self.projector_list: List[Projector] = projector_list

    def get_projector(self, 
                        source_model_idx, 
                        source_model_layer_idx, 
                        target_model_idx,
                        target_model_layer_idx):
        """
        获取指定 source 层到 target 层的投影器。

        查找流程：
        1. 从 projector_dict 中获取该 target 层对应的 (source_layer, projector_idx) 列表
        2. 优先返回 source_model_layer_idx 精确匹配的投影器
        3. 若无精确匹配，返回列表中第一个投影器作为后备

        Args:
            source_model_idx: 源模型索引
            source_model_layer_idx: 源模型层索引
            target_model_idx: 目标模型索引
            target_model_layer_idx: 目标模型层索引

        Returns:
            Projector: 对应的投影器实例

        Raises:
            ValueError: 如果没有为该 target 层配置任何投影器
        """
        # 从 projector_dict 获取该 target 层的映射对列表
        pair_list = self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        # Prefer exact source layer match / 优先精确匹配 source 层
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        # Fallback: return the first projector / 后备：返回第一个投影器
        return self.projector_list[pair_list[0][1]]

    @staticmethod
    def load_json(file_name):
        """
        从 JSON 文件加载数据。

        Args:
            file_name: JSON 文件路径

        Returns:
            dict/list: 解析后的 Python 对象
        """
        with open(file_name, "r") as f:
            result = json.load(f)
        return result
    
    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        """
        Recursively convert dictionary keys that look like integers back to int.
        This reverses json.dump's coercion of dict keys to strings.
        递归地将字典中形如整数的字符串键转换回 int 类型。
        这是为了逆转 json.dump 将 dict 键强制转为字符串的行为，
        因为 projector_dict 的键（模型索引、层索引）需要是整数。

        Args:
            obj: 待转换的对象（dict/list/基本类型）

        Returns:
            转换后的对象，所有形如整数的字符串键已转为 int
        """
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                # 判断键是否为整数字符串（支持负号）
                if isinstance(key, str) and key.lstrip('-').isdigit():
                    new_key = int(key)   # 将 "0" → 0, "12" → 12
                else:
                    new_key = key
                new_obj[new_key] = RosettaModel._convert_dict_keys_to_ints(value)  # 递归处理子对象
            return new_obj
        if isinstance(obj, list):
            # 递归处理列表中的每个元素
            return [RosettaModel._convert_dict_keys_to_ints(v) for v in obj]
        return obj  # 基本类型直接返回
    
    
    def save_projector_config(self, file_name):
        """
        将投影器配置（projector_dict）保存为 JSON 文件。
        注意：JSON 会将 int 键转为字符串，加载时需要用 _convert_dict_keys_to_ints 恢复。

        Args:
            file_name: 输出 JSON 文件路径
        """
        with open(file_name, "w") as f:
            json.dump(self.projector_dict, f)

    
    def load_projector_config(self, config_path):
        """
        从 JSON 文件加载投影器配置（projector_dict）。
        加载后自动将字符串键转换回整数。

        Args:
            config_path: JSON 配置文件路径
        """
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            # 加载后将字符串键转回整数，恢复原始的嵌套字典结构
            self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)

    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        """
        设置指定 source→target 模型对的 KV-Cache。
        用于在运行时存储/初始化各模型的 KV 缓存状态。

        Args:
            source_model_idx: 源模型索引
            target_model_idx: 目标模型索引
            cache: DynamicCache 对象，若为 None 则初始化为空的 DynamicCache
        """
        if target_model_idx not in self.kv_cache_dict.keys():
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            # Initialize with a DynamicCache instead of RosettaCache for now
            # 如果没有提供缓存，初始化一个空的 DynamicCache
            self.kv_cache_dict[target_model_idx][source_model_idx] = DynamicCache() # noqa, maybe we should use RosettaCache here
        else:
            # 使用提供的缓存对象
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

    @staticmethod
    def _monkeypatch_qwen3_attention_forward(attn_module, new_k_cache, new_v_cache):
        """
        Monkeypatch Qwen3Attention.forward so that *current step* attention uses the
        provided key/value (in cache space) before computing attention.
        【C2C 集成点】把「融合后的 KV」塞进学生注意力层的底层机制：不改 transformers 源码，
        而是临时替换注意力 forward，让当前步直接使用 new_k_cache/new_v_cache（已是 teacher 投影后的空间）。
        移植到 verl 时，若 teacher/student 同为 Qwen 系列可复用此思路；否则需对应改 student 模型的注意力类。
        猴子补丁（Monkey-patch）Qwen3 的注意力前向传播函数，使当前步骤的注意力计算
        使用提供的 key/value 张量（已经过投影融合），而非模型自身投影产生的原始 KV。

        This avoids editing transformers' Qwen3 code while ensuring the modified KV
        is used in the same forward pass (not just for the next token).
        这样可以在不修改 transformers 库源码的情况下，确保融合后的 KV 在当前前向传播中被使用
        （而非仅在下一个 token 的解码阶段生效）。

        new_k_cache/new_v_cache: (B, kv_heads, q_len, head_dim) in the SAME space as
        Qwen3Attention's key_states/value_states AFTER k_norm + RoPE (k) and reshape (v).
        注意：new_k_cache/new_v_cache 的形状为 (B, kv_heads, q_len, head_dim)，
        必须与 Qwen3Attention 中经过 k_norm + RoPE 旋转位置编码后的 key_states/value_states
        处于同一表示空间。

        Args:
            attn_module: 待 patch 的 Qwen3Attention 模块实例
            new_k_cache: 融合后的 key 张量，形状 (B, kv_heads, q_len, head_dim)
            new_v_cache: 融合后的 value 张量，形状 (B, kv_heads, q_len, head_dim)

        Returns:
            orig_forward: 原始的 forward 方法引用，用于后续恢复（remove_hooks）
        """
        import types

        # Lazy imports to avoid hard dependency at module import time
        # 延迟导入，避免在模块加载时产生对 Qwen3 的硬依赖
        from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
            apply_rotary_pos_emb,       # RoPE 旋转位置编码应用函数
            eager_attention_forward,    # 标准注意力实现（非 FlashAttention）
            ALL_ATTENTION_FUNCTIONS,    # 注意力函数注册表（包含 sdpa/flash_attn 等）
        )

        orig_forward = attn_module.forward  # 保存原始 forward，用于后续恢复

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            # This is essentially Qwen3Attention.forward with one injection point.
            # 以下是 Qwen3Attention.forward 的完整复刻，仅在一个注入点做了替换。

            # Step 1: 对 hidden_states 做 Q/K/V 线性投影 + 归一化 + reshape
            # hidden_states 形状: (B, q_len, hidden_dim)
            input_shape = hidden_states.shape[:-1]  # (B, q_len)
            hidden_shape = (*input_shape, -1, self.head_dim)  # (B, q_len, num_heads, head_dim)

            # Q 投影 + Q 归一化 + reshape + transpose → (B, num_heads, q_len, head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            # K 投影 + K 归一化 + reshape + transpose → (B, kv_heads, q_len, head_dim)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            # V 投影 + reshape + transpose → (B, kv_heads, q_len, head_dim)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # Step 2: 应用 RoPE 旋转位置编码到 Q 和 K
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # === Injection point (before cache update & attention) ===
            # === 注入点：在缓存更新和注意力计算之前 ===
            # Replace current-token key/value with provided cache-space tensors.
            # 用融合后的 KV 替换当前 token 的 key/value
            # Expect same shape as key_states/value_states at this moment:
            # 期望形状与当前 key_states/value_states 相同：(B, kv_heads, q_len, head_dim)
            if new_k_cache is not None and new_v_cache is not None:
                # Only replace if compatible / 仅在形状兼容时替换
                if key_states.shape == new_k_cache.shape:
                    key_states = new_k_cache    # 替换为融合后的 key
                if value_states.shape == new_v_cache.shape:
                    value_states = new_v_cache  # 替换为融合后的 value

            # Step 3: 更新 KV-Cache（将当前 token 的 KV 追加到缓存中）
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Step 4: 选择注意力计算接口
            attention_interface = eager_attention_forward  # 默认使用标准注意力
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    # fall back to eager, same as upstream behavior (warning omitted here)
                    # SDPA 不支持输出注意力权重时回退到 eager 实现
                    attention_interface = eager_attention_forward
                else:
                    # 使用配置的注意力实现（如 FlashAttention、SDPA 等）
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            # Step 5: 执行注意力计算
            # 输入: query (B, num_heads, q_len, head_dim), key/value (B, kv_heads, kv_len, head_dim)
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            # Step 6: 输出投影（O 投影）+ reshape
            # attn_output reshape: (B, q_len, num_heads * head_dim) → (B, q_len, hidden_dim)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)  # O 线性投影
            return attn_output, attn_weights

        # 用 types.MethodType 将 patched_forward 绑定为 attn_module 的方法
        attn_module.forward = types.MethodType(patched_forward, attn_module)
        return orig_forward  # 返回原始 forward 以供后续恢复

    def register_hooks(self, input_ids, attention_mask, position_ids, base_kv_cache, source_model_idx, source_kv_cache):
        """
        注册 Include-Response 模式的 monkey-patch 钩子。
        【C2C 集成点】这是「融合 KV 注入学生注意力」的装配入口：先分别跑 base/teacher 前向拿 KV，
        再用 Projector 投影替换 base 各层 KV，最后对每层注意力做 monkey-patch。
        在 verl 集成里可简化为：直接构造 fused_cache，然后 monkey-patch 或传 past_key_values。

        Include-Response 模式的核心思路：在 prefill 阶段，不仅缓存 KV，
        还通过投影融合让 base 模型在计算注意力时直接使用融合后的 KV，
        而不是等到下一个 token 的 decode 阶段才生效。

        实现流程：
        1. 分别用 base 模型和 source 模型对 input_ids 做前向传播，获取各自的 KV-Cache
        2. 遍历 projector_dict 中的层映射，对 source KV 做投影并替换 base KV
        3. 对 base 模型的每一层注意力模块做 monkey-patch，注入融合后的 KV

        Args:
            input_ids: 当前段的 token ID，形状 (B, seq_len)
            attention_mask: 注意力掩码，形状 (B, total_seq_len)
            position_ids: 位置 ID，形状 (B, seq_len)
            base_kv_cache: base 模型已有的 KV-Cache（来自前序段）
            source_model_idx: source 模型索引
            source_kv_cache: source 模型已有的 KV-Cache（来自前序段）

        Returns:
            hook_handlers: monkey-patch 处理器列表 [(attn_module, orig_forward), ...]
            base_output_kv_cache: base 模型前向传播后的 KV-Cache
            source_output_kv_cache: source 模型前向传播后的 KV-Cache
        """

        # 深拷贝 base 和 source 的 KV-Cache，避免前向传播修改原始缓存
        base_kv_copy = clone_kv_cache(base_kv_cache)
        source_kv_copy = clone_kv_cache(source_kv_cache)

        new_length = input_ids.shape[1]  # 当前段的 token 数量

        # 用 base 模型做前向传播，获取其 KV-Cache（包含历史 + 当前段）
        base_output_kv_cache = self.model_list[self.base_model_idx].forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    past_key_values=base_kv_copy,
                    labels=None,
                    use_cache=True, 
                ).past_key_values
        # 用 source 模型做前向传播，获取其 KV-Cache
        source_output_kv_cache = self.model_list[source_model_idx].forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    past_key_values=source_kv_copy,
                    labels=None,
                    use_cache=True, 
                ).past_key_values        
        # 以 base 输出为基底创建融合后的 KV-Cache
        fused_kv_cache = clone_kv_cache(base_output_kv_cache)

        # 遍历 projector_dict 中 base_model ← source_model 的层映射
        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
            # 提取 base 模型在 target 层的 KV，只取当前段的部分（最后 new_length 个 token）
            base_key_cache, base_value_cache = base_output_kv_cache[target_layer_idx]
            new_base_key_cache = base_key_cache[:, :, -new_length:, :]   # (B, kv_heads, new_length, head_dim)
            new_base_value_cache = base_value_cache[:, :, -new_length:, :]
            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

            pair_list = entry  # 该 target 层的 (source_layer, projector_idx) 映射列表

            projected_kv_list = []  # 投影后的 KV 列表
            source_kv_list = []     # 原始 source KV 列表
            for source_model_layer_idx, projector_idx in pair_list:
                # 提取 source 模型对应层的 KV，只取当前段部分
                source_key_cache, source_value_cache = source_output_kv_cache[source_model_layer_idx]
                new_source_key_cache = source_key_cache[:, :, -new_length:, :]   # (B, kv_heads, new_length, head_dim)
                new_source_value_cache = source_value_cache[:, :, -new_length:, :]
                new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                # 通过投影器将 source KV 投影到 target 空间
                projected_key, projected_value = self.projector_list[projector_idx].forward(
                    new_source_kv_cache, # tuple of (key, value), each of shape (B, N, H, D)
                    new_base_kv_cache    # base KV 作为投影的参考/条件
                )
                projected_kv_list.append((projected_key, projected_value))
                source_kv_list.append(new_source_kv_cache)

            # Use first projector result / 使用第一个投影器的结果作为融合结果
            agg_key, agg_value = projected_kv_list[0]

            # Update cache: 将融合后的 KV 写入 fused_kv_cache 的对应层（替换当前段部分）
            fused_kv_cache.key_cache[target_layer_idx][:, :, -new_length:, :] = agg_key
            fused_kv_cache.value_cache[target_layer_idx][:, :, -new_length:, :] = agg_value

        # Monkeypatch attention forward so the modified KV is used in *this* forward pass.
        # 对 base 模型的每一层注意力模块做 monkey-patch，注入融合后的 KV
        # 这样 base 模型在下一次 forward 时，注意力计算会直接使用融合后的 KV
        hook_handlers = []  # list of (attn_module, orig_forward)
        for i in range(self.model_list[self.base_model_idx].config.num_hidden_layers):
            attn = self.model_list[self.base_model_idx].model.layers[i].self_attn
            # 提取融合后 KV 在当前段的部分，形状 (B, kv_heads, new_length, head_dim)
            new_k = fused_kv_cache.key_cache[i][:, :, -new_length:, :]
            new_v = fused_kv_cache.value_cache[i][:, :, -new_length:, :]
            orig_forward = RosettaModel._monkeypatch_qwen3_attention_forward(attn, new_k, new_v)
            hook_handlers.append((attn, orig_forward))

        return hook_handlers, base_output_kv_cache, source_output_kv_cache
    
    def remove_hooks(self, hook_handlers):
        """
        移除之前注册的 monkey-patch 钩子，恢复注意力模块的原始 forward 方法。

        Args:
            hook_handlers: register_hooks 返回的处理器列表 [(attn_module, orig_forward), ...]
        """
        # Restore monkeypatched forwards / 恢复所有被 monkey-patch 的注意力模块
        for attn, orig_forward in hook_handlers:
            attn.forward = orig_forward

    def forward(
        self,
        kv_cache_index: Optional[List] = None,
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # **kwargs: Unpack[KwargsForCausalLM],
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass / 前向传播

        C2C 框架的核心前向传播方法。将输入按 kv_cache_index 分段处理：
        - 非最后段（Stage 1）: base 模型 + 所有 sharer 模型分别前向传播，缓存各自 KV-Cache
        - 最后段（Stage 2）: 对 base 模型前向传播，同时应用投影融合（若配置了 projector）

        Args:
            kv_cache_index: 分段控制列表，每个元素形状 (B, sec_seq_len, 2)。
                元素 [i][0][0][0] 控制 sharer 选择：
                - -1: 无投影（仅 receiver，跳过所有 sharer）
                - 0: 自投影（receiver 从自身投影）— 当前未使用
                - >0: 位掩码选择 sharer：1(001)=sharer1, 2(010)=sharer2, 3(011)=两者, 7(111)=全部三个
                每个 bit 对应一个 sharer：bit i 选择 model_list[i+1] 处的 sharer
            input_ids: token ID。若为 LongTensor，所有模型使用相同输入；若为 List，每个模型使用各自的输入
            attention_mask: 注意力掩码，格式同 input_ids
            position_ids: 位置 ID，形状 (B, seq_len)
            past_key_values: 已有的 KV-Cache（用于增量解码）
            inputs_embeds: 嵌入输入（替代 input_ids）
            labels: 标签，用于计算损失
            use_cache: 是否使用 KV-Cache
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            cache_position: 缓存位置
            logits_to_keep: 保留的 logits 数量

        Returns:
            CausalLMOutputWithPast: 包含 logits、past_key_values 等的输出
        """

        # Handle different input formats: if input_ids is a list, use per-model inputs
        # 处理不同输入格式：如果 input_ids 是列表，则为每个模型使用不同的输入
        if isinstance(input_ids, list):
            # Use list format: different input_ids and attention_mask for each model
            # 列表格式：每个模型有各自独立的 input_ids 和 attention_mask
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            # Use tensor format: same input_ids and attention_mask for all models (backward compatibility)
            # 张量格式：所有模型使用相同的 input_ids 和 attention_mask（向后兼容）
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        # 当序列长度 > 1 时，视为 prefill 阶段，清空 KV-Cache 字典
        # （单次 forward 调用中的多段处理需要从头开始缓存）
        if seqlen > 1:
            self.kv_cache_dict = dict()
            
        # 计算分段数量和每段的起止位置
        # kv_cache_index 将输入序列分成多个段，每段有不同的 KV-Cache 处理策略
        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1

        # 计算每段的长度和起始位置
        section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)] if kv_cache_index is not None else [seqlen]
        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)
        
        curr_base_kv_cache = past_key_values  # 当前 base 模型的 KV-Cache（跨段传递）

        # ========== 分段处理主循环 ==========
        # 遍历每个段，非最后段做 Stage 1（缓存 KV），最后段做 Stage 2（投影融合 + 输出）
        # 【C2C 集成点 · 路线 A 参考】这正是「把 teacher KV 融合进学生生成」的标准实现，
        # 移植到 verl 的 HFRollout._generate_minibatch 时，逻辑等价于：
        #   - 非最后段：student 与 teacher 各跑一次前缀前向，分别把 KV 存进 kv_cache_dict
        #   - 最后段：对 student 做前向时，把 teacher 投影后的 KV 融合进去再算注意力
        for i in range(num_sections):
            start = section_starts[i]      # 当前段在序列中的起始位置
            end = section_starts[i + 1]    # 当前段在序列中的结束位置
            # 切分当前段的输入
            prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
            # attention_mask 需要覆盖从序列开头到当前段末尾（因果关系需要看到所有历史 token）
            prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None
            prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None
            prefill_labels = labels[:, start:end] if labels is not None else None

            if i == num_sections - 1:
                # ========== Stage 2: 最后段 —— 投影融合 + base 模型前向传播 ==========
                # 【C2C 集成点 · 核心】融合后的 KV 在这里被「使用」：
                # 若 include_response，register_hooks 已把融合 KV monkey-patch 进注意力层；
                # 否则下方的 base 模型前向会用 curr_base_kv_cache（已被投影融合过的 KV）做上下文。
                # 这是学生基于「teacher 增强后的 KV」生成 token 的关键一步。

                if self.include_response:
                    # Include-Response 模式：在 base 模型前向传播之前，先注册 monkey-patch 钩子
                    # 使 base 模型在注意力计算时直接使用融合后的 KV
                    hook_handlers, base_output_kv_cache, source_output_kv_cache = self.register_hooks(input_ids=prefill_input_ids, attention_mask=prefill_attention_mask, position_ids=prefill_position_ids,
                                                        base_kv_cache=self.kv_cache_dict[self.base_model_idx][self.base_model_idx],
                                                        source_model_idx=1, 
                                                        source_kv_cache=self.kv_cache_dict[self.base_model_idx][1])

                # calculate target model kvcache
                # base 模型前向传播：使用当前段的 input_ids + 历史 KV-Cache
                # 如果启用了 include_response，注意力层已被 monkey-patch，会使用融合后的 KV
                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask, 
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,      # 最后段传入 labels 以计算损失
                    use_cache=True, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs
                )

                if self.include_response:
                    # 移除 monkey-patch 钩子，恢复原始注意力 forward
                    self.remove_hooks(hook_handlers)

                    # 更新 kv_cache_dict 中 base 和 source 模型的 KV-Cache
                    # 用于后续 decode 阶段（如果需要的话）
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(base_output_kv_cache)
                    self.kv_cache_dict[self.base_model_idx][1] = clone_kv_cache(source_output_kv_cache)

            else:
                # ========== Stage 1: 非最后段 —— 分别缓存 base 和各 sharer 模型的 KV ==========
                # 【C2C 集成点】这里同时拿到 student(base) 与 teacher(sharer) 各自对 prompt 的 KV-Cache，
                # 是后续投影融合的输入。移植到 verl 时，对应「先做 teacher/student 两次前缀前向」。

                # Step 1: base 模型前向传播，获取当前段的 KV-Cache
                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask, 
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=use_cache, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs
                )

                # 保存 base 模型的 KV-Cache 到 kv_cache_dict（深拷贝，防止后续操作修改）
                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)

                # 更新 curr_base_kv_cache 供下一段使用（包含历史 + 当前段的完整 KV）
                curr_base_kv_cache: DynamicCache = output.past_key_values
            
                # Step 2: 遍历所有 source/sharer 模型（index 1+），分别前向传播获取 KV-Cache
                for source_model_idx in range(1, len(self.model_list)):
                    # 初始化 kv_cache_dict 中 source 模型的条目
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                    # Get model-specific input_ids and attention_mask
                    # 获取 source 模型专属的 input_ids 和 attention_mask
                    if isinstance(input_ids, list):
                        # 列表格式：每个模型有独立的输入
                        source_input_ids = input_ids[source_model_idx]
                        source_attention_mask = attention_mask[source_model_idx] if attention_mask is not None else None
                        source_prefill_input_ids = source_input_ids[:, start:end] if source_input_ids is not None else None
                        source_prefill_attention_mask = source_attention_mask[:, :end] if source_attention_mask is not None else None
                    else:
                        # Backward compatibility: use same input for all models
                        # 向后兼容：所有模型使用相同输入
                        source_prefill_input_ids = prefill_input_ids
                        source_prefill_attention_mask = prefill_attention_mask

                    model = self.model_list[source_model_idx]
                    # 记录 source 模型当前的训练模式和梯度检查点状态
                    # 因为 source 模型在 Stage 1 只做前向传播（不计算梯度），需要临时切换到 eval 模式
                    was_training = model.training
                    had_gc = getattr(model, "is_gradient_checkpointing", False)

                    try:
                        if was_training:
                            model.eval()              # 临时切到 eval 模式，禁用 dropout 等训练行为
                        if had_gc:
                            model.gradient_checkpointing_disable()  # 禁用梯度检查点（推理不需要）

                        with torch.no_grad():  # 禁用梯度计算（sharer 模型不参与训练）
                            out = model(
                                input_ids=source_prefill_input_ids,
                                attention_mask=source_prefill_attention_mask,
                                position_ids=prefill_position_ids,
                                # 使用 source 模型之前缓存的 KV-Cache（增量更新）
                                past_key_values=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                                use_cache=True,
                                return_dict=True,
                            )
                            curr_source_kv_cache = out.past_key_values  # 获取 source 模型的 KV 输出
                    finally:
                        # 恢复 source 模型的原始训练模式和梯度检查点设置
                        if had_gc:
                            model.gradient_checkpointing_enable()
                        if was_training:
                            model.train()
                    
                    # 将 HybridCache 转为 DynamicCache（兼容不同模型的缓存格式）
                    curr_source_kv_cache = hybrid_to_dynamic(curr_source_kv_cache)
                    # 深拷贝并保存到 kv_cache_dict，供后续投影使用
                    self.kv_cache_dict[self.base_model_idx][source_model_idx] = clone_kv_cache(curr_source_kv_cache)

                # ========== Step 3: 应用投影融合（Projection + Fusion） ==========
                # calculate source model kvcache and apply projections
                # 【C2C 集成点】teacher KV → student 维度的投影与融合发生在这里：
                # 遍历 projector_dict 的层映射，调用 Projector.forward 把 source(teacher) KV
                # 投影并累加到 base(student) KV，得到 fused cache。这一步对应 verl 集成里的
                # `projector.cache_project(teacher_cache, student_cache)`。
                # 如果 base 模型在 projector_dict 中有配置，则对 source KV 做投影并融合到 base KV
                if self.base_model_idx in self.projector_dict:
                    # Iterate over all source models in projector_dict
                    # 从 kv_cache_index 读取 sharer 选择位掩码
                    sharer_mask = kv_cache_index[i][0][0][0].item()
                    if sharer_mask > 0:
                        # 克隆一份干净的 base cache（用于 parallel 模式的基准投影）
                        base_cache = clone_kv_cache(curr_base_kv_cache)

                        # For parallel mode, accumulate residuals for each target layer
                        # 并行模式：初始化残差累加器 {target_layer_idx: (delta_key, delta_value)}
                        parallel_delta_cache = {} if self.multi_source_fusion_mode == "parallel" else None
                        
                        # Compute and apply projections (shared logic for both modes)
                        # 遍历所有配置的 source 模型
                        for source_model_idx in self.projector_dict[self.base_model_idx].keys():
                            # Check if this sharer is selected: bit (source_model_idx - 1)
                            # 检查位掩码：bit (source_model_idx - 1) 是否被设置
                            # 例如 sharer_mask=3(011) → 选中 source_model_idx=1 和 2
                            if not (sharer_mask & (1 << (source_model_idx - 1))):
                                continue  # 该 sharer 未被选中，跳过
                            # 根据融合模式选择投影的基准 cache
                            if self.multi_source_fusion_mode == "sequential":
                                # 顺序模式：使用当前（可能已被前一个 source 修改的）base cache
                                base_cache_ref = curr_base_kv_cache
                            else:
                                # Parallel: always project from the clean cloned base cache
                                # 并行模式：始终使用干净的克隆 cache 作为投影基准
                                base_cache_ref = base_cache

                            # 遍历该 source → base 的所有目标层映射
                            for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                                # Get base KV cache slice for projection
                                # 提取 base 模型在目标层的 KV，取当前段的切片
                                base_key_cache, base_value_cache = base_cache_ref[target_layer_idx]
                                new_base_key_cache = base_key_cache[:, :, start:end, :]    # (B, kv_heads, sec_len, head_dim)
                                new_base_value_cache = base_value_cache[:, :, start:end, :]
                                new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

                                pair_list = entry  # (source_layer, projector_idx) 映射列表

                                projected_kv_list = []  # 存储各投影器的输出
                                source_kv_list = []     # 存储原始 source KV（可能用于调试/可视化）
                                for source_model_layer_idx, projector_idx in pair_list:
                                    # 从 kv_cache_dict 提取 source 模型在对应层的 KV，取当前段切片
                                    source_key_cache, source_value_cache = self.kv_cache_dict[self.base_model_idx][source_model_idx][source_model_layer_idx]
                                    new_source_key_cache = source_key_cache[:, :, start:end, :]    # (B, kv_heads, sec_len, head_dim)
                                    new_source_value_cache = source_value_cache[:, :, start:end, :]
                                    new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                                    # 调用投影器：将 source KV 投影到 target 空间
                                    # Projector.forward(source_kv, base_kv) → (projected_key, projected_value)
                                    # 投影器内部可能实现: projected = MLP(source_kv) + base_kv（残差连接）
                                    projected_key, projected_value = self.projector_list[projector_idx].forward(
                                        new_source_kv_cache,  # source KV: (B, kv_heads, sec_len, head_dim)
                                        new_base_kv_cache     # base KV 作为参考/条件
                                    )
                                    projected_kv_list.append((projected_key, projected_value))
                                    source_kv_list.append(new_source_kv_cache)

                                # Use first projector result / 使用第一个投影器的结果
                                agg_key, agg_value = projected_kv_list[0]

                                # Collect or apply projection based on mode
                                # 根据融合模式，立即应用或延迟累加投影结果
                                if self.multi_source_fusion_mode == "sequential":
                                    # Sequential: apply immediately so next source sees updated cache
                                    # 顺序模式：立即写入 curr_base_kv_cache，
                                    # 下一个 source 会看到已融合的结果
                                    curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                                    curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                                else:
                                    # Parallel: accumulate residuals (agg - base) for this target layer
                                    # 并行模式：累加残差 delta = projected - original_base
                                    # 每个 source 独立计算残差，最后一次性累加到 base 上
                                    if target_layer_idx not in parallel_delta_cache:
                                        # 初始化该层的残差累加器为零张量
                                        parallel_delta_cache[target_layer_idx] = (
                                            torch.zeros_like(new_base_key_cache),
                                            torch.zeros_like(new_base_value_cache),
                                        )
                                    delta_key, delta_value = parallel_delta_cache[target_layer_idx]
                                    # 累加当前 source 的残差：delta += (projected - base_original)
                                    delta_key = delta_key + (agg_key - new_base_key_cache)
                                    delta_value = delta_value + (agg_value - new_base_value_cache)
                                    parallel_delta_cache[target_layer_idx] = (delta_key, delta_value)

                        # For parallel mode, apply all accumulated residuals in one shot
                        # 并行模式：所有 source 的残差累加完毕后，一次性应用到 base cache
                        if self.multi_source_fusion_mode == "parallel":
                            for target_layer_idx, (delta_key, delta_value) in parallel_delta_cache.items():
                                # 从干净的 base_cache 中取出原始值
                                base_key_cache, base_value_cache = base_cache[target_layer_idx]
                                base_key_slice = base_key_cache[:, :, start:end, :]
                                base_value_slice = base_value_cache[:, :, start:end, :]
                                # 最终融合结果 = 原始 base + 所有 source 的残差之和
                                # fused = base_original + Σ(projected_i - base_original)
                                curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = base_key_slice + delta_key
                                curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = base_value_slice + delta_value

                # 将融合后的 KV-Cache 设为 output 的 past_key_values
                # 下一段（或 decode 阶段）将基于这个融合后的 cache 继续
                output.past_key_values = curr_base_kv_cache
                                                                             
        return output
    
    @torch.no_grad()
    def generate(
        self,
        kv_cache_index,
        input_ids,
        max_new_tokens: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        max_length: Optional[int] = None,
        use_cache: bool = True,
        streamer = None,
        *args,
        **kwargs,
    ):
        """
        New generation loop without using the base model's generate.
        自回归生成方法，不依赖 base 模型自带的 generate。

        - Uses this module's forward for prefill and per-token decode.
          使用本模块的 forward 做 prefill（处理完整 prompt）和逐 token 解码。
        - Samples tokens via rosetta.model.sampling.sample_token.
          通过 rosetta.model.sampling.sample_token 采样 token。
        Returns a tensor of shape [batch, prompt_len + generated_len] for the base model stream.
        返回形状为 [batch, prompt_len + generated_len] 的张量（base 模型流的完整序列）。

        Args:
            kv_cache_index: 分段控制信息（同 forward）
            input_ids: 输入 token ID（LongTensor 或 List）
            max_new_tokens: 最大生成 token 数
            past_key_values: 已有的 KV-Cache
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            eos_token_id: 结束 token ID（int 或 List[int]）
            pad_token_id: 填充 token ID
            temperature: 采样温度。0.0 = greedy（贪心），>0 = 随机采样
            top_p: nucleus 采样概率阈值
            top_k: top-k 采样的 k 值，-1 表示不限制
            repetition_penalty: 重复惩罚系数（HuggingFace 风格），1.0 = 无惩罚
            presence_penalty: 存在惩罚（出现过的 token 统一减去该值）
            frequency_penalty: 频率惩罚（按出现次数比例减去该值）
            do_sample: 是否使用随机采样（None/False = greedy）
            return_dict_in_generate: 是否以字典格式返回（兼容 HF generate 输出）
            output_scores: 是否输出每步的 logits
            max_length: 最大总长度（prompt + 生成），与 max_new_tokens 二选一
            use_cache: 是否使用 KV-Cache
            streamer: 流式输出器（如 TextStreamer），用于实时输出 token

        Returns:
            若 return_dict_in_generate=True: 返回 GreedySearchDecoderOnlyOutput 或 SampleDecoderOnlyOutput
            否则: 返回 all_input_ids 张量 (B, prompt_len + generated_len)
        """

        self.kv_cache_dict = dict()  # 清空 KV-Cache 字典，从头开始

        # Derive number of tokens to generate / 推导需要生成的 token 数量
        # If max_new_tokens not provided, infer from max_length
        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)  # prompt 的 token 数量

        # Default eos/pad from base model tokenizer/config if not provided
        # 如果未指定 eos/pad token，尝试从 base 模型的 config 中获取
        base_model = self.model_list[self.base_model_idx]
        gen_cfg = getattr(base_model, "generation_config", None)
        cfg_obj = gen_cfg if gen_cfg is not None else getattr(base_model, "config", None)
        if eos_token_id is None and cfg_obj is not None:
            eos_token_id = getattr(cfg_obj, "eos_token_id", None)
        if pad_token_id is None and cfg_obj is not None:
            pad_token_id = getattr(cfg_obj, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            # 如果 pad_token_id 未指定，用 eos_token_id 代替
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]

        # 计算 max_new_tokens（如果未提供，从 max_length 推导）
        if max_new_tokens is None:
            if max_length is not None:
                if max_length <= prompt_len:
                    max_new_tokens = 0
                else:
                    max_new_tokens = max_length - prompt_len
            else:
                raise ValueError("Provide max_new_tokens or max_length")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        # Resolve base inputs / 解析 base 模型的输入
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        # 如果没有提供 attention_mask，默认全 1（所有 token 都可见）
        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(base_input_ids, dtype=torch.long, device=base_input_ids.device)

        batch_size = base_input_ids.size(0)  # 批处理大小

        # ========== Prefill 阶段 ==========
        # 调用 forward 处理完整 prompt，构建所有模型的 KV-Cache，获取初始 logits
        prefill_output = self.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            *args,
            **kwargs,
        )

        current_past = prefill_output.past_key_values  # 融合后的 KV-Cache（用于后续 decode）
        all_input_ids = base_input_ids                  # 累积所有已生成的 token（prompt + 生成部分）
        current_attention_mask = base_attention_mask    # 累积注意力掩码

        # Initialize streamer with prompt if provided
        # 如果提供了 streamer，先输出完整 prompt
        if streamer is not None:
            streamer.put(base_input_ids)

        # EOS handling setup / EOS（结束标记）处理初始化
        eos_set = None
        if eos_token_id is not None:
            eos_set = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        # 每个 batch 项的完成状态，初始全部为 False
        finished = torch.zeros(batch_size, dtype=torch.bool, device=all_input_ids.device)

        # Start from last prefill logits / 从 prefill 的最后一个 token 的 logits 开始
        last_logits = prefill_output.logits[:, -1, :]  # (B, vocab_size)

        # Determine sampling mode / 确定采样模式
        if do_sample is None:
            do_sample = False
        # temperature=0.0 表示 greedy 采样（argmax），>0 表示随机采样
        effective_temperature = temperature if do_sample else 0.0

        # Optional scores collection / 可选的 logits 收集（用于返回每步的分数）
        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        # ========== Decode 阶段：逐 token 自回归生成 ==========
        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)  # 收集当前步的 logits

            # Apply repetition/presence/frequency penalties to logits before sampling
            # 在采样前对 logits 应用各种惩罚机制
            adjusted_logits = last_logits
            if (
                (repetition_penalty is not None and repetition_penalty != 1.0) or
                (presence_penalty is not None and presence_penalty != 0.0) or
                (frequency_penalty is not None and frequency_penalty != 0.0)
            ):
                adjusted_logits = last_logits.clone()  # 克隆 logits，避免修改原始值
                vocab_size = adjusted_logits.size(-1)
                # Per-batch penalty application for clarity and correctness
                # 逐 batch 应用惩罚（每个样本的惩罚取决于其自身的历史 token）
                for b in range(batch_size):
                    seq_tokens = all_input_ids[b]  # 当前样本的所有已生成 token
                    if seq_tokens.numel() == 0:
                        continue
                    # 统计每个 token 在序列中出现的次数
                    counts = torch.bincount(seq_tokens, minlength=vocab_size)
                    if counts.dtype != torch.float32 and counts.dtype != torch.float64:
                        counts = counts.to(adjusted_logits.dtype)
                    # Presence penalty: penalize any token that has appeared
                    # 存在惩罚：只要 token 出现过，就统一减去 presence_penalty
                    if presence_penalty and presence_penalty != 0.0:
                        presence_mask = counts > 0
                        if presence_mask.any():
                            adjusted_logits[b, presence_mask] = adjusted_logits[b, presence_mask] - presence_penalty
                    # Frequency penalty: penalize proportionally to frequency
                    # 频率惩罚：按出现次数比例减去 frequency_penalty * count
                    if frequency_penalty and frequency_penalty != 0.0:
                        adjusted_logits[b] = adjusted_logits[b] - frequency_penalty * counts
                    # Repetition penalty (HF-style): divide positive logits, multiply negative logits
                    # 重复惩罚（HuggingFace 风格）：
                    # 对正 logits 除以 repetition_penalty（使其变小），
                    # 对负 logits 乘以 repetition_penalty（使其更负）
                    if repetition_penalty and repetition_penalty != 1.0:
                        rep_mask = counts > 0
                        if rep_mask.any():
                            pos_mask = rep_mask & (adjusted_logits[b] > 0)
                            neg_mask = rep_mask & ~pos_mask
                            if pos_mask.any():
                                adjusted_logits[b, pos_mask] = adjusted_logits[b, pos_mask] / repetition_penalty
                            if neg_mask.any():
                                adjusted_logits[b, neg_mask] = adjusted_logits[b, neg_mask] * repetition_penalty

            # Sample next token / 采样下一个 token
            # sample_token 支持 greedy (temperature=0)、top-p、top-k 等多种策略
            next_token = sample_token(adjusted_logits, temperature=effective_temperature, top_p=top_p, top_k=top_k)
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor([next_token], device=all_input_ids.device, dtype=torch.long).repeat(batch_size)

            # Apply EOS logic / EOS 处理
            if eos_set is not None:
                just_finished = torch.zeros_like(finished)
                # 检查当前步是否有新生成的 EOS token
                for eid in eos_set:
                    just_finished |= (next_token == eid)
                finished = finished | just_finished  # 更新完成状态（OR 操作，一旦完成就不会恢复）
                if pad_token_id is not None:
                    # 已完成的序列，后续 token 替换为 pad_token（避免产生无意义输出）
                    next_token = torch.where(
                        finished,
                        torch.tensor(pad_token_id, device=next_token.device, dtype=next_token.dtype),
                        next_token,
                    )

            # Append sampled token / 将采样的 token 追加到序列中
            next_token_unsqueezed = next_token.unsqueeze(1)  # (B,) → (B, 1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)  # (B, seq_len+1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype),
                ],
                dim=1,
            )

            # Stream the new token if streamer provided
            if streamer is not None:
                streamer.put(next_token_unsqueezed)

            # Early stop if all sequences finished
            if eos_set is not None and torch.all(finished):
                break

            # Decode one step using cached states; pass base-stream tensors
            kv_cache_index = [torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(all_input_ids.device)]

            decode_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=next_token_unsqueezed,
                attention_mask=current_attention_mask,
                position_ids=None,
                past_key_values=current_past,
                use_cache=True,
                *args,
                **kwargs,
            )
            last_logits = decode_output.logits[:, -1, :]

        # End streaming if streamer provided
        if streamer is not None:
            streamer.end()

        # Return style compatible with HF generate
        if return_dict_in_generate:
            if GreedySearchDecoderOnlyOutput is not None and SampleDecoderOnlyOutput is not None:
                if do_sample:
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
                else:
                    return GreedySearchDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
            # Fallback to generic ModelOutput
            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        return all_input_ids