"""
Oracle 模型模块 —— C2C (Cache-to-Cache) 框架中的性能上界参考模型。
Oracle Model Module - Performance upper bound reference for C2C framework.

本模块实现了 OracleRosettaModel 类，它是 C2C 框架中用于评估 KV-Cache 投影
理论上界的参考模型。与标准 RosettaModel 的区别在于：

1. Oracle 模型使用「替换」策略而非「投影+融合」策略：
   - 标准 C2C：Sharer KV → Projector → 与 Receiver KV 融合
   - Oracle：Sharer KV → 直接替换/加权混合 Receiver KV

2. Oracle 模型在同一模型族内运行（如 Qwen3-0.6B + Qwen3-4B），
   因此 KV 维度天然兼容，不需要维度投影，只需学习权重。

3. 用途：作为 C2C 方法的性能天花板，用于评估投影器引入的额外开销。

核心类:
    - OracleRosettaModel: Oracle 版本的 RosettaModel，接口与标准 RosettaModel 相同
      - forward(): 前向传播，分为 prefill（序列长度>1）和 decode（序列长度=1）两个阶段
      - generate(): 自回归生成，手动实现 token-by-token 的解码循环
      - set_projector_config(): 配置 sharer→receiver 的层映射关系
      - save/load_projector_config(): 序列化/反序列化投影器配置

核心数据流（两阶段推理）:
    Stage 1 (Prefill, seqlen > 1):
      1. Receiver 模型处理 prompt → 产生 base KV-Cache
      2. Sharer 模型处理 prompt → 产生 source KV-Cache
      3. Projector 将 source KV 投影到 receiver 兼容的维度
      4. 用投影后的 KV 替换/覆盖 base KV-Cache 对应位置
    Stage 2 (Decode, seqlen == 1):
      仅使用 Receiver 模型 + 已融合的 KV-Cache 进行逐 token 生成

关键数据结构:
    - projector_dict: 三级嵌套字典，描述层间映射
      {target_model_idx: {source_model_idx: {target_layer_idx: [(source_layer_idx, projector_idx), ...]}}}
    - kv_cache_dict: 二级嵌套字典，缓存各模型的 KV-Cache
      {target_model_idx: {source_model_idx: DynamicCache}}
    - kv_cache_index: 列表，每个元素形状 (B, section_seq_len, 2)，指示源/目标模型索引

与其他模块的关系:
    - rosetta.model.wrapper.RosettaModel: 标准 C2C 模型（Oracle 是其特化版本）
    - rosetta.model.projector.Projector: 投影器基类（Oracle 可能使用简化版本）
    - rosetta.model.sampling.sample_token: Token 采样函数（支持 greedy/top-p/top-k）
    - 训练脚本使用 oracle_train.py / oracle_train_kvcache_mse.py
"""

# ======================== 标准库导入 ========================
from typing import List, Optional, Union  # 类型注解工具
import json  # 用于序列化/反序列化 projector_dict 配置

# ======================== PyTorch 核心 ========================
import torch
from torch import nn  # 神经网络基类，OracleRosettaModel 继承 nn.Module

# ======================== HuggingFace Transformers ========================
from transformers.cache_utils import Cache, DynamicCache  # KV-Cache 抽象基类和动态缓存实现
from transformers.modeling_utils import PreTrainedModel  # 预训练模型基类（如 Qwen3ForCausalLM）
from transformers.modeling_outputs import CausalLMOutputWithPast  # 因果语言模型输出结构体（含 logits + past_key_values）
from transformers.utils import ModelOutput  # 通用模型输出基类，用于 generate 的 fallback 返回

# ======================== C2C / Rosetta 内部模块 ========================
from rosetta.model.projector import Projector  # 投影器基类，负责将 sharer KV 投影到 receiver 维度
from rosetta.model.sampling import sample_token  # Token 采样函数，支持 greedy/top-p/top-k 等策略
from rosetta.model.wrapper import RosettaModel  # 标准 RosettaModel，用于复用其静态工具方法（如 _convert_dict_keys_to_ints）

# ======================== HuggingFace 生成输出类型（可选导入） ========================
# 尝试导入 HF generate 的输出类型，用于 generate() 返回兼容格式
# 如果版本不支持则降级为 None，后续用 ModelOutput 作为 fallback
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except Exception:
    GreedySearchDecoderOnlyOutput = None  # Greedy 搜索输出类型不可用
    SampleDecoderOnlyOutput = None  # 采样搜索输出类型不可用

class OracleRosettaModel(nn.Module):
    """
    Oracle 版本的 RosettaModel —— C2C 性能上界参考模型。
    Drop-in replacement for the standard transformers LLM models, like Qwen3ForCausalLM.

    与标准 RosettaModel 的区别：
    - 使用替换/加权混合策略而非维度投影
    - 适用于同一模型族（KV 维度天然兼容）
    - 用于评估 KV-Cache 通信的理论性能上限

    Args:
        model_list: 模型列表，index 0 = base/receiver, index 1+ = sharer/teacher
        base_model_idx: base 模型在 model_list 中的索引，默认 0
        projector_list: 投影器列表（Oracle 版本可能使用简化的 ReplaceProjector）
    """
    def __init__(self, model_list: List[PreTrainedModel], base_model_idx = 0, projector_list: List[Projector] = []):
        """
        初始化 OracleRosettaModel。

        参数:
            model_list: 预训练模型列表。
                - index 0 (默认): base/receiver 模型，负责最终生成回答
                - index 1+: sharer/teacher 模型，提供辅助 KV-Cache
            base_model_idx: base 模型在 model_list 中的索引，默认为 0
            projector_list: 投影器列表。在 Oracle 场景中，投影器可能是
                ReplaceProjector（直接替换）或简单的加权混合投影器。
                每个 Projector 接收 (source_kv, target_kv) 并输出投影后的 (key, value)。

        关键属性:
            projector_dict: 三级嵌套字典，记录「哪一层的 source KV 用哪个投影器投影到哪一层」
                结构: {target_model_idx: {source_model_idx: {target_layer_idx: [(source_layer_idx, projector_idx)]}}}
            kv_cache_dict: 二级嵌套字典，存储各模型各层的 KV-Cache
                结构: {target_model_idx: {source_model_idx: DynamicCache}}
        """
        super().__init__()
        # model list: a list of model, model 0 by default is the base model
        # projector list: a list of projector
        # standard init with additional model list parameter
        # kv-cache dict: key (source_model_idx, target_model_idx), value (Cache), assume only convert at prefill with one type of model
        # projector dict: key (source_model_idx, target_model_idx) value dict(key (source_model_layer_idx, M_target value )

        # 记录 base/receiver 模型的索引，后续 decode 阶段只使用 base 模型
        self.base_model_idx = base_model_idx
        # 将所有模型包装为 ModuleList，使其自动注册为子模块（支持 .parameters() 等）
        self.model_list = nn.ModuleList(model_list)

        # 获取 base 模型的设备（如 cuda:0）和数据类型（如 float16），用于统一投影器的设备
        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype
        # 将投影器列表转为 ModuleList 并移动到与 base 模型相同的设备/精度
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)

        # projector_dict: 存储层间映射配置，由 set_projector_config() 填充
        # 例如: {0: {1: {5: [(4, 0)]}}} 表示
        #   模型 0 的层 5 ← 模型 1 的层 4 经过投影器 0 投影
        self.projector_dict = {}
        # kv_cache_dict: 存储各模型的 KV-Cache，在 forward() 的 prefill 阶段动态填充
        # 每次 forward() 调用都会重置此字典
        self.kv_cache_dict = {}
        # 用于注册 generation 钩子的句柄列表（当前未使用，保留扩展接口）
        self._generation_hook_handlers = []

    @property
    def device(self):
        """返回 base 模型所在的设备（如 cuda:0），方便外部获取当前模型的设备信息。"""
        return self.model_list[self.base_model_idx].device
    
    def to(self, device):
        """
        将整个 OracleRosettaModel（包括所有子模型和投影器）移动到指定设备。
        Move the RosettaModel and all underlying models and projectors to the specified device.

        参数:
            device: 目标设备，如 'cuda:0', 'cpu' 等

        返回:
            self: 返回自身以支持链式调用
        """
        super().to(device)
        # 逐个移动每个 LLM 模型到目标设备
        for model in self.model_list:
            model.to(device)
        # 逐个移动每个投影器到目标设备
        for projector in self.projector_list:
            projector.to(device)
        return self
        
    # set projector 
    # 设置投影器配置：定义 sharer 层 → receiver 层的 KV-Cache 映射关系
    def set_projector_config(self, 
                        source_model_idx: int, 
                        source_model_layer_idx: int, 
                        target_model_idx: int,
                        target_model_layer_idx: int, 
                        projector_idx: int):
        """
        设置投影器配置，建立「源模型某层 → 目标模型某层」的 KV-Cache 映射。
        Set the projector configuration.

        这是 C2C 的核心配置：指定 sharer 的哪一层 KV 通过哪个投影器
        投影到 receiver 的哪一层。支持多对多映射（同一目标层可接收多个源层的 KV）。

        参数:
            source_model_idx: 源模型（sharer）在 model_list 中的索引
            source_model_layer_idx: 源模型中的 Transformer 层索引
            target_model_idx: 目标模型（receiver）在 model_list 中的索引
            target_model_layer_idx: 目标模型中的 Transformer 层索引
            projector_idx: 投影器在 projector_list 中的索引

        projector_dict 结构示例:
            {
                0: {                        # target_model_idx = 0 (receiver)
                    1: {                    # source_model_idx = 1 (sharer)
                        5: [                # target_model_layer_idx = 5
                            (4, 0),         # sharer 层 4 → 投影器 0 → receiver 层 5
                            (5, 1),         # sharer 层 5 → 投影器 1 → receiver 层 5（可叠加）
                        ],
                        6: [(5, 0)],        # sharer 层 5 → 投影器 0 → receiver 层 6
                    }
                }
            }

        注意:
            对同一 (target_model_idx, source_model_idx, target_layer_idx) 的
            重复调用会追加新的 (source_layer, projector_idx) 对，实现多源融合。
            Repeated calls for the same (target, source, target_layer) append additional pairs.
        """

        # 第一步：确保 target_model_idx 这一级字典已创建
        if target_model_idx not in self.projector_dict.keys():
            self.projector_dict[target_model_idx] = {}
        # 第二步：确保 source_model_idx 这一级字典已创建
        if source_model_idx not in self.projector_dict[target_model_idx].keys():
            self.projector_dict[target_model_idx][source_model_idx] = {}
        # Accumulate list of (source_layer, projector_idx) for this target layer
        # 第三步：获取目标层当前的映射列表（如果不存在则为 None）
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(target_model_layer_idx)
        if layer_entry is None:
            # 首次为该目标层设置映射，创建新列表
            self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx] = [(source_model_layer_idx, projector_idx)]
        else:
            # 已有映射，追加新的 (源层, 投影器) 对 —— 支持多源 KV 融合到同一目标层
            layer_entry.append((source_model_layer_idx, projector_idx))


    def load_projector(self, projector_list):
        """
        加载/替换投影器列表。
        参数:
            projector_list: Projector 对象列表，替换当前的 self.projector_list
        """
        self.projector_list: List[Projector] = projector_list

    def get_projector(self, 
                        source_model_idx, 
                        source_model_layer_idx, 
                        target_model_idx,
                        target_model_layer_idx):
        """
        根据源模型层和目标模型层查找对应的投影器。
        
        参数:
            source_model_idx: 源模型索引
            source_model_layer_idx: 源模型层索引
            target_model_idx: 目标模型索引
            target_model_layer_idx: 目标模型层索引

        返回:
            Projector: 对应的投影器实例

        查找逻辑:
            1. 从 projector_dict 中取出该目标层的所有 (source_layer, projector_idx) 对
            2. 优先匹配源层号完全一致的投影器
            3. 若无精确匹配，回退到列表中第一个投影器
        """
        # 从三级嵌套字典中查找映射列表
        pair_list = self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        # Prefer exact source layer match
        # 优先精确匹配：源层号 == source_model_layer_idx 的投影器
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        # Fallback: return the first projector
        # 回退：如果没有精确匹配，返回第一个配置的投影器
        return self.projector_list[pair_list[0][1]]

    @staticmethod
    def load_json(file_name):
        """
        从 JSON 文件加载配置。
        
        参数:
            file_name: JSON 文件路径
        返回:
            dict: 解析后的 Python 字典
        """
        with open(file_name, "r") as f:
            result = json.load(f)
        return result
    
    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        """
        递归地将字典中形似整数的字符串键转换回 int 类型。
        Recursively convert dictionary keys that look like integers back to int.
        This reverses json.dump's coercion of dict keys to strings.

        背景：json.dump() 会将所有字典键转为字符串，但 projector_dict 的键
        是模型/层索引（整数），加载时需要恢复为 int。

        参数:
            obj: 待转换的对象（可以是 dict、list 或标量）
        返回:
            转换后的同结构对象
        """
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                # 检查键是否为整数字符串（支持负数，如 "-1" 表示不使用 C2C）
                if isinstance(key, str) and key.lstrip('-').isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                # 递归处理嵌套结构（注意：这里调用了 RosettaModel 的静态方法）
                new_obj[new_key] = RosettaModel._convert_dict_keys_to_ints(value)
            return new_obj
        if isinstance(obj, list):
            # 对列表中的每个元素递归转换
            return [RosettaModel._convert_dict_keys_to_ints(v) for v in obj]
        # 标量直接返回
        return obj
    
    
    def save_projector_config(self, file_name):
        """
        将 projector_dict 序列化为 JSON 文件。
        
        参数:
            file_name: 输出文件路径（.json）
        
        注意：JSON 会将 int 键自动转为字符串，加载时需用 _convert_dict_keys_to_ints 恢复。
        """
        with open(file_name, "w") as f:
            json.dump(self.projector_dict, f)

    
    def load_projector_config(self, config_path):
        """
        从 JSON 文件加载 projector_dict 配置。
        
        参数:
            config_path: JSON 配置文件路径
        
        加载流程:
            1. 读取 JSON 文件
            2. 递归地将字符串键恢复为 int（因为 json.dump 会强制将键转为字符串）
        """
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            # 使用 RosettaModel 的静态方法恢复整数键
            self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)

    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        """
        设置指定 (source, target) 模型对的 KV-Cache。
        
        参数:
            source_model_idx: 源模型索引
            target_model_idx: 目标模型索引
            cache: KV-Cache 对象。如果为 None，则自动创建空的 DynamicCache。
        
        用途:
            在 prefill 之前手动初始化 KV-Cache 容器，或由 forward() 内部调用。
        """
        # 确保二级字典结构存在
        if target_model_idx not in self.kv_cache_dict.keys():
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            # Initialize with a DynamicCache instead of RosettaCache for now
            # 如果未提供缓存，创建空的 DynamicCache（HuggingFace 标准动态缓存）
            self.kv_cache_dict[target_model_idx][source_model_idx] = DynamicCache() # noqa, maybe we should use RosettaCache here
        else:
            # 使用外部提供的缓存对象
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

    def forward(
        self,
        kv_cache_index: Optional[List] = None,  # KV-Cache 索引列表，每个元素形状 (B, section_seq_len, 2)
                                                 # 其中 2 个值分别为 [source_model_idx, target_model_idx]
                                                 # source_model_idx == -1 表示该段不使用 C2C
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,  # 输入 token ID
                                                 # LongTensor: 所有模型共享同一输入
                                                 # List[LongTensor]: 每个模型有各自的输入
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,  # 注意力掩码，格式同 input_ids
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码 ID，形状 (B, seqlen)
        past_key_values: Optional[Cache] = None,  # 之前的 KV-Cache（用于 decode 阶段传入历史缓存）
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 预计算的 token 嵌入（可选，仅 decode 使用）
        labels: Optional[torch.LongTensor] = None,  # 训练标签，用于计算交叉熵损失
        use_cache: Optional[bool] = None,  # 是否返回 KV-Cache（推理时为 True）
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐层状态
        cache_position: Optional[torch.LongTensor] = None,  # 缓存位置（decode 时指定写入 KV-Cache 的位置）
        logits_to_keep: Union[int, torch.Tensor] = 0,  # 控制输出 logits 的范围
        # **kwargs: Unpack[KwargsForCausalLM],
        identifier = -1,  # 调试/日志标识符，用于保存 KV 缓存文件时的命名
        subject = None,   # 调试/日志用的主题名称（如数据集名）
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        前向传播 —— C2C Oracle 模型的核心方法。
        Forward pass.

        根据序列长度自动切换两个阶段:
        - Prefill 阶段 (seqlen > 1): 处理整个 prompt
            1. 按 kv_cache_index 将输入切分为多个 section（段）
            2. 对每个 section:
               a. Receiver (base) 模型前向传播 → 产生 base KV-Cache
               b. 所有 Sharer 模型前向传播 → 产生 source KV-Cache
               c. 对配置了投影的层: 提取 source KV → 通过 Projector → 替换 base KV 对应位置
            3. 返回融合后的 KV-Cache + logits + loss

        - Decode 阶段 (seqlen == 1): 逐 token 生成
            仅使用 base (receiver) 模型 + 已融合的 KV-Cache 进行单步前向传播

        参数:
            kv_cache_index: KV-Cache 索引列表。每个元素形状 (B, section_seq_len, 2)，
                指示该段中每个 token 位置的源/目标模型索引。
                source_model_idx == -1 表示该段不做 C2C（No Rosetta: (-1, 0)）。
            input_ids: 输入 token ID。LongTensor 时所有模型共享，List 时各自独立。
            identifier: 调试标识，用于保存 KV 文件命名
            subject: 调试主题，用于保存 KV 文件命名

        返回:
            CausalLMOutputWithPast: 包含 logits, past_key_values, loss 等字段

        KV-Cache 形状约定:
            DynamicCache[key_cache|value_cache][layer_idx]: (B, num_heads, seq_len, head_dim)
            其中 B=batch_size, num_heads=注意力头数, seq_len=序列长度, head_dim=每个头的维度
        """
        
        # noqa
        # 每次 forward 调用都重置 KV-Cache 字典，避免跨 batch 污染
        self.kv_cache_dict = dict()

        # Handle different input formats: if input_ids is a list, use per-model inputs
        # ========== 输入格式处理 ==========
        # input_ids 可以是 LongTensor（所有模型共享输入）或 List[LongTensor]（每个模型独立输入）
        if isinstance(input_ids, list):
            # Use list format: different input_ids and attention_mask for each model
            # List 格式：每个模型有各自的 input_ids 和 attention_mask
            # 取出 base 模型的输入用于获取序列长度和基础 logits
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            # Use tensor format: same input_ids and attention_mask for all models (backward compatibility)
            # Tensor 格式：所有模型共享同一输入（向后兼容）
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        # ========== Section 切分 ==========
        # kv_cache_index 将输入序列切分为多个 section（段），每段可以有不同的 KV-Cache 策略
        # 例如：段 0 使用 C2C（sharer 提供 KV），段 1 不使用 C2C（纯 receiver）
        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1

        # 计算每个 section 的长度和起始位置
        # section_lengths[i] = 第 i 段在序列方向上的 token 数
        section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)] if kv_cache_index is not None else [seqlen]
        # section_starts[i] = 第 i 段在整个序列中的起始 token 位置
        # 例如 section_lengths=[10, 20] → section_starts=[0, 10, 30]
        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)
        
        # curr_base_kv_cache: base (receiver) 模型的累积 KV-Cache，随 section 递增
        curr_base_kv_cache = past_key_values

        # ======================== Prefill 阶段 (seqlen > 1) ========================
        # 处理整个 prompt 序列，按 section 逐段执行
        if seqlen > 1:
            for i in range(num_sections):
                # 第 i 个 section 在序列中的起止位置
                start = section_starts[i]
                end = section_starts[i + 1]
                # 从 base 模型输入中切出当前 section 的子序列
                prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
                # 注意力掩码需要包含从 0 到 end 的所有历史 token（因为注意力会看到前面的 token）
                prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None
                prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None
                prefill_labels = labels[:, start:end] if labels is not None else None

                # ---- Step A: Receiver (base) 模型前向传播 ----
                # calculate target model kvcache
                # base 模型处理当前 section 的 token，累积到 curr_base_kv_cache 中
                # KV-Cache 形状: (B, num_heads, cumulated_seq_len, head_dim)
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

                # 将 base 模型的 KV-Cache 存入字典: kv_cache_dict[base][base] = DynamicCache
                # 用于后续 KV-Cache 索引引用
                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = output.past_key_values

                # 更新累积的 base KV-Cache，下一个 section 会在此基础上继续追加
                curr_base_kv_cache: DynamicCache = output.past_key_values
                
                # ---- Step B: 所有 Sharer 模型前向传播 ----
                # if i != num_sections - 1:
                # 遍历所有非 base 的模型（sharer/teacher 模型）
                for source_model_idx in range(1, len(self.model_list)):
                    # 初始化该 sharer 模型的 KV-Cache 槽位
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                    # Get model-specific input_ids and attention_mask
                    # 获取 sharer 模型专属的输入（可能和 base 不同，也可能相同）
                    if isinstance(input_ids, list):
                        # List 格式：每个模型有独立的 input_ids
                        source_input_ids = input_ids[source_model_idx]
                        source_attention_mask = attention_mask[source_model_idx] if attention_mask is not None else None
                        source_prefill_input_ids = source_input_ids[:, start:end] if source_input_ids is not None else None
                        source_prefill_attention_mask = source_attention_mask[:, :end] if source_attention_mask is not None else None
                    else:
                        # Backward compatibility: use same input for all models
                        # 向后兼容：所有模型共享相同输入
                        source_prefill_input_ids = prefill_input_ids
                        source_prefill_attention_mask = prefill_attention_mask

                    # Sharer 模型前向传播，产生 source KV-Cache
                    # 注意: past_key_values 传入该 sharer 之前的缓存，实现增量更新
                    curr_source_kv_cache = self.model_list[source_model_idx].forward(
                        input_ids=source_prefill_input_ids,
                        attention_mask=source_prefill_attention_mask,
                        position_ids=prefill_position_ids,
                        past_key_values=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                        use_cache=use_cache, 
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        *args,
                        **kwargs
                    ).past_key_values  # 只取 KV-Cache，不关心 sharer 的 logits
                    # 存储 sharer 的 KV-Cache，供后续投影使用
                    self.kv_cache_dict[self.base_model_idx][source_model_idx] = curr_source_kv_cache

                # ---- Step C: KV-Cache 投影与替换（Oracle 核心逻辑） ----
                # calculate source model kvcache and apply projections
                # 检查是否有配置从某个 source 模型到 base 模型的投影
                if self.base_model_idx in self.projector_dict:
                    # 从 kv_cache_index 中解析当前 section 的 source 模型索引
                    # kv_cache_index[i] 形状: (B, section_seq_len, 2)
                    # [0][0][0] 取 batch=0, position=0 的 source_model_idx
                    source_model_idx = kv_cache_index[i][0][0][0].item()  # Get the source model index from the kv_cache_index
                    # source_model_idx == -1 表示该段不使用 C2C，跳过投影
                    if source_model_idx != -1:
                        # 遍历配置了投影的每个目标层 (target_layer_idx)
                        # entry = [(source_layer_idx, projector_idx), ...]
                        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                            # ---- 提取 base KV-Cache 当前 section 的切片 ----
                            # KV-Cache 形状: (B, num_heads, total_seq_len, head_dim)
                            base_key_cache, base_value_cache = curr_base_kv_cache[target_layer_idx]
                            # 切片到当前 section 的 token 范围 [start:end]
                            # 形状: (B, num_heads, section_len, head_dim)
                            new_base_key_cache = base_key_cache[:, :, start:end, :]
                            new_base_value_cache = base_value_cache[:, :, start:end, :]
                            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

                            # pair_list = 当前目标层的所有 (源层, 投影器) 映射
                            pair_list = entry

                            # ---- 对每个映射执行投影 ----
                            projected_kv_list = []  # 收集所有投影后的 KV
                            source_kv_list = []      # 收集原始 source KV（用于调试/可视化）
                            for source_model_layer_idx, projector_idx in pair_list:
                                # 从 sharer 模型的 KV-Cache 中取出源层的 KV
                                # KV-Cache 形状: (B, num_heads, total_seq_len, head_dim)
                                source_key_cache, source_value_cache = self.kv_cache_dict[self.base_model_idx][source_model_idx][source_model_layer_idx]
                                # 同样切片到当前 section 的 token 范围
                                new_source_key_cache = source_key_cache[:, :, start:end, :]
                                new_source_value_cache = source_value_cache[:, :, start:end, :]
                                new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                                # 调用投影器的 forward: (source_kv, target_kv) → (projected_key, projected_value)
                                # 在 Oracle 场景中，Projector 可能是 ReplaceProjector（直接替换）
                                # 或加权混合（alpha * source + (1-alpha) * target）
                                projected_key, projected_value = self.projector_list[projector_idx].forward(
                                    new_source_kv_cache, # tuple of (key, value), each of shape (B, N, H, D)
                                    new_base_kv_cache
                                )
                                projected_kv_list.append((projected_key, projected_value))

                                # --------------
                                # save base and projected kv cache
                                # 调试用：将投影后和原始的 KV-Cache 保存到磁盘，用于离线分析
                                # 文件名格式: oracle/{projected|target}_kv/{subject}_{identifier}_{section_idx}.pt
                                torch.save((projected_key, projected_value), f"oracle/projected_kv/{subject}_{identifier}_{i}.pt")
                                torch.save(new_base_kv_cache, f"oracle/target_kv/{subject}_{identifier}_{i}.pt")
                                # --------------
                                source_kv_list.append(new_source_kv_cache)

                            # Use first projector result
                            # 取第一个投影器的结果作为聚合后的 KV（Oracle 版本通常只有一个投影器）
                            agg_key, agg_value = projected_kv_list[0]

                            # ---- 用投影后的 KV 替换 base KV-Cache 对应位置 ----
                            # 这是 Oracle 的核心操作：直接覆写 DynamicCache 中的 key/value
                            # 形状不变: (B, num_heads, section_len, head_dim) 写入 [start:end] 切片
                            # Update cache
                            curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                            curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                        
                        # 将修改后的 KV-Cache 写回 output 对象
                        output.past_key_values = curr_base_kv_cache
                                                                             
        # ======================== Decode 阶段 (seqlen == 1) ========================
        # use base model for decode phase
        # 自回归生成的每一步，仅使用 base (receiver) 模型 + 已融合的 KV-Cache
        # 此阶段不涉及 sharer 模型或投影操作，因为 KV-Cache 已在 Prefill 阶段准备好
        else:
            # Handle list input format for decode phase as well
            # 处理 List 格式的输入（取 base 模型的输入）
            decode_input_ids = input_ids[self.base_model_idx] if isinstance(input_ids, list) else input_ids
            decode_attention_mask = attention_mask[self.base_model_idx] if isinstance(attention_mask, list) else attention_mask
            
            # 仅用 base 模型做单步前向传播
            # past_key_values 包含 Prefill 阶段已融合的 KV-Cache + 之前 decode 步骤累积的 KV
            output = self.model_list[self.base_model_idx].forward(
                input_ids=decode_input_ids,       # 形状: (B, 1)，当前步的单个 token
                attention_mask=decode_attention_mask,  # 形状: (B, total_seq_len_so_far + 1)
                position_ids=position_ids,         # 位置编码
                past_key_values=curr_base_kv_cache,  # 累积的 KV-Cache
                inputs_embeds=inputs_embeds,       # 可选的预计算嵌入
                labels=labels,                     # 训练标签（如果有的话）
                use_cache=use_cache,               # True: 返回更新后的 KV-Cache
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,     # KV-Cache 写入位置
                *args,
                **kwargs
            )

        return output  # 返回 CausalLMOutputWithPast: logits + past_key_values + loss(如有 labels)

    @torch.no_grad()  # 生成阶段不需要梯度，节省内存和计算
    def generate(
        self,
        kv_cache_index,          # KV-Cache 索引列表，格式同 forward()
        input_ids,               # prompt token ID，LongTensor 或 List[LongTensor]
        max_new_tokens: Optional[int] = None,  # 最多生成的新 token 数
        past_key_values: Optional[Cache] = None,  # 可选的初始 KV-Cache
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,  # 位置编码
        eos_token_id: Optional[Union[int, List[int]]] = None,  # EOS token ID（支持多个）
        pad_token_id: Optional[int] = None,  # PAD token ID，用于填充已结束的序列
        temperature: float = 1.0,  # 采样温度，0.0 表示 greedy
        top_p: float = 1.0,       # nucleus 采样参数
        top_k: int = -1,          # top-k 采样参数，-1 表示不使用
        do_sample: Optional[bool] = None,  # 是否使用采样（None 时默认 False = greedy）
        return_dict_in_generate: Optional[bool] = None,  # 是否以字典格式返回
        output_scores: Optional[bool] = None,  # 是否返回每步的 logits 分数
        max_length: Optional[int] = None,  # 最大总长度（prompt + 生成）
        use_cache: bool = True,  # 是否使用 KV-Cache（生成时通常为 True）
        *args,
        **kwargs,
    ):
        """
        自回归生成 —— 手动实现的 token-by-token 解码循环。
        New generation loop without using the base model's generate.
        - Uses this module's forward for prefill and per-token decode.
        - Samples tokens via rosetta.model.sampling.sample_token.

        不使用 HuggingFace 内置的 generate()，而是手动实现循环，原因：
        - 需要在 Prefill 阶段执行 C2C 特有的 KV-Cache 投影/替换
        - 需要支持每个模型有独立 input_ids 的 List 输入格式

        生成流程:
        1. Prefill: 调用 self.forward(seqlen > 1) 处理整个 prompt，
           执行 C2C KV-Cache 融合，获取初始 logits
        2. Decode 循环: 逐 token 采样，每次调用 self.forward(seqlen == 1)
           获取下一步 logits，直到达到 max_new_tokens 或全部序列遇到 EOS

        参数:
            kv_cache_index: KV-Cache 索引，格式同 forward()
            input_ids: prompt token ID
            max_new_tokens: 最多生成的新 token 数
            temperature: 采样温度。0.0=greedy, >0=采样
            top_p: nucleus 采样，保留累计概率 >= top_p 的最小 token 集合
            top_k: top-k 采样，保留概率最高的 k 个 token
            do_sample: True=采样, False/None=greedy
            eos_token_id: 结束 token ID，支持 int 或 List[int]
            pad_token_id: 填充 token ID，已结束的序列用此 token 填充
            return_dict_in_generate: True 时返回 GreedySearchDecoderOnlyOutput / SampleDecoderOnlyOutput
            output_scores: True 时收集每步的 logits

        返回:
            - return_dict_in_generate=True 时: GreedySearchDecoderOnlyOutput 或 SampleDecoderOnlyOutput
              包含 sequences 形状 (B, prompt_len + generated_len) 和可选的 scores
            - return_dict_in_generate=False 时: LongTensor 形状 (B, prompt_len + generated_len)

        Returns a tensor of shape [batch, prompt_len + generated_len] for the base model stream.
        """
        # ======================== 参数预处理 ========================
        # Derive number of tokens to generate
        # If max_new_tokens not provided, infer from max_length
        # 获取 base 模型的 prompt 长度（用于计算 max_new_tokens）
        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)

        # Default eos/pad from base model tokenizer/config if not provided
        # 从 base 模型的 config 中获取默认的 EOS/PAD token ID
        base_model = self.model_list[self.base_model_idx]
        gen_cfg = getattr(base_model, "generation_config", None)
        cfg_obj = gen_cfg if gen_cfg is not None else getattr(base_model, "config", None)
        if eos_token_id is None and cfg_obj is not None:
            eos_token_id = getattr(cfg_obj, "eos_token_id", None)
        if pad_token_id is None and cfg_obj is not None:
            pad_token_id = getattr(cfg_obj, "pad_token_id", None)
        # 如果 PAD 未指定但 EOS 已指定，则用 EOS 作为 PAD（常见做法）
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]

        # 计算 max_new_tokens: 如果未显式指定，从 max_length - prompt_len 推导
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

        # Resolve base inputs
        # 解析 base 模型的输入 tensor
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        # 如果没有 attention_mask，默认全 1（所有 token 都可见）
        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(base_input_ids, dtype=torch.long, device=base_input_ids.device)

        batch_size = base_input_ids.size(0)

        # ======================== Stage 1: Prefill ========================
        # Prefill to build caches and obtain initial logits
        # 调用 forward() 处理整个 prompt，执行 C2C KV-Cache 融合
        # 此时 seqlen > 1，走 forward 的 Prefill 分支
        prefill_output = self.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,          # 可能包含所有模型的输入
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            *args,
            **kwargs,
        )

        # Prefill 后的 KV-Cache（已融合 C2C 信息）
        current_past = prefill_output.past_key_values
        # 累积的生成序列（初始为 prompt）
        all_input_ids = base_input_ids
        # 累积的 attention_mask
        current_attention_mask = base_attention_mask

        # ======================== EOS 处理准备 ========================
        # EOS handling setup
        # 将 eos_token_id 统一为 set，方便多 EOS 检测
        eos_set = None
        if eos_token_id is not None:
            eos_set = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        # finished: 布尔向量，标记每个 batch 是否已经生成 EOS
        finished = torch.zeros(batch_size, dtype=torch.bool, device=all_input_ids.device)

        # Start from last prefill logits
        # 取 Prefill 最后一个 token 的 logits 作为首次采样的输入
        # 形状: (B, vocab_size)
        last_logits = prefill_output.logits[:, -1, :]

        # Determine sampling mode
        # 确定采样模式：do_sample=None 默认为 greedy
        if do_sample is None:
            do_sample = False
        # temperature=0.0 时表示 greedy（sample_token 内部处理此特殊情况）
        effective_temperature = temperature if do_sample else 0.0

        # Optional scores collection
        # 如果需要收集每步的 logits（用于分析），初始化 scores 列表
        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        # ======================== Stage 2: Decode 循环 ========================
        # 逐 token 自回归生成，最多 max_new_tokens 步
        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)
            # ---- 采样下一个 token ----
            # sample_token 支持 greedy（temperature=0）、top-p、top-k 等策略
            # last_logits 形状: (B, vocab_size) → next_token 形状: (B,)
            next_token = sample_token(last_logits, temperature=effective_temperature, top_p=top_p, top_k=top_k)
            # 确保 next_token 是 tensor（sample_token 可能返回标量）
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor([next_token], device=all_input_ids.device, dtype=torch.long).repeat(batch_size)

            # ---- EOS 检测与处理 ----
            # Apply EOS logic
            if eos_set is not None:
                # just_finished: 当前步刚刚产生 EOS 的 batch
                just_finished = torch.zeros_like(finished)
                for eid in eos_set:
                    just_finished |= (next_token == eid)
                # 累积：一旦 finished 就永远 finished
                finished = finished | just_finished
                # 对已结束的序列，将生成的 token 替换为 pad_token
                # 保证后续 attention 不受已结束序列的干扰
                if pad_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.tensor(pad_token_id, device=next_token.device, dtype=next_token.dtype),
                        next_token,
                    )

            # ---- 将新 token 拼接到序列中 ----
            # Append sampled token
            next_token_unsqueezed = next_token.unsqueeze(1)  # (B,) → (B, 1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)  # 沿序列维度拼接
            # 扩展 attention_mask，新 token 位置设为 1（可见）
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype),
                ],
                dim=1,
            )

            # Early stop if all sequences finished
            # 如果 batch 中所有序列都已生成 EOS，提前终止
            if eos_set is not None and torch.all(finished):
                break

            # ---- Decode 单步前向传播 ----
            # Decode one step using cached states; pass base-stream tensors
            # 构造 decode 阶段的 kv_cache_index：(-1, 0) 表示不使用 C2C（纯 receiver 模型）
            # 形状: (1, 1, 2) → [batch_dim=1, seq_dim=1, (source=-1, target=0)]
            kv_cache_index = [torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(all_input_ids.device)]

            # 调用 forward()，此时 seqlen == 1，走 forward 的 Decode 分支
            decode_output = self.forward(
                kv_cache_index=kv_cache_index,       # (-1, 0) 不做 C2C
                input_ids=next_token_unsqueezed,     # (B, 1) 单个新 token
                attention_mask=current_attention_mask,  # (B, total_len)
                position_ids=None,                    # 由模型自动计算
                past_key_values=current_past,         # 累积的 KV-Cache
                use_cache=True,                       # 返回更新后的 KV-Cache
                *args,
                **kwargs,
            )
            # 更新 KV-Cache 和 logits
            current_past = decode_output.past_key_values  # 累积的 KV-Cache
            last_logits = decode_output.logits[:, -1, :]  # 形状: (B, vocab_size)，用于下一步采样

        # ======================== 返回值构造 ========================
        # Return style compatible with HF generate
        # 构造与 HuggingFace generate() 兼容的返回格式
        if return_dict_in_generate:
            # 如果 HF 的专用输出类型可用，使用它们
            if GreedySearchDecoderOnlyOutput is not None and SampleDecoderOnlyOutput is not None:
                if do_sample:
                    # 采样模式：返回 SampleDecoderOnlyOutput
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,  # (B, prompt_len + generated_len)
                        scores=scores if collect_scores else None,  # List[Tensor(B, vocab_size)]
                    )
                else:
                    # Greedy 模式：返回 GreedySearchDecoderOnlyOutput
                    return GreedySearchDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
            # Fallback to generic ModelOutput
            # 降级：使用通用的 ModelOutput（当 HF 版本不支持专用类型时）
            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        # 默认直接返回 token 序列 tensor
        return all_input_ids  # LongTensor, 形状 (B, prompt_len + generated_len)