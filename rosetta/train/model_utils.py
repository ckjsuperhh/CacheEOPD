"""
Model setup utilities for RosettaModel training/evaluation.
模型构建工具函数模块，用于 RosettaModel 的训练与评估。

本模块作用:
    提供 C2C (Cache-to-Cache) 框架中模型初始化与层映射策略的核心工具函数。
    主要包含两大类功能:
    1. 层映射策略 (Mapping Strategies): 定义 sharer (teacher) 模型与 receiver (base) 模型
       之间的 Transformer 层对应关系，解决两个模型层数不同时的 KV-Cache 对齐问题。
    2. 模型构建 (setup_models): 加载 base 模型、teacher 模型、创建 Projector，
       组装为 RosettaModel 并配置层映射。

核心函数:
    - k_nearest_sources: 基于归一化位置距离的 K 近邻层映射策略
    - last_aligned_sources: 尾部对齐后向前展开的层映射策略
    - setup_models: 一站式模型初始化入口

与其他模块的关系:
    - rosetta.model.wrapper.RosettaModel: 本模块构建并返回的包装模型对象
    - rosetta.model.projector.create_projector: 本模块调用的 Projector 工厂函数
    - rosetta.train.*: 训练流程调用 setup_models 获取初始化好的模型
"""

# PyTorch 深度学习框架
import torch
from typing import Dict, Any, List  # 类型注解工具
# HuggingFace Transformers: 用于加载预训练的因果语言模型和分词器
from transformers import AutoModelForCausalLM, AutoTokenizer

# RosettaModel: C2C 框架的核心包装模型，统一管理多个 LLM 和 Projector
from rosetta.model.wrapper import RosettaModel
# create_projector: Projector 工厂函数，根据类型创建不同的投影器（用于 KV 维度转换）
from rosetta.model.projector import create_projector

"""
Mapping strategies
层映射策略: 定义 teacher (sharer) 模型的层如何映射到 base (receiver) 模型的层。
在 C2C 框架中，当 sharer 和 receiver 的 Transformer 层数不同时，需要确定
哪些 teacher 层的 KV-Cache 会被投影并融合到哪些 base 层。
"""


def k_nearest_sources(num_target_layers: int, num_source_layers: int, k: int) -> Dict[int, List[int]]:
    """
    Compute a per-target mapping to K nearest source layers.
    基于归一化位置距离计算每个目标层最近的 K 个源层。

    算法说明:
        将 target 层和 source 层分别均匀分布在 [0, 1] 区间上，
        例如 target 有 4 层 → 位置为 [0.0, 0.333, 0.667, 1.0]。
        对每个 target 层，按绝对距离排序所有 source 层，取最近的 K 个。
        这种策略确保即使两个模型层数差异很大，也能建立合理的层对应关系。

    在 C2C 中的作用:
        target = base (receiver) 模型的层
        source = teacher (sharer) 模型的层
        映射结果决定: 对 base 模型的第 i 层，应该从 teacher 模型的哪些层获取 KV-Cache

    Args:
        num_target_layers: 目标模型 (receiver/base) 的 Transformer 层数
        num_source_layers: 源模型 (sharer/teacher) 的 Transformer 层数
        k: 为每个目标层选择的最近源层数量

    Returns:
        Dict[int, List[int]]: 映射字典，key 为目标层索引，value 为对应的源层索引列表。
        仅包含有可映射源层的目标层。

    示例:
        k_nearest_sources(4, 8, 2) 可能返回:
        {0: [0, 1], 1: [2, 3], 2: [5, 4], 3: [7, 6]}
    """
    # 将 target 层均匀映射到 [0, 1] 区间的位置坐标
    if num_target_layers <= 1:
        # 只有一层时，位置固定为 0.0
        target_positions = [0.0]
    else:
        # 多层时等距分布: 第 i 层位置 = i / (层数 - 1)
        target_positions = [i / (num_target_layers - 1) for i in range(num_target_layers)]
    # 将 source 层同样均匀映射到 [0, 1] 区间
    if num_source_layers <= 1:
        source_positions = [0.0]
    else:
        source_positions = [j / (num_source_layers - 1) for j in range(num_source_layers)]

    # 对每个 target 层，找到距离最近的 K 个 source 层
    mapping: Dict[int, List[int]] = {}
    for t_idx, t_pos in enumerate(target_positions):
        # 按位置绝对距离对所有 source 层排序
        sorted_src = sorted(range(num_source_layers), key=lambda j: abs(source_positions[j] - t_pos))
        # 取前 K 个最近的源层
        chosen = sorted_src[:max(0, k)]
        if len(chosen) > 0:
            mapping[t_idx] = chosen
    return mapping


def last_aligned_sources(num_target_layers: int, num_source_layers: int, k: int = 1) -> Dict[int, List[int]]:
    """
    Return a per-target mapping that aligns the last target layer to the last
    source layer and walks toward the front.
    尾部对齐映射策略：将 target 的最后一层与 source 的最后一层对齐，然后向前展开。

    算法说明:
        与 k_nearest_sources 的归一化距离不同，此策略采用"尾部对齐"方式:
        1. 计算 offset = num_source_layers - num_target_layers
        2. target 的第 t 层对应 source 的第 (offset + t) 层（即锚点）
        3. 从锚点开始，优先向后方（更浅层）取 K 个源层
        4. 若后方不够 K 个，则向前方（更深层）补充

    在 C2C 中的意义:
        深层 Transformer 中，靠近输出的层（尾部）通常包含更高级的语义信息。
        尾部对齐策略优先保证深层的精确对应，适合层数差异不大或 receiver
        比 sharer 层数少的场景。

    Args:
        num_target_layers: 目标模型 (receiver/base) 的 Transformer 层数
        num_source_layers: 源模型 (sharer/teacher) 的 Transformer 层数
        k: 为每个目标层选择的源层数量，默认为 1

    Returns:
        Dict[int, List[int]]: 映射字典，key 为目标层索引，value 为对应的源层索引列表。

    示例 (T=11, S=33):
        target 10 -> [32, 31, ...]  (最后一层精确对齐)
        target 9  -> [31, 30, ...]  (倒数第二层对齐)
    """
    mapping: Dict[int, List[int]] = {}
    if num_target_layers <= 0 or num_source_layers <= 0:
        return mapping

    # 计算尾部对齐的偏移量: offset >= 0 表示 source 模型前面多出的层数
    # 例如: source 32 层, target 11 层 → offset = 21
    # target 第 0 层 → source 第 21 层, target 第 10 层 → source 第 31 层 (最后一层)
    offset = num_source_layers - num_target_layers

    def take_k_from(s0: int) -> List[int]:
        """
        从锚点 s0 开始选择 K 个源层。
        优先向后（浅层方向）选取，不足时向前（深层方向）补充。
        
        Args:
            s0: source 层的锚点索引
        Returns:
            选中的源层索引列表
        """
        result: List[int] = []
        # 优先从锚点向后（向浅层方向）移动选取
        for back in range(k):
            idx = s0 - back
            if 0 <= idx < num_source_layers:
                result.append(idx)
        # 如果因为边界限制（已到达第 0 层）还不够 K 个，则向前（深层方向）扩展
        next_idx = s0 + 1
        while len(result) < k and next_idx < num_source_layers:
            result.append(next_idx)
            next_idx += 1
        return result

    for t in range(num_target_layers):
        # 计算当前 target 层在 source 模型中的锚点位置
        s0 = offset + t
        # 边界保护: 将锚点限制在 source 层的有效范围内
        if s0 < 0:
            s0 = 0
        elif s0 > num_source_layers - 1:
            s0 = num_source_layers - 1
        # 从锚点位置选取 K 个源层
        chosen = take_k_from(s0)
        if len(chosen) > 0:
            mapping[t] = chosen

    return mapping


def setup_models(model_config: Dict[str, Any], device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """
    Setup RosettaModel with base model, teacher model, and projectors.
    一站式构建 C2C 框架所需的全部模型组件。

    流程概述:
        1. 加载 tokenizer 并配置 pad_token（用于序列填充对齐）
        2. 加载 base 模型 (receiver, 接收 KV-Cache 的一方)
        3. 加载 teacher 模型 (sharer, 提供 KV-Cache 的一方)
        4. 根据配置创建 Projector (将 teacher 的 KV 维度投影到 base 的维度)
        5. 将 base 模型、teacher 模型、projector 组装为 RosettaModel
        6. 配置层映射关系 (1:1 逐层对应，取两者层数的较小值)

    Args:
        model_config: 配置字典，包含以下键:
            - base_model: base/receiver 模型的 HuggingFace 模型名称或路径
            - teacher_model: teacher/sharer 模型的 HuggingFace 模型名称或路径
            - projector: Projector 配置字典，包含:
                - type: Projector 类型 (如 "linear", "mlp" 等)
                - params: Projector 的额外参数 (维度、层数等)
        device: 模型加载到的设备，默认 "cuda"
        dtype: 模型的数据类型，默认 torch.bfloat16（节省显存）

    Returns:
        tuple: (rosetta_model, tokenizer)
            - rosetta_model: 组装好的 RosettaModel 实例
            - tokenizer: 加载好的分词器
    """
    
    # ========== 第一步: 加载分词器 ==========
    tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"])
    if tokenizer.pad_token is None:
        # 某些模型（如 GPT-2/Llama）没有预设 pad_token，用 eos_token 替代
        # 这样在 batch 填充时不会对 loss 计算产生影响
        tokenizer.pad_token = tokenizer.eos_token
    
    # ========== 第二步: 加载 base 模型 (receiver) ==========
    # base 模型是接收 KV-Cache 的一方，通常是较小的模型
    # 使用 device_map=device 将模型放到指定设备上
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=dtype,      # 使用 bfloat16 减少显存占用
        device_map=device       # 指定设备
    )
    
    # ========== 第三步: 加载 teacher 模型 (sharer) ==========
    # teacher 模型是提供 KV-Cache 的一方，通常是较大的模型
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_config["teacher_model"],
        torch_dtype=dtype,
        device_map=device
    )
    
    # ========== 第四步: 创建 Projector (维度投影器) ==========
    # 当 teacher 和 base 的 head_dim 不同时，需要 Projector 进行维度转换
    # 例如: teacher head_dim=128, base head_dim=64 → Projector 做 128→64 的投影
    projector_config = model_config["projector"]
    projector_params = projector_config["params"].copy()
    projector_params["dtype"] = dtype  # 保持与模型一致的精度
    
    # 通过工厂函数创建 Projector，支持不同类型 (linear, mlp, etc.)
    projector = create_projector(
        projector_config["type"],          # Projector 类型
        source_dim=teacher_model.config.head_dim,  # teacher 的 KV head 维度
        target_dim=base_model.config.head_dim,     # base 的 KV head 维度
        **projector_params                  # 额外参数（如隐藏层维度等）
    )

    # ========== 第五步: 组装 RosettaModel ==========
    # RosettaModel 是 C2C 的核心包装器，统一管理:
    # - model_list: [base_model, teacher_model] (索引 0=base, 1=teacher)
    # - base_model_idx: 指定哪个是 base 模型
    # - projector_list: 所有 Projector 的列表
    rosetta_model = RosettaModel(
        model_list=[base_model, teacher_model],
        base_model_idx=0,           # 索引 0 对应 base (receiver) 模型
        projector_list=[projector]  # 只有一个 Projector
    ).to(device)
    
    # ========== 第六步: 配置层映射关系 ==========
    # 将 teacher 的每一层映射到 base 的对应层
    # 取两者层数的较小值，确保映射不越界
    # 例如: base 11 层, teacher 32 层 → 只映射前 11 层
    num_layers_to_map = min(
        base_model.config.num_hidden_layers, 
        teacher_model.config.num_hidden_layers
    )
    
    for layer_idx in range(num_layers_to_map):
        # 建立 1:1 的逐层映射: teacher 第 i 层 → base 第 i 层
        # 这是一种简单的映射策略；更复杂的映射可使用上面的
        # k_nearest_sources 或 last_aligned_sources 函数
        rosetta_model.set_projector_config(
            source_model_idx=1,              # source = teacher (索引 1)
            source_model_layer_idx=layer_idx, # teacher 的层索引
            target_model_idx=0,              # target = base (索引 0)
            target_model_layer_idx=layer_idx, # base 的层索引
            projector_idx=0                  # 使用第一个（唯一的）Projector
        )
    
    return rosetta_model, tokenizer 