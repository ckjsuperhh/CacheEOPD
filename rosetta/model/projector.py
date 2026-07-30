"""
Projector nn module for the unified memory
===========================================================================
投影器（Projector）神经网络模块 —— C2C (Cache-to-Cache) 框架的核心组件之一。

模块作用:
    在 C2C 框架中，Sharer 模型产生的 KV-Cache 需要被「投影」到 Receiver 模型兼容的
    维度空间，然后与 Receiver 自身的 KV-Cache 进行「融合」。本模块实现了这一过程
    所需的全部投影与融合逻辑。

核心类:
    - Projector          : 投影器基类，定义 forward() 和 cache_project() 接口
    - AllInOneProjector  : 统一投影器，支持多种门控（gate）、权重（weight）粒度配置，
                           可选 Gumbel-Sigmoid、SwiGLU、残差连接等现代模式
    - C2CProjector       : 面向 C2C 的精简投影器，使用共享隐层同时产生投影特征与权重
    - ModernMLP          : 带残差/LayerNorm/Dropout 的现代 MLP
    - SwiGLUBlock        : SwiGLU 激活块
    - QwenStyleLayer     : Qwen3 风格的 SwiGLU + RMSNorm 子层
    - StandardFFNLayer   : 经典前馈网络子层（RMSNorm + 激活 + 残差）
    - RegularMLP         : 由 StandardFFNLayer 堆叠而成的 MLP

与其他模块的关系:
    - rosetta.utils.registry : 提供 @register_model / @capture_init_args 装饰器，
                               以及 save_object / load_object 序列化接口
    - 被 Fuser 模块调用      : Fuser 使用 Projector 将 sharer KV 投影到 receiver 空间
    - transformers.Cache    : DynamicCache 是 KV-Cache 的标准容器

两阶段推理中的位置:
    Stage 1 — Sharer 处理 prompt 产生 KV-Cache
    Stage 2 — Projector 把 Sharer KV 投影到 Receiver 空间，Fuser 完成融合，
              Receiver 基于融合后的 KV-Cache 生成回答
"""

# ── 标准库与第三方依赖 ──────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch import Tensor
from transformers import Cache, DynamicCache  # HuggingFace KV-Cache 容器
from typing import Optional, Tuple, Literal, Union
import copy
import math

# ── Rosetta 内部工具 ────────────────────────────────────────────────────────
# register_model    : 将类注册到 PROJECTOR_REGISTRY 字典中，供工厂函数按名称查找
# capture_init_args : 保存 __init__ 调用参数，方便序列化/反序列化时重建实例
# get_projector_class: 根据字符串名称获取已注册的投影器类
# save_object/load_object: 基于 pickle 的通用序列化/反序列化工具
from rosetta.utils.registry import register_model, get_projector_class, PROJECTOR_REGISTRY, capture_init_args, save_object, load_object

class Projector(nn.Module):
    """
    Base projector class for unified memory
    ========================================
    投影器基类 —— 定义将 Sharer KV 投影到 Receiver KV 空间的标准接口。

    所有具体投影器（如 AllInOneProjector、C2CProjector）都必须继承此类并实现
    forward() 方法。基类还提供了 cache_project() 方法，用于逐层遍历
    DynamicCache 并调用 forward() 完成整个 KV-Cache 的投影。

    子类需实现:
        forward(source_kv, target_kv) -> (projected_key, projected_value)
    """

    def forward(self, source_kv: Tuple[Tensor, Tensor], target_kv: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Project and combine the source key-value tensors to the target key-value tensors.
        将 source (Sharer) 的 KV 张量投影并与 target (Receiver) 的 KV 张量融合。

        Args:
            source_kv: Tuple of (key, value) tensors, each (..., D_s) where ... are arbitrary leading dimensions
                       Sharer 侧的 (key, value)，最后一维为 source 模型的头维度 D_s
            target_kv: Tuple of (key, value) tensors, each (..., D_t) where ... are arbitrary leading dimensions
                       Receiver 侧的 (key, value)，最后一维为 target 模型的头维度 D_t
        Returns:
            Tuple of (key, value) tensors, each (..., D_t) with same leading dimensions as input
            融合后的 (key, value)，形状与 target_kv 一致，即 (..., D_t)
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def cache_project(self, source_kv_cache: Cache, target_kv_cache: Cache) -> Cache:
        """
        Project the source kv cache to the target kv cache.
        逐层遍历 source_kv_cache，调用 self.forward() 进行投影，返回新的 DynamicCache。

        这是 cache_project 的默认实现，适用于 DynamicCache。它会把每一层的 KV 从
        DynamicCache 格式 (B, H, N, D) 转换为 projector 期望的 (B, N, H, D)，调用
        forward()，再转回 (B, H, N, D) 存入新的 DynamicCache。

        Args:
            source_kv_cache: Sharer 模型输出的 KV-Cache (DynamicCache)
            target_kv_cache: Receiver 模型当前的 KV-Cache (DynamicCache)
        Returns:
            projected_cache: 投影/融合后的新 DynamicCache
        """
        if not isinstance(source_kv_cache, DynamicCache) or not isinstance(target_kv_cache, DynamicCache):
            raise ValueError("Only DynamicCache is supported")

        # 创建空的 DynamicCache 用于存放投影结果
        projected_cache = DynamicCache()

        # 逐层处理 —— source_kv_cache 和 target_kv_cache 按层索引对齐
        # Process each layer
        for layer_idx in range(len(source_kv_cache.key_cache)):
            # DynamicCache 存储格式: (B=batch, H=num_heads, N=seq_len, D=head_dim)
            source_key = source_kv_cache.key_cache[layer_idx]    # (B, H_s, N, D_s)
            source_value = source_kv_cache.value_cache[layer_idx]  # (B, H_s, N, D_s)

            # Get corresponding target tensors (for reference/combination)
            # 获取 target 侧对应层的 KV；若 target 层数不够则创建零张量
            if layer_idx < len(target_kv_cache.key_cache):
                target_key = target_kv_cache.key_cache[layer_idx]    # (B, H_t, N, D_t)
                target_value = target_kv_cache.value_cache[layer_idx]  # (B, H_t, N, D_t)
            else:
                # If target cache doesn't have this layer, create dummy tensors
                # target 模型层数不足时，创建同形状零张量作为占位
                B, H, N, D_s = source_key.shape
                D_t = source_key.shape[-1]  # Assume same dimension for simplicity / 简化起见假设维度相同
                target_key = torch.zeros(B, H, N, D_t, device=source_key.device, dtype=source_key.dtype)
                target_value = torch.zeros(B, H, N, D_t, device=source_value.device, dtype=source_value.dtype)

            # Reshape for forward pass: DynamicCache format (B, H, N, D) -> projector format (B, N, H, D)
            # 维度转置: DynamicCache 是 (B, H, N, D)，投影器内部约定 (B, N, H, D)
            source_key_reshaped = source_key.transpose(1, 2)    # (B, N, H_s, D_s)
            source_value_reshaped = source_value.transpose(1, 2)  # (B, N, H_s, D_s)
            target_key_reshaped = target_key.transpose(1, 2)    # (B, N, H_t, D_t)
            target_value_reshaped = target_value.transpose(1, 2)  # (B, N, H_t, D_t)

            # Project using forward method with tuple input/output
            # 调用子类实现的 forward() 进行投影+融合
            source_kv = (source_key_reshaped, source_value_reshaped)
            target_kv = (target_key_reshaped, target_value_reshaped)
            projected_key, projected_value = self.forward(source_kv, target_kv)

            # Reshape back: projector format (B, N, H, D) -> DynamicCache format (B, H, N, D)
            # 转回 DynamicCache 存储格式 (B, H, N, D)
            projected_key = projected_key.transpose(1, 2)    # (B, H_t, N, D_t)
            projected_value = projected_value.transpose(1, 2)  # (B, H_t, N, D_t)

            # Update cache / 将投影结果写入新 cache 的对应层
            projected_cache.update(projected_key, projected_value, layer_idx)

        return projected_cache

class ModernMLP(nn.Module):
    """
    Modern MLP with residual connections, layer normalization, and configurable architecture.
    现代多层感知机 —— 支持残差连接、LayerNorm、多种激活函数（GELU/ReLU/SiLU）以及 SwiGLU。

    用于 AllInOneProjector 内部的投影网络和门控/权重生成网络。

    架构特点:
        - 可选 SwiGLU 激活（中间层使用 SwiGLUBlock 替代 Linear + activation）
        - 每层之间可选插入 LayerNorm、激活、Dropout
        - 最后一层不加 LayerNorm / 激活 / Dropout
        - 当 input_dim == output_dim 时自动启用残差连接；否则通过线性投影对齐

    Args:
        input_dim      : 输入特征维度
        output_dim     : 输出特征维度
        hidden_dim     : 隐藏层维度（默认 512）
        num_layers     : 线性层数量（含输入/输出层）
        activation     : 激活函数名称 "gelu" / "relu" / "silu"
        use_layer_norm : 是否在中间层后插入 LayerNorm
        use_residual   : 是否启用残差连接（仅在 input_dim == output_dim 时生效）
        dropout        : Dropout 概率
        use_swiglu     : 中间层是否使用 SwiGLU 替代普通 Linear + activation
        dtype          : 张量数据类型
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 仅当输入输出维度相同时才启用残差连接
        self.use_residual = use_residual and (input_dim == output_dim)
        self.use_swiglu = use_swiglu

        # 根据字符串选择激活函数
        # Activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 逐层构建网络 —— 第一层接受 input_dim，最后一层输出 output_dim
        # Build layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            layer_output_dim = output_dim if i == num_layers - 1 else hidden_dim

            if self.use_swiglu and i < num_layers - 1:  # Don't use SwiGLU on output layer / 输出层不使用 SwiGLU
                layer = SwiGLUBlock(layer_input_dim, layer_output_dim, dtype=dtype)
            else:
                layer = nn.Linear(layer_input_dim, layer_output_dim, dtype=dtype)

            self.layers.append(layer)

            # Add layer norm after each layer except the last one
            # 在除最后一层之外的每层后面插入 LayerNorm
            if use_layer_norm and i < num_layers - 1:
                self.layers.append(nn.LayerNorm(layer_output_dim, dtype=dtype))

            # Add activation after each layer except the last one
            # 在除最后一层之外的每层后面插入激活函数（SwiGLU 已内含激活，跳过）
            if i < num_layers - 1 and not self.use_swiglu:
                self.layers.append(copy.deepcopy(self.activation))

            # Add dropout after activation / 在激活函数后插入 Dropout
            if dropout > 0 and i < num_layers - 1:
                self.layers.append(nn.Dropout(dropout))

        # Residual projection if dimensions don't match
        # 若残差连接开启但维度不匹配，用线性投影对齐（实际上 use_residual 已限制 input==output）
        if self.use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        else:
            self.residual_proj = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with optional residual connection.
        前向传播，可选残差连接。

        流程: x → 各层顺序变换 → 若 use_residual 则加上原始输入 (或线性投影后的输入)

        Args:
            x: 输入张量，形状 (..., input_dim)
        Returns:
            输出张量，形状 (..., output_dim)
        """
        residual = x  # 保存输入用于残差

        for layer in self.layers:
            x = layer(x)

        # Add residual connection / 残差连接: output = transform(x) + residual
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)  # 维度对齐
            x = x + residual

        return x


class SwiGLUBlock(nn.Module):
    """
    SwiGLU activation block for modern transformer architectures.
    SwiGLU 激活块 —— 现代 Transformer（如 LLaMA / Qwen）常用的门控激活结构。

    计算公式:
        output = SiLU(gate_proj(x)) * up_proj(x)
        即：先通过两个独立的线性投影分别得到 gate 和 up，gate 经过 SiLU 激活后
        与 up 逐元素相乘，实现门控效果。

    Args:
        input_dim  : 输入维度
        output_dim : 输出维度
        dtype      : 数据类型
    """

    def __init__(self, input_dim: int, output_dim: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, output_dim, dtype=dtype)  # 门控投影
        self.up_proj = nn.Linear(input_dim, output_dim, dtype=dtype)    # 值投影
        self.activation = nn.SiLU()  # SiLU 也叫 Swish

    def forward(self, x: Tensor) -> Tensor:
        gate = self.activation(self.gate_proj(x))  # SiLU 激活后的门控信号
        up = self.up_proj(x)                       # 值信号
        return gate * up                           # 逐元素相乘，实现门控


@register_model   # 注册到 PROJECTOR_REGISTRY，使工厂函数可通过名称字符串创建实例
@capture_init_args # 保存 __init__ 参数，便于序列化/反序列化
class AllInOneProjector(Projector):
    """
    Unified projector that consolidates all projection functionalities with modern patterns.
    统一投影器 —— 整合了所有投影功能的高度可配置实现。

    这是 C2C 框架中最通用、功能最全的投影器，支持以下特性：

    Features / 特性:
    1. Gate logit granularity 门控粒度: scalar（标量）, token（逐 token）,
       head（逐注意力头）, head_merged（头合并后）, value（逐元素）
    2. Key/Value weight granularity 权重粒度: 同上
    3. Input-dependent gates and weights 输入相关: 可通过 MLP 根据 target/projected KV 动态生成
    4. Optional concatenation with combiner networks 拼接+组合网络
    5. Modern MLP architecture with residual connections and SwiGLU
    6. Configurable target preservation 目标保留策略:
       - preserve_target_weight=True  : output = (1-weight)*target + gate*weight*projected
       - preserve_target_weight=False : output = target + gate*weight*projected
    7. Optional adding of target (self) signal to outputs via add_self

    融合公式（核心）:
        output = target_term + gate * normalized_weight * projected
        其中 target_term 取决于 add_self 和 preserve_target_weight 配置

    Args:
        source_dim         : Sharer 模型每注意力头的维度 D_s
        target_dim         : Receiver 模型每注意力头的维度 D_t
        source_num_heads   : Sharer 模型的注意力头数 H_s
        target_num_heads   : Receiver 模型的注意力头数 H_t
        hidden_dim         : 投影 MLP 的隐藏层维度
        num_layers         : 投影 MLP 的层数
        dropout            : Dropout 概率
        activation         : 激活函数名称
        use_layer_norm     : 投影 MLP 是否使用 LayerNorm
        use_residual       : 投影 MLP 是否使用残差连接
        use_swiglu         : 投影 MLP 是否使用 SwiGLU
        gate_granularity   : 门控粒度 ("scalar"/"token"/"head"/"head_merged"/"value")
        gate_depends_on_input: 门控是否为输入相关的（True=MLP 动态生成, False=可学习参数）
        gate_input_features  : 门控 MLP 的输入特征选择
        gate_init_value      : 门控参数初始值
        weight_granularity   : 权重粒度
        weight_depends_on_input: 权重是否为输入相关的
        weight_input_features  : 权重 MLP 的输入特征选择
        weight_init_value      : 权重参数初始值
        preserve_target_weight: 是否在 target 项上乘 (1-normalized_weight)
        add_self             : 是否将 target (self) 加到输出中
        use_concat           : 是否将 projected 和 target 拼接后再通过 combiner
        weight_hidden_dim    : 权重生成 MLP 的隐藏层维度
        use_gumbel           : 训练时是否使用 Gumbel-Sigmoid
        initial_temperature  : 门控温度退火的初始值
        final_temperature    : 门控温度退火的终止值
        anneal_steps         : 温度退火总步数
        scalar_temperature   : 权重 sigmoid 归一化时的温度
        max_sequence_length  : token 级参数的最大序列长度
        pos_emb              : 是否使用位置编码（保留参数）
        dtype                : 数据类型
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        use_swiglu: bool = False,

        # Gate configuration / 门控配置
        gate_granularity: Literal["scalar", "token", "head", "head_merged", "value"] = "scalar",
        gate_depends_on_input: bool = False,
        gate_input_features: Optional[str] = "target_key",  # "target_key", "target_value", "both", "target_projected_key", "target_projected_value", "target_projected_both"
        gate_init_value: float = 0.0,

        # Weight configuration / 权重配置
        weight_granularity: Literal["scalar", "token", "head", "head_merged", "value"] = "scalar",
        weight_depends_on_input: bool = False,
        weight_input_features: Optional[str] = "target_key",  # "target_key", "target_value", "both", "target_projected_key", "target_projected_value", "target_projected_both"
        weight_init_value: float = 0.0,

        # Target preservation configuration / 目标保留配置
        preserve_target_weight: bool = True,  # If False, target won't be multiplied by (1 - normalized_weight) / 若为 False，target 不乘 (1-w)
        add_self: bool = True,  # If False, target (self) won't be added to outputs / 若为 False，输出中不含 target 自身

        # Concat configuration / 拼接组合配置
        use_concat: bool = False,
        # combiner_hidden_dim: int = 128,
        weight_hidden_dim: int = 1024,

        # Temperature and gumbel / 温度与 Gumbel 配置
        use_gumbel: bool = True,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        anneal_steps: int = 1360,
        scalar_temperature: float = 0.005,

        # Sequence length configuration / 序列长度配置
        max_sequence_length: int = 8192,  # Maximum sequence length for token-level parameters / token 级参数的最大序列长度

        pos_emb: bool = False,

        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        # ── 保存基本维度信息 ──
        self.source_dim = source_dim         # Sharer 每头维度 D_s
        self.target_dim = target_dim         # Receiver 每头维度 D_t
        self.source_num_heads = source_num_heads  # Sharer 头数 H_s
        self.target_num_heads = target_num_heads  # Receiver 头数 H_t
        self.hidden_dim = hidden_dim
        self.weight_hidden_dim = weight_hidden_dim
        self.max_sequence_length = max_sequence_length

        # ── 保存配置项 ──
        # Configuration
        self.gate_granularity = gate_granularity
        self.gate_depends_on_input = gate_depends_on_input
        self.gate_input_features = gate_input_features
        self.weight_granularity = weight_granularity
        self.weight_depends_on_input = weight_depends_on_input
        self.weight_input_features = weight_input_features
        self.preserve_target_weight = preserve_target_weight
        self.add_self = add_self
        self.use_concat = use_concat
        self.use_gumbel = use_gumbel
        self.scalar_temperature = scalar_temperature

        # ── 温度退火：训练时从高到低退火，使 Gumbel-Sigmoid 逐渐趋近硬门控 ──
        # Temperature annealing for gate
        self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps

        # ── 构建 Key/Value 投影 MLP ──
        # 输入维度 = source_dim * source_num_heads（将多头展平为一个大向量）
        # 输出维度 = target_dim * target_num_heads
        # Build projection networks
        self.key_projection = self._build_projection_mlp(
            source_dim * source_num_heads,  # H_s * D_s
            target_dim * target_num_heads,  # H_t * D_t
            hidden_dim, num_layers, activation, use_layer_norm,
            use_residual, dropout, use_swiglu, dtype
        )
        self.value_projection = self._build_projection_mlp(
            source_dim * source_num_heads,  # H_s * D_s
            target_dim * target_num_heads,  # H_t * D_t
            hidden_dim, num_layers, activation, use_layer_norm,
            use_residual, dropout, use_swiglu, dtype
        )

        # ── 构建门控（Gate）组件 ──
        # Build gate components
        self._build_gate_components(dtype)

        # ── 构建权重（Weight）组件 ──
        # Build weight components
        self._build_weight_components(weight_init_value, dtype)

        # ── 构建拼接组合器（Combiner）──
        # 当 use_concat=True 时，先拼接 [projected, target] 再通过线性层降维
        # Build concat components if needed
        if self.use_concat:
            in_dim = target_dim * target_num_heads * 2   # 拼接后维度 = 2 * H_t * D_t
            out_dim = target_dim * target_num_heads       # 输出维度 = H_t * D_t
            self.key_combiner = nn.Linear(in_dim, out_dim, dtype=dtype)    # Key 组合器
            self.value_combiner = nn.Linear(in_dim, out_dim, dtype=dtype)  # Value 组合器
        
    def _build_projection_mlp(
        self, input_dim: int, output_dim: int, hidden_dim: int,
        num_layers: int, activation: str, use_layer_norm: bool,
        use_residual: bool, dropout: float, use_swiglu: bool, dtype: torch.dtype
    ) -> ModernMLP:
        """
        Build modern MLP for projection.
        构建用于 Key/Value 维度投影的现代 MLP。

        将 Sharer 的展平多头向量 (H_s * D_s) 映射到 Receiver 的展平多头向量 (H_t * D_t)。
        """
        return ModernMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
            use_residual=use_residual,
            dropout=dropout,
            use_swiglu=use_swiglu,
            dtype=dtype
        )

    def _build_gate_components(self, dtype: torch.dtype):
        """
        Build gate logit components based on configuration.
        根据配置构建门控（gate）组件。

        门控决定「是否启用」投影后的 KV 信号：
        - 若 gate_depends_on_input=False: 门控为可学习的静态参数（nn.Parameter）
        - 若 gate_depends_on_input=True:  门控由 MLP 根据输入特征动态生成

        门控粒度（granularity）:
        - scalar  : 全局一个标量
        - token   : 每个 token 一个值
        - head    : 每个注意力头一个值
        - head_merged: 合并头部后每个 token 一个值
        - value   : 每个元素一个值
        """
        if not self.gate_depends_on_input:
            # Parameter-based gate / 基于参数的静态门控
            gate_shape = self._get_parameter_shape(self.gate_granularity)
            self.gate_logit = nn.Parameter(torch.zeros(gate_shape, dtype=dtype))
        else:
            # Input-dependent gate via MLP / 基于 MLP 的输入相关门控
            input_dim = self._get_gate_input_dim()
            output_dim = self._get_gate_output_dim()

            self.gate_generator = ModernMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=self.hidden_dim,
                num_layers=2,
                activation="gelu",
                use_layer_norm=True,
                use_residual=False,
                dropout=0.1,
                dtype=dtype
            )
    
    def _build_weight_components(self, weight_init_value: float, dtype: torch.dtype):
        """
        Build weight components based on configuration.
        根据配置构建权重（weight）组件。

        权重控制投影后 KV 信号的「融合强度」，决定 source 信息在最终输出中的占比：
        - 若 weight_depends_on_input=False: 权重为可学习的静态参数
        - 若 weight_depends_on_input=True:  权重由共享隐层 MLP + 两个独立头动态生成

        权重最终会经过 sigmoid 归一化到 [0, 1] 范围，用于加权 projected 和 target 的贡献。
        """
        if not self.weight_depends_on_input:
            # Parameter-based weights / 基于参数的静态权重
            weight_shape = self._get_parameter_shape(self.weight_granularity)
            self.key_weight = nn.Parameter(torch.full(weight_shape, weight_init_value, dtype=dtype))
            self.value_weight = nn.Parameter(torch.full(weight_shape, weight_init_value, dtype=dtype))
        else:
            # Input-dependent weights via MLP / 基于 MLP 的输入相关权重
            input_dim = self._get_weight_input_dim()
            output_dim = self._get_weight_output_dim()

            # Shared hidden layer for efficiency / 共享隐层以提高计算效率
            self.weight_hidden = ModernMLP(
                input_dim=input_dim,
                output_dim=self.weight_hidden_dim,
                hidden_dim=self.weight_hidden_dim,
                num_layers=2,
                activation="gelu",
                use_layer_norm=True,
                use_residual=False,
                dropout=0.1,
                dtype=dtype
            )

            # Separate heads for key and value weights / Key 和 Value 各有独立的权重输出头
            self.key_weight_head = nn.Linear(self.weight_hidden_dim, output_dim, dtype=dtype)
            self.value_weight_head = nn.Linear(self.weight_hidden_dim, output_dim, dtype=dtype)
    
    def _get_parameter_shape(self, granularity: str) -> tuple:
        """
        Get parameter shape based on granularity.
        根据粒度获取静态参数的形状。

        返回值说明:
        - scalar      → ()         标量，全局共享
        - token       → (N,)       每个 token 一个参数
        - head        → (N, H)     每个 token 每个注意力头一个参数
        - head_merged → (N, H)     同 head，但处理方式略有不同
        - value       → (N, H, D)  每个 token 每个头每个维度一个参数
        """
        if granularity == "scalar":
            return ()  # Scalar / 标量参数
        elif granularity == "token":
            return (self.max_sequence_length,)  # Token-level parameters with max sequence length / 每个 token 位置一个参数
        elif granularity == "head":
            return (self.max_sequence_length, self.target_num_heads)  # Token and head level parameters / 每个 token 每个头一个参数
        elif granularity == "head_merged":
            return (self.max_sequence_length, self.target_num_heads)  # Token and head level parameters / 与 head 相同形状
        elif granularity == "value":
            return (self.max_sequence_length, self.target_num_heads, self.target_dim)  # Token, head and value level parameters / 最细粒度
        else:
            raise ValueError(f"Invalid granularity: {granularity}")

    def _get_gate_input_dim(self) -> int:
        """
        Get input dimension for gate generator.
        获取门控生成网络的输入维度。

        根据 gate_input_features 决定基础维度 base_dim，
        再根据 gate_granularity 决定是否乘以头数。
        """
        base_dim = 0
        if self.gate_input_features == "target_key":
            base_dim = self.target_dim  # 仅使用 target 的 key 特征，维度为 D_t
        elif self.gate_input_features == "target_value":
            base_dim = self.target_dim  # 仅使用 target 的 value 特征，维度为 D_t
        elif self.gate_input_features == "both":
            base_dim = self.target_dim * 2  # 拼接 target 的 key 和 value，维度为 2*D_t
        elif self.gate_input_features == "target_projected_key":
            base_dim = self.target_dim * 2  # target_key + projected_key / target key 与投影后 key 拼接
        elif self.gate_input_features == "target_projected_value":
            base_dim = self.target_dim * 2  # target_value + projected_value / target value 与投影后 value 拼接
        elif self.gate_input_features == "target_projected_both":
            base_dim = self.target_dim * 4  # target_key + target_value + projected_key + projected_value / 四者拼接
        else:
            raise ValueError(f"Invalid gate input features: {self.gate_input_features}")

        # Adjust for granularity processing strategy
        # 根据粒度调整输入维度
        if self.gate_granularity == "scalar":
            # Scalar: process aggregated features across all heads / 标量粒度：池化所有头和 token 的特征
            return base_dim
        elif self.gate_granularity == "token":
            # Token: process merged head dimensions / Token 粒度：将所有头展平
            return base_dim * self.target_num_heads  # Flatten (H, D) to (H*D)
        elif self.gate_granularity == "head_merged":
            # Head-merged: similar to token granularity, merge H and D / Head 合并粒度：类似 token，展平头维度
            return base_dim * self.target_num_heads  # (B, N, H*D)
        elif self.gate_granularity == "head":
            # Head-local: per head processing, do not merge heads / Head 粒度：保持每个头独立处理
            return base_dim  # (B, H, N, D)
        else:  # value
            # Value: process per-head features / Value 粒度：同样保持每个头独立处理
            return base_dim

    def _get_gate_output_dim(self) -> int:
        """
        Get output dimension for gate generator.
        获取门控生成网络的输出维度。

        根据粒度决定输出维度:
        - scalar      → 1（标量输出）
        - token       → 1（每个 token 一个标量）
        - head_merged → H（每个 token 输出 H 个头的值）
        - head        → 1（每个头一个标量）
        - value       → D（每个头输出 D 个维度的值）
        """
        if self.gate_granularity == "scalar":
            return 1
        elif self.gate_granularity == "token":
            return 1  # Per token / 每个 token 一个标量
        elif self.gate_granularity == "head_merged":
            # Per token per head after merge: output one value per head / 每个 token 输出 H 个头的值
            return self.target_num_heads
        elif self.gate_granularity == "head":
            # Per token per head: scalar per head / 每个头一个标量
            return 1
        elif self.gate_granularity == "value":
            return self.target_dim  # Per token per head per value / 每个头输出 D 维
        else:
            raise ValueError(f"Invalid gate granularity: {self.gate_granularity}")

    def _get_weight_input_dim(self) -> int:
        """
        Get input dimension for weight generator.
        获取权重生成的输入维度，逻辑与 _get_gate_input_dim 对称。
        """
        base_dim = 0
        if self.weight_input_features == "target_key":
            base_dim = self.target_dim
        elif self.weight_input_features == "target_value":
            base_dim = self.target_dim
        elif self.weight_input_features == "both":
            base_dim = self.target_dim * 2
        elif self.weight_input_features == "target_projected_key":
            base_dim = self.target_dim * 2  # target_key + projected_key / target key 与投影后 key 拼接
        elif self.weight_input_features == "target_projected_value":
            base_dim = self.target_dim * 2  # target_value + projected_value / target value 与投影后 value 拼接
        elif self.weight_input_features == "target_projected_both":
            base_dim = self.target_dim * 4  # target_key + target_value + projected_key + projected_value / 四者拼接
        else:
            raise ValueError(f"Invalid weight input features: {self.weight_input_features}")

        # Adjust for granularity processing strategy / 根据粒度调整输入维度
        if self.weight_granularity == "scalar":
            # Scalar: process aggregated features across all heads / 标量粒度：池化
            return base_dim
        elif self.weight_granularity == "token":
            # Token: process merged head dimensions / Token 粒度：展平所有头
            return base_dim * self.target_num_heads
        elif self.weight_granularity == "head_merged":
            # Head-merged: similar to token granularity / Head 合并粒度
            return base_dim * self.target_num_heads
        elif self.weight_granularity == "head":
            # Head-local: per head processing / Head 粒度：保持独立
            return base_dim
        else:  # value
            # Value: process per-head features / Value 粒度：保持独立
            return base_dim

    def _get_weight_output_dim(self) -> int:
        """
        Get output dimension for weight generator.
        获取权重生成的输出维度，逻辑与 _get_gate_output_dim 对称。
        """
        if self.weight_granularity == "scalar":
            return 1
        elif self.weight_granularity == "token":
            return 1  # Per token / 每个 token 一个标量
        elif self.weight_granularity == "head_merged":
            # Per token per head after merge / 每个 token 输出 H 个头
            return self.target_num_heads
        elif self.weight_granularity == "head":
            # Per token per head: scalar per head / 每个头一个标量
            return 1
        elif self.weight_granularity == "value":
            return self.target_dim  # Per token per head per value / 每个头输出 D 维
        else:
            raise ValueError(f"Invalid weight granularity: {self.weight_granularity}")
    
    def _generate_gates(self, target_key: Tensor, target_value: Tensor, projected_key: Tensor = None, projected_value: Tensor = None) -> Tensor:
        """
        Generate gate logits based on configuration.
        根据配置生成门控 logits。

        门控决定投影后的 KV 是否被启用（0/1 或 0~1 的软值）。
        - 静态模式: 直接返回可学习参数 self.gate_logit
        - 动态模式: 根据输入特征通过 MLP 生成门控

        Args:
            target_key      : Receiver 的 key，形状 (B, H_t, N, D_t)
            target_value    : Receiver 的 value，形状 (B, H_t, N, D_t)
            projected_key   : 投影后的 key（可选，部分配置需要）
            projected_value : 投影后的 value（可选，部分配置需要）
        Returns:
            gate_logit: 门控 logits，后续通过 sigmoid 转为 [0,1] 范围
        """
        if not self.gate_depends_on_input:
            # 静态参数门控：直接返回可学习参数
            # Use parameter-based gate
            return self.gate_logit
        else:
            # 输入相关门控：根据配置的输入特征构建门控
            # Generate input-dependent gate
            # First, prepare the base input features / 第一步：准备基础输入特征
            if self.gate_input_features == "target_key":
                base_input = target_key
            elif self.gate_input_features == "target_value":
                base_input = target_value
            elif self.gate_input_features == "both":
                base_input = torch.cat([target_key, target_value], dim=-1)  # 拼接 key 和 value
            elif self.gate_input_features == "target_projected_key":
                if projected_key is None:
                    raise ValueError("projected_key is required for target_projected_key input features")
                base_input = torch.cat([target_key, projected_key], dim=-1)  # 拼接原始 key 与投影后 key
            elif self.gate_input_features == "target_projected_value":
                if projected_value is None:
                    raise ValueError("projected_value is required for target_projected_value input features")
                base_input = torch.cat([target_value, projected_value], dim=-1)  # 拼接原始 value 与投影后 value
            elif self.gate_input_features == "target_projected_both":
                if projected_key is None or projected_value is None:
                    raise ValueError("Both projected_key and projected_value are required for target_projected_both input features")
                base_input = torch.cat([target_key, target_value, projected_key, projected_value], dim=-1)  # 拼接四者

            # Now process based on granularity / 第二步：根据粒度处理张量形状
            # base_input shape: (B, H, N, D_input)
            B, H, N, D_input = base_input.shape

            if self.gate_granularity == "scalar":
                # 标量粒度：对所有头和 token 取平均 → (B, D_input)
                # For scalar granularity, aggregate all dimensions: (B, H, N, D_input) -> (B, D_input)
                gate_input = base_input.mean(dim=(1, 2))  # Average over heads and tokens
            elif self.gate_granularity == "token":
                # Token 粒度：转置并展平所有头维度 → (B, N, H*D_input)
                # For token granularity, merge H and D_input dimensions: (B, H, N, D_input) -> (B, N, H*D_input)
                gate_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.gate_granularity == "head_merged":
                # Head 合并粒度：同 token 粒度，转置并展平 → (B, N, H*D_input)
                # For head granularity, merge H and D like token: (B, H, N, D_in) -> (B, N, H*D_in)
                gate_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.gate_granularity == "head":
                # Head 粒度：保持原始形状 (B, H, N, D_input)，逐头独立处理
                # For head granularity, keep per-head processing: (B, H, N, D_input)
                gate_input = base_input
            elif self.gate_granularity == "value":
                # Value 粒度：同样保持原始形状 (B, H, N, D_input)
                # For value granularity, keep per-head processing: (B, H, N, D_input)
                gate_input = base_input

            # 通过门控 MLP 生成 logits
            return self.gate_generator(gate_input)
    
    def _generate_weights(self, target_key: Tensor, target_value: Tensor, projected_key: Tensor = None, projected_value: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Generate weights based on configuration.
        根据配置生成融合权重。

        权重控制 projected 信号与 target 信号的混合比例：
        - 静态模式: 返回可学习参数 self.key_weight, self.value_weight
        - 动态模式: 通过共享隐层 MLP + 两个独立输出头生成 key_weight 和 value_weight

        Args:
            target_key      : Receiver 的 key，形状 (B, H_t, N, D_t)
            target_value    : Receiver 的 value，形状 (B, H_t, N, D_t)
            projected_key   : 投影后的 key（可选）
            projected_value : 投影后的 value（可选）
        Returns:
            (key_weight, value_weight): 融合权重张量对
        """
        if not self.weight_depends_on_input:
            # 静态参数权重：直接返回可学习参数
            # Use parameter-based weights
            return self.key_weight, self.value_weight
        else:
            # 输入相关权重：根据配置的输入特征动态生成
            # Generate input-dependent weights
            # First, prepare the base input features / 第一步：准备基础输入特征
            if self.weight_input_features == "target_key":
                base_input = target_key
            elif self.weight_input_features == "target_value":
                base_input = target_value
            elif self.weight_input_features == "both":
                base_input = torch.cat([target_key, target_value], dim=-1)  # 拼接 key 和 value
            elif self.weight_input_features == "target_projected_key":
                if projected_key is None:
                    raise ValueError("projected_key is required for target_projected_key input features")
                base_input = torch.cat([target_key, projected_key], dim=-1)  # 拼接原始 key 与投影后 key
            elif self.weight_input_features == "target_projected_value":
                if projected_value is None:
                    raise ValueError("projected_value is required for target_projected_value input features")
                base_input = torch.cat([target_value, projected_value], dim=-1)  # 拼接原始 value 与投影后 value
            elif self.weight_input_features == "target_projected_both":
                if projected_key is None or projected_value is None:
                    raise ValueError("Both projected_key and projected_value are required for target_projected_both input features")
                base_input = torch.cat([target_key, target_value, projected_key, projected_value], dim=-1)  # 拼接四者

            # Now process based on granularity / 第二步：根据粒度处理张量形状
            # base_input shape: (B, H, N, D_input)
            B, H, N, D_input = base_input.shape

            if self.weight_granularity == "scalar":
                # 标量粒度：对所有头和 token 取平均 → (B, D_input)
                weight_input = base_input.mean(dim=(1, 2))  # Average over heads and tokens
            elif self.weight_granularity == "token":
                # Token 粒度：转置并展平所有头维度 → (B, N, H*D_input)
                weight_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.weight_granularity == "head_merged":
                # Head 合并粒度：转置并展平 → (B, N, H*D_input)
                weight_input = base_input.transpose(1, 2).contiguous().view(B, N, H * D_input)
            elif self.weight_granularity == "head":
                # Head 粒度：保持原始形状 (B, H, N, D_input)
                weight_input = base_input
            elif self.weight_granularity == "value":
                # Value 粒度：保持原始形状 (B, H, N, D_input)
                weight_input = base_input

            # 通过共享隐层 + 独立输出头生成 key 和 value 权重
            weight_hidden = self.weight_hidden(weight_input)          # 共享特征提取
            key_weight = self.key_weight_head(weight_hidden)          # Key 权重输出头
            value_weight = self.value_weight_head(weight_hidden)      # Value 权重输出头

            return key_weight, value_weight
    
    def _apply_gumbel_sigmoid(self, gate_logit: Tensor) -> Tensor:
        """
        Apply Gumbel sigmoid trick for training.
        应用 Gumbel-Sigmoid 技巧，训练时对门控施加随机噪声以实现可微的离散门控。

        训练时: gate = sigmoid((logit + gumbel_noise) / temperature)
            - temperature 从高到低退火，使输出逐渐趋近 0/1 的离散值
        推理时: gate = (logit > 0).float()
            - 直接硬门控：logit > 0 则为 1，否则为 0

        Args:
            gate_logit: 门控 logits 张量
        Returns:
            gate: [0,1] 范围内的门控值
        """
        if self.training and self.use_gumbel:
            gumbel_noise = self._sample_gumbel(gate_logit.shape, gate_logit.device, gate_logit.dtype)
            # Gumbel-Sigmoid: 加噪后除以温度再 sigmoid，温度越低越接近硬门控
            return torch.sigmoid((gate_logit + gumbel_noise) / self.gate_temperature)
        else:
            # 推理时使用硬门控（不可微但确定性）
            return (gate_logit > 0).float()

    @staticmethod
    def _sample_gumbel(shape: tuple, device: torch.device, dtype: torch.dtype, eps: float = 1e-20) -> Tensor:
        """
        Sample from Gumbel distribution.
        从 Gumbel 分布采样，用于 Gumbel-Sigmoid 技巧。

        公式: G = -log(-log(U)), 其中 U ~ Uniform(0,1)
        eps 用于防止 log(0)。

        Args:
            shape : 采样张量的形状
            device: 目标设备
            dtype : 数据类型
            eps   : 防止 log(0) 的小量
        Returns:
            采样结果张量
        """
        u = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(u + eps) + eps)

    def _reshape_for_granularity(self, tensor: Tensor, granularity: str, target_shape: tuple) -> Tensor:
        """
        Reshape tensor to match target shape based on granularity.
        将静态参数张量按粒度广播到目标形状 (B, H, N, D)。

        用于将可学习参数（如标量、token 级向量等）扩展为与 KV 张量同形的权重张量。

        Args:
            tensor      : 原始参数张量
            granularity : 粒度类型
            target_shape: 目标形状 (B, H_t, N, D_t)
        Returns:
            广播后的张量，形状为 target_shape
        """
        B, H, N, D = target_shape  # B=batch, H=头数, N=序列长度, D=头维度

        if granularity == "scalar":
            # 标量 → 广播到所有维度 (B, H, N, D)
            # Scalar -> (B, H, N, D)
            return tensor.view(1, 1, 1, 1).expand(B, H, N, D)
        elif granularity == "token":
            # (max_seq_len,) → 截取前 N 个 token → 广播到 (B, H, N, D)
            # (max_seq_len,) -> (B, H, N, D) - slice to actual sequence length
            token_params = tensor[:N]  # Take first N tokens / 截取前 N 个 token
            return token_params.view(1, 1, N, 1).expand(B, H, N, D)
        elif granularity == "head":
            # (max_seq_len, H) → 截取前 N 个 token → 转置为 (1, H, N, 1) → 广播
            # (max_seq_len, H) -> (B, H, N, D) - slice to actual sequence length, each token each head independent
            head_params = tensor[:N, :]  # Take first N tokens, all heads: (N, H)
            return head_params.view(1, N, H, 1).transpose(1, 2).expand(B, H, N, D)  # (1, N, H, 1) -> (1, H, N, 1) -> (B, H, N, D)
        elif granularity == "head_merged":
            raise NotImplementedError  # head_merged 粒度不支持静态参数广播
        elif granularity == "value":
            # (max_seq_len, H, D) → 截取前 N 个 token → 转置为 (1, H, N, D) → 广播
            # (max_seq_len, H, D) -> (B, H, N, D) - slice to actual sequence length
            value_params = tensor[:N, :, :]  # Take first N tokens: (N, H, D)
            return value_params.view(1, N, H, D).transpose(1, 2).expand(B, H, N, D)  # (1, N, H, D) -> (1, H, N, D) -> (B, H, N, D)
        else:
            raise ValueError(f"Invalid granularity: {granularity}")

    def update_temperature(self, step: int):
        """
        Update temperature using exponential annealing schedule for gate only.
        使用指数退火策略更新门控温度。

        公式: temp = init_temp * (final_temp / init_temp) ^ (step / anneal_steps)
        温度从高逐渐降到低，使 Gumbel-Sigmoid 从平滑过渡到接近离散。

        Args:
            step: 当前训练步数
        """
        # 计算退火比例（限制在 [0, 1]）
        gate_ratio = min(step / self.anneal_steps, 1.0)
        # 指数退火：从 initial_temperature 平滑过渡到 final_temperature
        gate_temp = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** gate_ratio
        self.gate_temperature.fill_(gate_temp)
    
    
    def forward(self, source_kv: Tuple[Tensor, Tensor], target_kv: Tuple[Tensor, Tensor], position_ids: Optional[Tensor] = None, max_pos: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with unified projection logic.
        AllInOneProjector 的核心前向传播：投影 + 融合。

        整体流程:
        1. 将 source KV 的多头展平为一个大向量
        2. 通过 key/value projection MLP 投影到 target 维度
        3. (可选) 拼接 target KV 并通过 combiner 组合
        4. 生成门控 (gate) 和权重 (weight)
        5. 按融合公式组合输出:
           output = target_term + gate * normalized_weight * projected

        Args:
            source_kv   : (source_key, source_value)，形状各为 (B, H_s, N, D_s)
                          Sharer 模型的 KV
            target_kv   : (target_key, target_value)，形状各为 (B, H_t, N, D_t)
                          Receiver 模型的 KV
            position_ids: 位置 ID (B, N)，当 pos_emb=True 时使用（可选）
            max_pos     : 最大位置（可选，保留参数）
        Returns:
            (output_key, output_value): 融合后的 KV，形状各为 (B, H_t, N, D_t)
        """
        source_key, source_value = source_kv
        target_key, target_value = target_kv

        # ── 获取形状信息 ──
        # Get shapes
        B, H_s, N, D_s = source_key.shape   # Sharer: batch, 头数, 序列长度, 头维度
        _, H_t, _, D_t = target_key.shape    # Receiver: batch, 头数, 序列长度, 头维度

        # ── Step 1: 展平多头维度 ──
        # Reshape for projection: (B, H, N, D) -> (B, N, H*D)
        # 将 (B, H, N, D) 转置为 (B, N, H, D) 再展平为 (B, N, H*D)
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, H_s * D_s)     # (B, N, H_s*D_s)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(B, N, H_s * D_s)  # (B, N, H_s*D_s)

        # ── Step 2: 通过 MLP 投影到 target 维度 ──
        # Project source to target dimension / 将 source 投影到 target 的展平维度
        projected_key_flat = self.key_projection(source_key_flat)    # (B, N, H_t * D_t)
        projected_value_flat = self.value_projection(source_value_flat)  # (B, N, H_t * D_t)

        # ── Step 3: (可选) 拼接 target KV 并通过 combiner 组合 ──
        # Handle concatenation if enabled / 若启用拼接模式
        if self.use_concat:
            # 将 target KV 也展平
            target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, H_t * D_t)    # (B, N, H_t*D_t)
            target_value_flat = target_value.transpose(1, 2).contiguous().view(B, N, H_t * D_t)  # (B, N, H_t*D_t)

            # Concatenate and combine / 拼接 [projected, target] 后通过线性组合器降维
            combined_key = torch.cat([projected_key_flat, target_key_flat], dim=-1)    # (B, N, 2*H_t*D_t)
            combined_value = torch.cat([projected_value_flat, target_value_flat], dim=-1)  # (B, N, 2*H_t*D_t)

            final_projected_key_flat = self.key_combiner(combined_key)      # (B, N, H_t*D_t)
            final_projected_value_flat = self.value_combiner(combined_value)  # (B, N, H_t*D_t)
        else:
            final_projected_key_flat = projected_key_flat
            final_projected_value_flat = projected_value_flat

        # ── Step 4: 恢复多头形状 (B, N, H_t*D_t) -> (B, H_t, N, D_t) ──
        # Reshape back: (B, N, H_t * D_t) -> (B, H_t, N, D_t)
        projected_key = final_projected_key_flat.view(B, N, H_t, D_t).transpose(1, 2)    # (B, H_t, N, D_t)
        projected_value = final_projected_value_flat.view(B, N, H_t, D_t).transpose(1, 2)  # (B, H_t, N, D_t)

        # ── Step 5: 生成门控和权重 ──
        # Generate gates and weights (may need projected tensors for input features)
        # 判断是否需要在生成门控/权重时使用投影后的张量
        needs_projected_for_gate = self.gate_depends_on_input and self.gate_input_features in [
            "target_projected_key", "target_projected_value", "target_projected_both"
        ]
        needs_projected_for_weight = self.weight_depends_on_input and self.weight_input_features in [
            "target_projected_key", "target_projected_value", "target_projected_both"
        ]

        if needs_projected_for_gate or needs_projected_for_weight:
            # 需要投影后张量作为输入 → 传入 projected_key/value
            gate_logit = self._generate_gates(target_key, target_value, projected_key, projected_value)
            key_weight, value_weight = self._generate_weights(target_key, target_value, projected_key, projected_value)
        else:
            # 不需要投影后张量 → 仅使用 target KV
            gate_logit = self._generate_gates(target_key, target_value)
            key_weight, value_weight = self._generate_weights(target_key, target_value)

        # ── Step 6: 将门控和权重重塑为目标形状 (B, H_t, N, D_t) ──
        # Reshape gates and weights to match target shape
        target_shape = (B, H_t, N, D_t)
        if self.gate_depends_on_input:
            # 动态门控：根据粒度进行重塑，所有粒度都保留 token 维度 N
            if self.gate_granularity == "scalar":
                # MLP 输出 (B, 1) → 广播到 (B, H, N, D)
                gate_logit = gate_logit.view(B, 1, 1, 1).expand(target_shape)
            elif self.gate_granularity == "token":
                # MLP 输出 (B, N, 1) → unsqueeze → 广播到 (B, H, N, D)
                gate_logit = gate_logit.unsqueeze(1).unsqueeze(-1).expand(target_shape)
            elif self.gate_granularity == "head_merged":
                # MLP 输出 (B, N, H) → permute → 广播到 (B, H, N, D)
                gate_logit = gate_logit.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
            elif self.gate_granularity == "head":
                # MLP 输出 (B, H, N, 1) → 在 D 维度上广播
                gate_logit = gate_logit.expand(B, H_t, N, D_t)
            elif self.gate_granularity == "value":
                # MLP 输出 (B, H, N, D) → 已是目标形状
                pass
        else:
            # 静态门控：通过 _reshape_for_granularity 广播
            gate_logit = self._reshape_for_granularity(gate_logit, self.gate_granularity, target_shape)

        if self.weight_depends_on_input:
            # 动态权重：根据粒度进行重塑（与门控对称）
            if self.weight_granularity == "scalar":
                key_weight = key_weight.view(B, 1, 1, 1).expand(target_shape)
                value_weight = value_weight.view(B, 1, 1, 1).expand(target_shape)
            elif self.weight_granularity == "token":
                key_weight = key_weight.unsqueeze(1).expand(target_shape)
                value_weight = value_weight.unsqueeze(1).expand(target_shape)
            elif self.weight_granularity == "head_merged":
                key_weight = key_weight.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
                value_weight = value_weight.permute(0, 2, 1).unsqueeze(-1).expand(B, H_t, N, D_t)
            elif self.weight_granularity == "head":
                key_weight = key_weight.expand(B, H_t, N, D_t)
                value_weight = value_weight.expand(B, H_t, N, D_t)
            elif self.weight_granularity == "value":
                pass  # 已在正确形状
        else:
            # 静态权重：通过 _reshape_for_granularity 广播
            key_weight = self._reshape_for_granularity(key_weight, self.weight_granularity, target_shape)
            value_weight = self._reshape_for_granularity(value_weight, self.weight_granularity, target_shape)

        # ── Step 7: 应用门控和权重归一化 ──
        # Apply gating and selection / 门控：训练时 Gumbel-Sigmoid，推理时硬门控
        gate = self._apply_gumbel_sigmoid(gate_logit)

        # Normalize weights using dynamic temperature / 用 sigmoid 将权重归一化到 [0,1]
        # scalar_temperature 控制 sigmoid 的锐度，温度越低曲线越陡
        normalized_key_weight = torch.sigmoid(key_weight / self.scalar_temperature)
        normalized_value_weight = torch.sigmoid(value_weight / self.scalar_temperature)

        # ── Step 8: 最终融合 ──
        # Final combination / 计算 projected 部分的贡献（始终存在）
        # projected_term = gate * normalized_weight * projected
        projected_key_term = gate * normalized_key_weight * projected_key
        projected_value_term = gate * normalized_value_weight * projected_value

        # Compute target (self) contribution depending on flags
        # 根据配置计算 target (self) 部分的贡献
        if self.add_self:
            if self.preserve_target_weight:
                # 传统混合模式: target_term = (1 - normalized_weight) * target
                # 当 weight→1 时，target 信号被抑制；weight→0 时，target 信号完全保留
                target_key_term = (1 - normalized_key_weight) * target_key
                target_value_term = (1 - normalized_value_weight) * target_value
            else:
                # 简化模式: target_term = target（target 不受权重影响，始终完整保留）
                target_key_term = target_key
                target_value_term = target_value
        else:
            # 不包含 target 自身信号: target_term = 0
            target_key_term = torch.zeros_like(target_key)
            target_value_term = torch.zeros_like(target_value)

        # Final outputs / 最终输出 = target_term + projected_term
        output_key = target_key_term + projected_key_term
        output_value = target_value_term + projected_value_term
        
        return (output_key, output_value)

class QwenStyleLayer(nn.Module):
    """
    One Qwen3-style MLP sublayer:
      y = x + Dropout( down( SiLU(gate(LN(x))) * up(LN(x)) ) )
    Qwen3 风格的 MLP 子层：Pre-RMSNorm + SwiGLU + 残差连接。

    计算流程:
        1. RMSNorm 归一化输入 x → h
        2. SwiGLU: gate = SiLU(gate_linear(h)), up = up_linear(h), 逐元素相乘
        3. down_linear 降维回 hidden_size
        4. Dropout（可选）
        5. 残差连接: output = x + h

    - Pre-norm with RMSNorm / 使用 RMSNorm 进行预归一化
    - Bias-free linears / 线性层无偏置（现代 LLM 常见做法）

    Args:
        hidden_size      : 隐藏层维度
        intermediate_size: 中间扩展维度
        dropout          : Dropout 概率（0 表示不使用）
        dtype            : 数据类型
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)  # Pre-RMSNorm
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)  # SwiGLU 门控投影
        self.up   = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)  # SwiGLU 值投影
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)  # 降维回 hidden_size
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)  # Pre-norm
        h = self.act(self.gate(h)) * self.up(h)  # SwiGLU 门控激活
        h = self.down(h)  # 降维
        h = self.drop(h)  # Dropout
        return x + h      # 残差连接


class StandardFFNLayer(nn.Module):
    """
    Pre-norm RMSNorm, classic MLP:
      y = x + Dropout( W2( Act( W1( RMSNorm(x) ) ) ) )
    经典前馈网络子层：Pre-RMSNorm + 单层非线性变换 + 残差连接。

    与 QwenStyleLayer 的区别: 不使用 SwiGLU，而是单一的激活函数（GELU/ReLU/SiLU）。

    计算流程:
        1. RMSNorm 归一化
        2. W1 升维 → 激活函数
        3. W2 降维
        4. Dropout + 残差

    - No SwiGLU: single hidden nonlinearity (GELU/ReLU/SiLU) / 无 SwiGLU
    - Bias-free linears (common in modern LLM FFNs) / 线性层无偏置

    Args:
        hidden_size      : 隐藏层维度
        intermediate_size: 中间扩展维度
        dropout          : Dropout 概率
        dtype            : 数据类型
        activation       : 激活函数名称 "gelu" / "relu" / "silu"
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, dtype=dtype)  # Pre-RMSNorm
        self.w1   = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)  # 升维
        self.w2   = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)  # 降维
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 根据字符串选择激活函数
        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)      # Pre-norm
        h = self.act(self.w1(h))  # 升维 + 激活
        h = self.w2(h)        # 降维
        h = self.drop(h)      # Dropout
        return x + h          # 残差连接


class RegularMLP(nn.Module):
    """
    Qwen3-style stacked MLP operating at a fixed hidden size.
    固定隐藏维度的堆叠式 MLP —— 由多个 StandardFFNLayer 子层组成。

    用于 C2CProjector 内部的特征提取。不包含输入/输出投影层，
    调用者需要自行处理维度对齐。

    - No input/output projections; caller is responsible for projections.
    - num_layers repeats of Qwen-style FFN sublayer (pre-RMSNorm, SwiGLU, bias-free)

    Args:
        hidden_dim      : 隐藏层维度（所有层共享）
        intermediate_dim: FFN 中间扩展维度
        num_layers      : FFN 子层堆叠数量（必须 >= 1）
        dropout         : Dropout 概率
        dtype           : 数据类型
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        intermediate_dim: int = 3072,
        num_layers: int = 3,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        # 堆叠 num_layers 个 StandardFFNLayer 子层
        self.blocks = nn.ModuleList([
            StandardFFNLayer(hidden_size=hidden_dim, intermediate_size=intermediate_dim, dropout=dropout, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # 依次通过每个 FFN 子层
        for blk in self.blocks:
            x = blk(x)
        return x

@register_model   # 注册到 PROJECTOR_REGISTRY
@capture_init_args # 保存 __init__ 参数
class C2CProjector(Projector):
    """
    Concise projector specialized to a fixed C2C configuration using StandardMLP.
    C2C 专用精简投影器 —— 使用 RegularMLP（StandardFFNLayer 堆叠）实现。

    与 AllInOneProjector 相比，C2CProjector 是一个更精简、更特化的实现：
    - 使用共享的中间表示同时产生投影特征和权重
    - 拼接 source + target 后通过统一网络处理

    网络结构（核心）:
        1. 拼接 [source_flat, target_flat] → Linear 投影到 hidden_dim
        2. 1 层 RegularMLP 提取共享中间表示
        3a. 中间表示 → 权重路径: RegularMLP → Linear → (B, N, Ht) 每头权重
        3b. 中间表示 → 投影路径: RegularMLP → Linear → (B, N, Ht*Dt) 投影特征
        4. 融合: output = target + gate * sigmoid(scalar) * projected

    固定配置:
    - Projections: RegularMLP (pre-RMSNorm, SwiGLU, residual per sublayer)
    - Concat: 始终启用，拼接 source + target
    - Gate: 标量参数 + Gumbel-Sigmoid（训练时）
    - Weights: 每头（head_merged 粒度）的输入相关权重
    - Target preservation: add_self=True, preserve_target_weight=False
    - Temperatures: 退火门控温度 (1.0 → 0.001 over 1929 steps), scalar_temperature=1.0

    Args:
        source_dim          : Sharer 每头维度 D_s
        target_dim          : Receiver 每头维度 D_t
        source_num_heads    : Sharer 头数 H_s
        target_num_heads    : Receiver 头数 H_t
        intermediate_dim    : FFN 中间扩展维度
        hidden_dim          : 隐藏层维度
        num_layers          : 总层数（必须 >= 3）
        dropout             : Dropout 概率
        initial_temperature : 门控温度退火初始值
        final_temperature   : 门控温度退火终止值
        anneal_steps        : 退火总步数
        dtype               : 数据类型
        zero_init           : 是否将投影输出层初始化为零
    """

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        source_num_heads: int = 1,
        target_num_heads: int = 1,
        intermediate_dim: int = 1024,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.001,
        anneal_steps: int = 1929,
        dtype: torch.dtype = torch.float32,
        zero_init: bool = False
    ):
        super().__init__()

        assert num_layers >= 3, "num_layers must be >= 3"

        # ── 保存维度信息 ──
        # Dimensions
        self.source_dim = source_dim         # Sharer 每头维度 D_s
        self.target_dim = target_dim         # Receiver 每头维度 D_t
        self.source_num_heads = source_num_heads  # Sharer 头数 H_s
        self.target_num_heads = target_num_heads  # Receiver 头数 H_t

        # 展平后的维度
        # Sizes
        in_dim = source_dim * source_num_heads   # H_s * D_s (source 展平)
        out_dim = target_dim * target_num_heads   # H_t * D_t (target 展平)

        # ── 1) 拼接 [source, target] 并投影到 hidden_dim ──
        # concat(source_X, target_X) then project to hidden_dim
        self.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)   # Key: (H_s*D_s + H_t*D_t) → hidden
        self.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)  # Value: 同上

        # ── 2) 1 层 RegularMLP 提取共享中间表示 ──
        # one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        self.key_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.value_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)

        # ── 3a) 权重路径: 中间表示 → RegularMLP → Linear → 每头权重 ──
        # intermediate representation → (L-2)-layer MLP for weights → project to head dim
        self.key_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.value_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)    # 输出每头一个权重值
        self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)  # 输出每头一个权重值

        # ── 3b) 投影路径: 中间表示 → RegularMLP → Linear → 投影特征 ──
        # intermediate representation → (L-2)-layer MLP for projected_X → finally project hidden_dim → out_dim
        self.key_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        self.value_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)    # hidden → H_t*D_t
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)  # hidden → H_t*D_t

        # 可选：将投影输出层初始化为零（使训练初期投影信号为零）
        if zero_init:
            print("Initializing projector weights to zero")
            nn.init.zeros_(self.key_proj_out.weight)
            nn.init.zeros_(self.key_proj_out.bias)
            nn.init.zeros_(self.value_proj_out.weight)
            nn.init.zeros_(self.value_proj_out.bias)

        # ── 门控参数与温度调度 ──
        # Scalar key/value gate parameters and temperature schedule
        self.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))    # Key 全局门控标量
        self.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))  # Value 全局门控标量
        self.use_gumbel = True
        self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.anneal_steps = anneal_steps

        # Temperature for weight normalization / 权重归一化温度（固定为 1.0）
        self.scalar_temperature = 1.0

    def update_temperature(self, step: int):
        """
        更新门控温度（指数退火）。
        公式: temp = init * (final / init) ^ (step / anneal_steps)
        """
        ratio = min(step / self.anneal_steps, 1.0)
        temp = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** ratio
        self.gate_temperature.fill_(temp)

    def forward(
        self,
        source_kv: Tuple[Tensor, Tensor],
        target_kv: Tuple[Tensor, Tensor],
        position_ids: Optional[Tensor] = None,
        max_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        C2CProjector 的前向传播。

        核心流程:
        1. 展平多头维度并拼接 source + target
        2. Linear 投影到 hidden_dim → 1 层 RegularMLP → 共享中间表示
        3a. 权重路径: 中间表示 → MLP → Linear → 每头权重 (B, N, Ht)
        3b. 投影路径: 中间表示 → MLP → Linear → 投影特征 (B, N, Ht*Dt)
        4. 应用 Gumbel-Sigmoid 门控 + sigmoid 权重归一化
        5. 融合: output = target + gate * sigmoid(scalar) * projected

        Args:
            source_kv   : (source_key, source_value)，形状各为 (B, Hs, N, Ds)
            target_kv   : (target_key, target_value)，形状各为 (B, Ht, N, Dt)
            position_ids: 可选（未使用）
            max_pos     : 可选（未使用）
        Returns:
            (output_key, output_value): 融合后的 KV，形状各为 (B, Ht, N, Dt)
        """
        source_key, source_value = source_kv
        target_key, target_value = target_kv

        B, Hs, N, Ds = source_key.shape   # Sharer: batch, 头数, 序列长度, 头维度
        _, Ht, _, Dt = target_key.shape    # Receiver: batch, 头数, 序列长度, 头维度

        # ── Step 1: 展平多头维度 ──
        # Flatten heads / 将 (B, H, N, D) → (B, N, H*D)
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, Hs * Ds)      # (B, N, Hs*Ds)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(B, N, Hs * Ds)   # (B, N, Hs*Ds)
        target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, Ht * Dt)      # (B, N, Ht*Dt)
        target_value_flat = target_value.transpose(1, 2).contiguous().view(B, N, Ht * Dt)   # (B, N, Ht*Dt)

        # ── Step 2: 拼接 source + target 并投影到 hidden_dim ──
        # 1) concat source and target features along channel / 在通道维度拼接
        key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)      # (B, N, Hs*Ds + Ht*Dt)
        value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1)  # (B, N, Hs*Ds + Ht*Dt)

        # 2) project to hidden dim / 通过 Linear 投影到 hidden_dim
        key_hidden = self.key_in(key_cat)      # (B, N, hidden_dim)
        value_hidden = self.value_in(value_cat)  # (B, N, hidden_dim)

        # ── Step 3: 1 层 RegularMLP 提取共享中间表示 ──
        # 3) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        key_hidden = self.key_mlp1(key_hidden)      # (B, N, hidden_dim) 共享中间表示
        value_hidden = self.value_mlp1(value_hidden)  # (B, N, hidden_dim) 共享中间表示

        # ── Step 4b: 投影路径 —— 中间表示 → 投影特征 ──
        # 4b) intermediate representation -> projected feature path
        key_proj_hidden = self.key_proj_out(self.key_proj_mlp2(key_hidden))    # (B, N, Ht * Dt)
        value_proj_hidden = self.value_proj_out(self.value_proj_mlp2(value_hidden))  # (B, N, Ht * Dt)
        # 恢复多头形状: (B, N, Ht*Dt) → (B, Ht, N, Dt)
        projected_key = key_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)    # (B, Ht, N, Dt)
        projected_value = value_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)  # (B, Ht, N, Dt)

        # ── Step 4a: 权重路径 —— 中间表示 → 每头权重标量 ──
        # 4a) intermediate representation -> scalar path
        key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))        # (B, N, Ht)
        value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden))  # (B, N, Ht)
        # 转置为 (B, Ht, N, 1) 以便与 KV 张量对齐
        key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)    # (B, N, Ht) → (B, Ht, N, 1)
        value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)

        # ── Step 5: 门控计算 (Gumbel-Sigmoid) ──
        # Key/value gates: element-wise Gumbel noise with scalar logits (broadcast over channels)
        # 将标量门控参数扩展为 (1,1,1,1) 以便广播
        key_gate_logit = self.key_gate_logit.view(1, 1, 1, 1)
        value_gate_logit = self.value_gate_logit.view(1, 1, 1, 1)
        if self.training and self.use_gumbel:
            # 训练时：采样 Gumbel 噪声，实现可微的离散门控
            # 噪声形状 (B, Ht, N, 1)：每个 batch、每个头、每个 token 独立采样
            u1 = torch.rand(B, Ht, N, 1, device=key_gate_logit.device, dtype=key_gate_logit.dtype)
            u2 = torch.rand(B, Ht, N, 1, device=value_gate_logit.device, dtype=value_gate_logit.dtype)
            g1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)  # Gumbel 采样: -log(-log(U))
            g2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
            key_gate = torch.sigmoid((key_gate_logit + g1) / self.gate_temperature)    # Gumbel-Sigmoid
            value_gate = torch.sigmoid((value_gate_logit + g2) / self.gate_temperature)  # Gumbel-Sigmoid
        else:
            # 推理时：硬门控
            key_gate = (key_gate_logit > 0).float()
            value_gate = (value_gate_logit > 0).float()

        # ── Step 6: 权重归一化 ──
        # Normalize scalars (scalar_temperature=1.0) / sigmoid 归一化到 [0,1]
        norm_key_scalar = torch.sigmoid(key_scalar)      # (B, Ht, N, 1) 每头每 token 的权重
        norm_value_scalar = torch.sigmoid(value_scalar)  # (B, Ht, N, 1)

        # ── Step 7: 最终融合 ──
        # Combine (preserve_target_weight=False, add_self=True)
        # 融合公式: output = target + gate * sigmoid(scalar) * projected
        # - target 始终完整保留（preserve_target_weight=False）
        # - gate 控制是否启用投影信号（Gumbel-Sigmoid）
        # - sigmoid(scalar) 控制每头的融合强度
        output_key = target_key + key_gate * norm_key_scalar * projected_key
        output_value = target_value + value_gate * norm_value_scalar * projected_value

        # ── 保存中间变量用于下游分析/可视化 ──
        # Expose capture attributes for downstream analysis scripts
        try:
            # Store normalized scalars (detach to avoid autograd, keep device-agnostic via CPU)
            # 保存归一化后的权重（脱离计算图，用于分析）
            self.last_norm_key_scalar = norm_key_scalar.detach().cpu()
            self.last_norm_value_scalar = norm_value_scalar.detach().cpu()
            # Store gate logits as python floats (parameters are scalar)
            # 保存门控 logits 为 Python 标量
            self.last_key_gate_logit = float(self.key_gate_logit.detach().cpu().item())
            self.last_value_gate_logit = float(self.value_gate_logit.detach().cpu().item())
        except Exception:
            # Best-effort capture; never break forward path / 尽力捕获，不影响前向传播
            pass

        return output_key, output_value

def save_projector(obj: Projector, file_path: str) -> None:
    """
    将投影器对象序列化保存到文件。

    使用 pickle 序列化（通过 rosetta.utils.registry.save_object）。

    Args:
        obj       : 要保存的 Projector 实例
        file_path : 目标文件路径
    """
    save_object(obj, file_path)

def load_projector(file_path: str, override_args: Optional[dict] = None) -> Projector:
    """
    从文件加载投影器对象。

    使用 pickle 反序列化（通过 rosetta.utils.registry.load_object）。
    override_args 可用于覆盖保存时的 __init__ 参数。

    Args:
        file_path     : 源文件路径
        override_args : 可选，覆盖 __init__ 参数的字典
    Returns:
        反序列化后的 Projector 实例
    """
    return load_object(file_path, get_projector_class, override_args)

def create_projector(projector_type: str, **kwargs) -> Projector:
    """
    Factory function to create a projector based on type.
    工厂函数 —— 根据类型字符串创建对应的投影器实例。

    通过 PROJECTOR_REGISTRY 查找已注册的投影器类，并用 kwargs 实例化。

    Args:
        projector_type: 投影器类型名称字符串（如 "AllInOneProjector", "C2CProjector"）
                        String indicating the type of projector
        **kwargs      : 传递给投影器构造函数的参数
                        Additional arguments to pass to the projector constructor

    Returns:
        对应类型的投影器实例
        An instance of the appropriate projector
    """
    # Prefer using the unified registry getter (handles case-insensitive keys)
    # 通过注册表查找投影器类（支持大小写不敏感的键名）
    try:
        cls = get_projector_class(projector_type)
    except ValueError as e:
        raise e
    return cls(**kwargs)