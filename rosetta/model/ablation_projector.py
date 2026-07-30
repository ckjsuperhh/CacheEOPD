"""
Ablation Projector: A configurable projector for ablation studies based on C2CProjector.
Allows gradual removal of components to study their individual contributions.

消融实验投影器（Ablation Projector）模块说明
=============================================

本文件定义了 ``AblationProjector`` 类，它是 C2C（Cache-to-Cache）框架中用于消融实验
(ablation study) 的可配置投影器。基于 ``C2CProjector`` 的架构设计，通过逐步移除不同组件
来研究每个组件对最终性能的独立贡献。

核心类:
    - ``AblationProjector``: 继承自 ``Projector`` 基类，支持 5 个消融级别 (0-4)：
        - Level 0: 完整 C2C（基线 baseline）
        - Level 1: 移除标量权重（scalar weights 固定为 1.0）
        - Level 2: 移除门控（gates 固定为 1.0）+ 移除标量权重
        - Level 3: 移除目标端贡献（仅使用 source/sharer 特征）+ 无门控 + 无标量
        - Level 4: 仅移除门控，保留标量权重和目标端贡献

与其他模块的关系:
    - ``rosetta.model.projector.Projector``: 投影器基类，定义了投影器的接口
    - ``rosetta.model.projector.RegularMLP``: 常规 MLP 模块，用于构建投影网络
    - ``rosetta.utils.registry``: 模型注册机制，``@register_model`` 将此类注册到模型工厂
    - ``C2CProjector``: 完整的 C2C 投影器（本类的参考实现）

在 C2C 框架中的位置:
    Sharer 模型产生 KV-Cache → **AblationProjector 投影并融合** → Receiver 模型接收融合后 KV-Cache
    投影器负责：
    1. 将 sharer 的 KV 维度投影到 receiver 兼容的维度
    2. 将投影后的 KV 与 receiver 自身的 KV 进行融合
    3. 通过消融配置控制融合方式
"""

# === 标准库导入 ===
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Literal

# === Rosetta 框架内部模块导入 ===
# register_model: 将类注册到全局模型注册表，使其可以通过配置文件按名称实例化
# capture_init_args: 装饰器，自动捕获 __init__ 参数用于序列化/反序列化
from rosetta.utils.registry import register_model, capture_init_args
# Projector: 投影器基类，定义了 source_kv + target_kv → output_kv 的接口
from rosetta.model.projector import Projector
# RegularMLP: 标准多层感知机模块（Linear → GELU → Dropout 重复堆叠）
from rosetta.model.projector import RegularMLP


@register_model        # 将此类注册到 rosetta 模型注册表，可通过配置文件中 "AblationProjector" 名称创建实例
@capture_init_args     # 捕获 __init__ 的所有参数，便于模型保存与恢复
class AblationProjector(Projector):
    """
    Ablation study projector based on C2CProjector with configurable component removal.
    
    基于 C2CProjector 的消融实验投影器，支持可配置的组件移除。
    
    Ablation levels:
    0. Full C2C (baseline)
    1. Remove scalar weights (set to 1.0)
    2. Remove gates (set to 1.0) 
    3. Remove target contribution (only use source)
    4. Remove gates only (gates=1.0), keep scalars and target
    
    Each level builds on the previous one, allowing gradual degradation study.

    消融级别说明:
        - Level 0: 完整 C2C 基线 —— 包含标量权重、门控、目标端贡献
        - Level 1: 移除标量权重 —— 标量固定为 1.0，研究标量权重的贡献
        - Level 2: 在 Level 1 基础上再移除门控 —— 门控固定为 1.0
        - Level 3: 在 Level 2 基础上再移除目标端 —— 仅使用 source（sharer）特征
        - Level 4: 特殊消融 —— 仅移除门控（=1.0），保留标量权重和目标端贡献

    网络结构（完整 Level 0）::

        source_kv ─┐
                   ├─ concat → key_in/value_in (Linear)
        target_kv ─┘         ↓
                      key_mlp1/value_mlp1 (1层公共MLP，提取共享中间表示)
                             ↓
               ┌─────────────┴──────────────┐
               ↓                            ↓
    scalar_mlp2 → scalar_head        proj_mlp2 → proj_out
    (逐头标量权重, sigmoid)          (投影后的KV, 维度 = Ht*Dt)
               ↓                            ↓
          norm_scalar               projected_kv
               ↓                            ↓
               └─── gate * scalar * projected ──┐
                                                ↓
                              output = target_kv + (gate * scalar * projected)

    门控（Gate）机制:
        - 使用 Gumbel-Softmax 采样实现可微分的二值门控
        - 训练时：采样 Gumbel 噪声 + sigmoid 近似
        - 推理时：直接阈值判断（logit > 0 → 开，否则关）
        - 温度退火：从高到低指数退火，使训练过程中门控从软决策逐步变为硬决策
    """

    def __init__(
        self,
        source_dim: int,           # source（sharer）模型每个注意力头的维度 Ds
        target_dim: int,           # target（receiver）模型每个注意力头的维度 Dt
        source_num_heads: int = 1, # source 模型的注意力头数 Hs
        target_num_heads: int = 1, # target 模型的注意力头数 Ht
        intermediate_dim: int = 1024, # MLP 中间层维度
        hidden_dim: int = 1024,       # 隐藏层维度（统一投影后的特征维度）
        num_layers: int = 3,          # 总 MLP 层数（含输入层和输出层）
        dropout: float = 0.1,         # Dropout 比率
        initial_temperature: float = 1.0,    # Gumbel 门控的初始温度（高温 → 软门控）
        final_temperature: float = 0.001,    # Gumbel 门控的最终温度（低温 → 硬门控）
        anneal_steps: int = 1929,            # 温度退火总步数
        dtype: torch.dtype = torch.float32,  # 参数数据类型
        
        # === 消融实验配置（Ablation configuration） ===
        ablation_level: int = 0,  # 消融级别: 0=完整, 1=无标量, 2=无门控+无标量, 3=无目标端, 4=仅无门控
        use_scalar_weights: bool = True,  # 是否使用标量权重（可被 ablation_level 覆盖）
        use_gates: bool = True,          # 是否使用门控（可被 ablation_level 覆盖）
        use_target: bool = True,         # 是否使用目标端 KV（可被 ablation_level 覆盖）
    ):
        """
        初始化消融实验投影器。

        Args:
            source_dim: Sharer 模型每个注意力头的特征维度（例如 128）
            target_dim: Receiver 模型每个注意力头的特征维度（例如 128）
            source_num_heads: Sharer 模型的注意力头数（例如 32）
            target_num_heads: Receiver 模型的注意力头数（例如 32）
            intermediate_dim: MLP 中间扩展层维度
            hidden_dim: 统一隐藏表示维度
            num_layers: 投影路径 MLP 总层数
            dropout: Dropout 概率
            initial_temperature: Gumbel-Softmax 初始温度
            final_temperature: Gumbel-Softmax 最终温度
            anneal_steps: 温度退火总训练步数
            dtype: 张量数据类型
            ablation_level: 消融级别 (0-4)
            use_scalar_weights: 是否启用标量权重
            use_gates: 是否启用门控
            use_target: 是否使用目标端 KV 特征
        """
        super().__init__()

        assert 0 <= ablation_level <= 4, "ablation_level must be 0, 1, 2, 3, or 4"
        # 消融级别必须在 0-4 之间

        # === 基本维度信息 ===
        self.source_dim = source_dim               # Sharer 每头维度 Ds
        self.target_dim = target_dim               # Receiver 每头维度 Dt
        self.source_num_heads = source_num_heads   # Sharer 头数 Hs
        self.target_num_heads = target_num_heads   # Receiver 头数 Ht
        self.ablation_level = ablation_level       # 消融级别

        # === 根据消融级别覆盖组件开关 ===
        # Override component usage based on ablation level
        if ablation_level == 4:
            # Level 4 特殊情况：仅禁用门控，保留标量权重和目标端
            # Special case: disable gates only, keep scalars and target
            use_scalar_weights = True
            use_gates = False
            use_target = True
        else:
            # Level 1+: 逐步禁用各组件（累积式消融）
            if ablation_level >= 1:
                use_scalar_weights = False  # 移除标量权重
            if ablation_level >= 2: 
                use_gates = False           # 移除门控
            if ablation_level >= 3:
                use_target = False          # 移除目标端贡献
            
        self.use_scalar_weights = use_scalar_weights  # 是否使用逐头标量权重
        self.use_gates = use_gates                    # 是否使用可学习门控
        self.use_target = use_target                  # 是否使用 target KV 作为输入和残差

        # === 计算展平后的输入/输出维度 ===
        # Sizes: 将多头展平后的总维度
        in_dim = source_dim * source_num_heads    # Hs * Ds: sharer 展平后总维度
        out_dim = target_dim * target_num_heads   # Ht * Dt: receiver 展平后总维度

        # === 步骤 1: 输入投影层 ===
        # 1) concat(source_X, target_X) then project to hidden_dim
        # 将 source 和 target 的 KV 拼接后投影到统一的 hidden_dim
        # If not using target, only use source features
        if self.use_target:
            # 完整模式：输入为 concat(source_flat, target_flat)，维度 = in_dim + out_dim
            self.key_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
            self.value_in = nn.Linear(in_dim + out_dim, hidden_dim, bias=True, dtype=dtype)
        else:
            # 消融 Level 3：仅使用 source 特征，输入维度 = in_dim
            # Only use source features
            self.key_in = nn.Linear(in_dim, hidden_dim, bias=True, dtype=dtype)
            self.value_in = nn.Linear(in_dim, hidden_dim, bias=True, dtype=dtype)

        # === 步骤 2: 公共嵌入 MLP（1层） ===
        # 2) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        # 从 hidden_dim 映射到 intermediate_dim 再映射回 hidden_dim
        # 用于提取 source 和 target 的共享中间表示
        self.key_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)
        self.value_mlp1 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=1, dropout=dropout, dtype=dtype)

        # === 步骤 3a: 标量权重分支（Scalar Weight Branch） ===
        # intermediate representation → MLP → 每个 target 注意力头一个标量权重
        # 用于控制投影后 KV 在每个头上的贡献比例
        # Only build if using scalar weights
        if self.use_scalar_weights:
            # 标量 MLP：hidden_dim → hidden_dim（1层）
            self.key_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
            self.value_scalar_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=hidden_dim, num_layers=1, dropout=dropout, dtype=dtype)
            # 标量头：hidden_dim → target_num_heads，为每个 target 头输出一个标量
            self.key_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)
            self.value_scalar_head = nn.Linear(hidden_dim, target_num_heads, dtype=dtype)

        # === 步骤 3b: 投影特征分支（Projected Feature Branch） ===
        # 3b) intermediate representation → (L-2)-layer MLP for projected_X → finally project hidden_dim → out_dim
        # 从中间表示经过 (num_layers-2) 层 MLP 投影到 out_dim = Ht * Dt
        self.key_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        self.value_proj_mlp2 = RegularMLP(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim, num_layers=num_layers-2, dropout=dropout, dtype=dtype)
        # 最终输出投影：hidden_dim → out_dim (Ht * Dt)
        self.key_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)
        self.value_proj_out = nn.Linear(hidden_dim, out_dim, bias=True, dtype=dtype)

        # === 门控参数与温度调度 ===
        # Scalar key/value gate parameters and temperature schedule
        # 可学习的门控 logit 参数和 Gumbel-Softmax 温度
        # Only build if using gates
        if self.use_gates:
            # 可学习的标量 logit，经 sigmoid 后作为门控系数
            # 初始化为 0.0 → sigmoid(0) = 0.5，即门控初始为半开状态
            self.key_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.value_gate_logit = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.use_gumbel = True  # 是否使用 Gumbel-Softmax 采样
            # 温度缓冲区（不作为梯度参数更新，由 update_temperature 手动设置）
            self.register_buffer("gate_temperature", torch.tensor(initial_temperature, dtype=dtype))
            self.initial_temperature = initial_temperature   # 初始温度（高温 → 软决策）
            self.final_temperature = final_temperature     # 最终温度（低温 → 硬决策）
            self.anneal_steps = anneal_steps               # 退火总步数

        # 标量权重的温度（目前固定为 1.0，未使用退火）
        # Temperature for weight normalization
        self.scalar_temperature = 1.0

    def update_temperature(self, step: int):
        """
        Update temperature using exponential annealing schedule for gates.
        
        更新 Gumbel-Softmax 门控的温度参数。
        采用指数退火策略，使门控从软决策（高温，sigmoid 输出接近 0.5）
        逐步过渡到硬决策（低温，sigmoid 输出接近 0 或 1）。
        
        温度公式: temp = T_init * (T_final / T_init) ^ (step / anneal_steps)
        - step=0 时: temp = T_init（最高温度，最软的决策）
        - step=anneal_steps 时: temp = T_final（最低温度，最硬的决策）
        - step > anneal_steps 时: 保持 T_final 不变

        Args:
            step: 当前训练步数，用于计算退火进度 ratio = step / anneal_steps
        """
        if self.use_gates:
            # 计算退火进度 ratio ∈ [0, 1]，限制最大为 1.0
            ratio = min(step / self.anneal_steps, 1.0)
            # 指数退火：在对数空间中进行线性插值
            # log(temp) = log(T_init) + ratio * (log(T_final) - log(T_init))
            # 等价于 temp = T_init * (T_final / T_init) ^ ratio
            temp = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** ratio
            # 用 fill_ 原地更新温度缓冲区（不创建新张量）
            self.gate_temperature.fill_(temp)

    def forward(
        self,
        source_kv: Tuple[Tensor, Tensor],   # Sharer 的 KV-Cache: (key, value)
        target_kv: Tuple[Tensor, Tensor],   # Receiver 的 KV-Cache: (key, value)
        position_ids: Optional[Tensor] = None,  # 位置编码 ID（本实现中未使用）
        max_pos: Optional[Tensor] = None,       # 最大位置（本实现中未使用）
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播：将 source（sharer）的 KV-Cache 投影并融合到 target（receiver）的 KV-Cache。

        KV-Cache 形状约定:
            输入: (B, H, N, D)
            - B: batch size（批次大小）
            - H: number of attention heads（注意力头数）
            - N: sequence length / number of tokens（序列长度）
            - D: head dimension（每个头的维度）

        整体流程:
            1. 展平多头维度: (B, H, N, D) → (B, N, H*D)
            2. 拼接 source 与 target 特征（消融 Level 3 时仅用 source）
            3. 输入线性投影 → hidden_dim
            4. 公共 MLP 提取中间表示
            5. 双分支输出:
               a. 标量权重分支: 为每个 target 注意力头生成一个 [0,1] 标量
               b. 投影特征分支: 生成投影后的 KV，形状 (B, Ht, N, Dt)
            6. 门控 * 标量 * 投影特征 → 投影贡献
            7. 输出 = target_kv + 投影贡献（消融 Level 3 时无 target 残差）

        Args:
            source_kv: Sharer 模型的 (key, value) 元组，形状均为 (B, Hs, N, Ds)
            target_kv: Receiver 模型的 (key, value) 元组，形状均为 (B, Ht, N, Dt)
            position_ids: 可选的位置 ID（本实现未使用）
            max_pos: 可选的最大位置（本实现未使用）

        Returns:
            (output_key, output_value): 融合后的 KV-Cache，形状均为 (B, Ht, N, Dt)
        """
        # === 解包 source 和 target 的 key/value ===
        source_key, source_value = source_kv     # 均为 (B, Hs, N, Ds)
        target_key, target_value = target_kv     # 均为 (B, Ht, N, Dt)

        # === 获取形状信息 ===
        B, Hs, N, Ds = source_key.shape   # B: batch, Hs: source 头数, N: 序列长度, Ds: source 头维度
        _, Ht, _, Dt = target_key.shape   # Ht: target 头数, Dt: target 头维度

        # === 展平多头维度: (B, H, N, D) → (B, N, H*D) ===
        # Flatten heads: 将注意力头维度移到最后一维并展平
        # transpose(1,2): (B, H, N, D) → (B, N, H, D)
        # contiguous().view(): (B, N, H, D) → (B, N, H*D)
        source_key_flat = source_key.transpose(1, 2).contiguous().view(B, N, Hs * Ds)    # (B, N, Hs*Ds)
        source_value_flat = source_value.transpose(1, 2).contiguous().view(B, N, Hs * Ds) # (B, N, Hs*Ds)
        target_key_flat = target_key.transpose(1, 2).contiguous().view(B, N, Ht * Dt)    # (B, N, Ht*Dt)
        target_value_flat = target_value.transpose(1, 2).contiguous().view(B, N, Ht * Dt) # (B, N, Ht*Dt)

        # === 步骤 1: 根据消融级别准备输入特征 ===
        # 1) Prepare input features based on ablation level
        if self.use_target:
            # 完整 C2C: 拼接 source 和 target 特征
            # Full C2C: concat source and target features
            key_cat = torch.cat([source_key_flat, target_key_flat], dim=-1)    # (B, N, Hs*Ds + Ht*Dt)
            value_cat = torch.cat([source_value_flat, target_value_flat], dim=-1) # (B, N, Hs*Ds + Ht*Dt)
        else:
            # 消融 Level 3: 仅使用 source 特征（完全忽略 target 输入）
            # Ablation level 3: only use source features
            key_cat = source_key_flat    # (B, N, Hs*Ds)
            value_cat = source_value_flat # (B, N, Hs*Ds)

        # === 步骤 2: 输入线性投影到 hidden_dim ===
        # 2) project to hidden dim
        key_hidden = self.key_in(key_cat)      # (B, N, hidden_dim)
        value_hidden = self.value_in(value_cat) # (B, N, hidden_dim)

        # === 步骤 3: 公共嵌入 MLP（1层）提取共享中间表示 ===
        # 3) one-layer common embedding MLP to get intermediate representation (at hidden_dim)
        key_hidden = self.key_mlp1(key_hidden)      # (B, N, hidden_dim) → (B, N, hidden_dim)
        value_hidden = self.value_mlp1(value_hidden) # (B, N, hidden_dim) → (B, N, hidden_dim)

        # === 步骤 4b: 投影特征分支（Projected Feature Branch） ===
        # intermediate representation → (num_layers-2) 层 MLP → 线性投影到 out_dim
        # 4b) intermediate representation -> projected feature path
        key_proj_hidden = self.key_proj_out(self.key_proj_mlp2(key_hidden))       # (B, N, hidden_dim) → (B, N, Ht*Dt)
        value_proj_hidden = self.value_proj_out(self.value_proj_mlp2(value_hidden)) # (B, N, hidden_dim) → (B, N, Ht*Dt)
        # 重塑回多头格式: (B, N, Ht*Dt) → (B, N, Ht, Dt) → (B, Ht, N, Dt)
        projected_key = key_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)       # (B, Ht, N, Dt)
        projected_value = value_proj_hidden.view(B, N, Ht, Dt).transpose(1, 2)   # (B, Ht, N, Dt)

        # === 步骤 4a: 标量权重分支（Scalar Weight Branch） ===
        # 4a) intermediate representation -> scalar path (if using scalar weights)
        if self.use_scalar_weights:
            # 中间表示 → 标量 MLP → 标量头 → 每个 target 头一个标量值
            key_scalar = self.key_scalar_head(self.key_scalar_mlp2(key_hidden))         # (B, N, Ht)
            value_scalar = self.value_scalar_head(self.value_scalar_mlp2(value_hidden)) # (B, N, Ht)
            # 重塑为广播兼容格式: (B, N, Ht) → (B, Ht, N) → (B, Ht, N, 1)
            # permute(0,2,1): 将 Ht 维度移到第 2 维
            # unsqueeze(-1): 添加末尾维度 1，用于后续与 (B, Ht, N, Dt) 广播相乘
            key_scalar = key_scalar.permute(0, 2, 1).unsqueeze(-1)   # (B, Ht, N, 1)
            value_scalar = value_scalar.permute(0, 2, 1).unsqueeze(-1)  # (B, Ht, N, 1)
            # Normalize scalars: 用 sigmoid 将标量映射到 [0, 1] 区间
            # sigmoid 输出表示每个 target 注意力头的贡献权重
            norm_key_scalar = torch.sigmoid(key_scalar)        # (B, Ht, N, 1)，值域 [0, 1]
            norm_value_scalar = torch.sigmoid(value_scalar)    # (B, Ht, N, 1)，值域 [0, 1]
        else:
            # 消融 Level 1+: 标量权重固定为 1.0（等权贡献，无自适应调节）
            # Ablation level 1+: set scalar weights to 1.0
            norm_key_scalar = torch.ones(B, Ht, N, 1, device=projected_key.device, dtype=projected_key.dtype)
            norm_value_scalar = torch.ones(B, Ht, N, 1, device=projected_value.device, dtype=projected_value.dtype)

        # === 门控机制（Gate Mechanism） ===
        # Key/value gates (if using gates)
        if self.use_gates:
            # 将可学习标量 logit 扩展为 (B, Ht, N, 1) 以进行广播
            key_gate_logit = self.key_gate_logit.view(1, 1, 1, 1)      # 标量 → (1,1,1,1)
            value_gate_logit = self.value_gate_logit.view(1, 1, 1, 1)  # 标量 → (1,1,1,1)
            if self.training and self.use_gumbel:
                # --- 训练阶段：Gumbel-Softmax 采样 ---
                # Gumbel-Softmax trick: 用连续分布近似离散门控，实现端到端可微
                # u ~ Uniform(0,1)，g = -log(-log(u)) 是 Gumbel 噪声
                u1 = torch.rand(B, Ht, N, 1, device=key_gate_logit.device, dtype=key_gate_logit.dtype)
                u2 = torch.rand(B, Ht, N, 1, device=value_gate_logit.device, dtype=value_gate_logit.dtype)
                # Gumbel 噪声采样: g = -log(-log(u))，加入小量 1e-20 避免 log(0)
                g1 = -torch.log(-torch.log(u1 + 1e-20) + 1e-20)
                g2 = -torch.log(-torch.log(u2 + 1e-20) + 1e-20)
                # Gumbel-Softmax: sigmoid((logit + gumbel_noise) / temperature)
                # 温度高 → 输出接近 0.5（软门控）；温度低 → 输出接近 0 或 1（硬门控）
                key_gate = torch.sigmoid((key_gate_logit + g1) / self.gate_temperature)
                value_gate = torch.sigmoid((value_gate_logit + g2) / self.gate_temperature)
            else:
                # --- 推理阶段：硬门控（阈值判断） ---
                # logit > 0 → gate = 1.0（开启），logit ≤ 0 → gate = 0.0（关闭）
                key_gate = (key_gate_logit > 0).float()
                value_gate = (value_gate_logit > 0).float()
        else:
            # 门控禁用：固定为 1.0（始终开启，不阻断投影贡献）
            # Gates disabled: set gates to 1.0 (always open)
            key_gate = torch.ones(B, Ht, N, 1, device=projected_key.device, dtype=projected_key.dtype)
            value_gate = torch.ones(B, Ht, N, 1, device=projected_value.device, dtype=projected_value.dtype)

        # === 计算投影贡献（Projected Contribution） ===
        # 三者逐元素相乘: 门控 * 标量权重 * 投影后 KV
        # 广播: gate(B,Ht,N,1) * scalar(B,Ht,N,1) * projected(B,Ht,N,Dt)
        # gate 和 scalar 在 Dt 维度上广播（对同一 token 的所有特征维度施加相同权重）
        # Compute projected contribution
        projected_key_term = key_gate * norm_key_scalar * projected_key          # (B, Ht, N, Dt)
        projected_value_term = value_gate * norm_value_scalar * projected_value  # (B, Ht, N, Dt)

        # === 融合输出（Final Fusion） ===
        # Compute target contribution (if using target)
        if self.use_target:
            # 完整 C2C: 残差连接，target_kv + 投影贡献
            # Full C2C: add target with projected
            output_key = target_key + projected_key_term          # (B, Ht, N, Dt)
            output_value = target_value + projected_value_term    # (B, Ht, N, Dt)
        else:
            # 消融 Level 3: 无 target 残差，仅使用投影后的 KV
            # Ablation level 3: only use projected (no target)
            output_key = projected_key_term      # (B, Ht, N, Dt)
            output_value = projected_value_term  # (B, Ht, N, Dt)

        return output_key, output_value  # 返回融合后的 KV-Cache，形状 (B, Ht, N, Dt)

    def get_ablation_info(self) -> dict:
        """
        Return information about current ablation configuration.
        
        返回当前消融配置的详细信息字典。
        可用于日志记录、实验追踪（如 wandb/tensorboard）中记录消融配置。
        
        Returns:
            dict: 包含以下键的字典:
                - 'ablation_level' (int): 消融级别 (0-4)
                - 'use_scalar_weights' (bool): 是否使用标量权重
                - 'use_gates' (bool): 是否使用门控
                - 'use_target' (bool): 是否使用目标端 KV
                - 'description' (str): 人类可读的消融级别描述
        """
        return {
            'ablation_level': self.ablation_level,
            'use_scalar_weights': self.use_scalar_weights,
            'use_gates': self.use_gates,
            'use_target': self.use_target,
            'description': self._get_ablation_description()
        }
    
    def _get_ablation_description(self) -> str:
        """
        Get human-readable description of current ablation level.
        
        获取当前消融级别的人类可读描述。
        
        Returns:
            str: 消融级别的描述文本
        """
        # 消融级别到描述的映射表
        descriptions = {
            0: "Full C2C (baseline)",                                           # 完整 C2C 基线
            1: "No scalar weights (scalars=1.0)",                               # 无标量权重
            2: "No gates (gates=1.0) + No scalar weights",                      # 无门控 + 无标量
            3: "No target (source-only) + No gates + No scalar weights",        # 无目标端 + 无门控 + 无标量
            4: "No gates (gates=1.0), keep scalars and target"                  # 仅无门控，保留标量和目标端
        }
        return descriptions.get(self.ablation_level, "Unknown ablation level")
        # 获取描述，未知级别返回默认文本


# =============================================================================
# 便捷工厂函数（Convenience Factory Functions）
# 用于快速创建特定消融级别的投影器实例
# =============================================================================

# Convenience functions for creating specific ablation levels
def create_ablation_projector(
    source_dim: int,       # Sharer 每头维度
    target_dim: int,       # Receiver 每头维度
    source_num_heads: int = 1,  # Sharer 头数
    target_num_heads: int = 1,  # Receiver 头数
    ablation_level: int = 0,    # 消融级别
    **kwargs                  # 其他参数透传给 AblationProjector
) -> AblationProjector:
    """
    Create an AblationProjector with specified ablation level.
    
    创建指定消融级别的投影器。
    通用的工厂函数，接受消融级别参数并创建对应的 AblationProjector 实例。
    
    Args:
        source_dim: Sharer 模型每个注意力头的维度
        target_dim: Receiver 模型每个注意力头的维度
        source_num_heads: Sharer 的注意力头数
        target_num_heads: Receiver 的注意力头数
        ablation_level: 消融级别 (0-4)
        **kwargs: 透传给 AblationProjector 的其他参数
                  （如 intermediate_dim, hidden_dim, num_layers, dropout 等）
    
    Returns:
        AblationProjector: 配置好的消融投影器实例
    """
    return AblationProjector(
        source_dim=source_dim,
        target_dim=target_dim,
        source_num_heads=source_num_heads,
        target_num_heads=target_num_heads,
        ablation_level=ablation_level,
        **kwargs
    )


def create_full_c2c_projector(**kwargs) -> AblationProjector:
    """
    Create full C2C projector (ablation level 0).
    
    创建完整 C2C 投影器（消融级别 0 = 基线）。
    所有组件均启用：标量权重 + 门控 + 目标端残差。
    
    Args:
        **kwargs: 透传给 create_ablation_projector 的参数
                  （需包含 source_dim, target_dim 等必要参数）
    
    Returns:
        AblationProjector: 完整 C2C 基线投影器
    """
    return create_ablation_projector(ablation_level=0, **kwargs)


def create_no_scalar_projector(**kwargs) -> AblationProjector:
    """
    Create projector without scalar weights (ablation level 1).
    
    创建无标量权重的投影器（消融级别 1）。
    标量权重固定为 1.0，研究自适应标量权重的贡献。
    
    Args:
        **kwargs: 透传给 create_ablation_projector 的参数
    
    Returns:
        AblationProjector: 无标量权重的投影器
    """
    return create_ablation_projector(ablation_level=1, **kwargs)


def create_no_gate_projector(**kwargs) -> AblationProjector:
    """
    Create projector without gates (ablation level 2).
    
    创建无门控的投影器（消融级别 2）。
    门控固定为 1.0 + 标量权重固定为 1.0，研究门控机制的贡献。
    
    Args:
        **kwargs: 透传给 create_ablation_projector 的参数
    
    Returns:
        AblationProjector: 无门控 + 无标量权重的投影器
    """
    return create_ablation_projector(ablation_level=2, **kwargs)


def create_source_only_projector(**kwargs) -> AblationProjector:
    """
    Create source-only projector (ablation level 3).
    
    创建仅使用 source 的投影器（消融级别 3）。
    无目标端 + 无门控 + 无标量权重，研究目标端 KV 信息的贡献。
    这等价于一个简单的跨模型 KV 投影，不做任何融合。
    
    Args:
        **kwargs: 透传给 create_ablation_projector 的参数
    
    Returns:
        AblationProjector: 仅使用 source 特征的投影器
    """
    return create_ablation_projector(ablation_level=3, **kwargs)


def create_no_gate_only_projector(**kwargs) -> AblationProjector:
    """
    Create projector without gates but with scalar weights and target (ablation level 4).
    
    创建仅无门控但保留标量权重和目标端的投影器（消融级别 4）。
    这是一个特殊的消融配置，用于单独研究门控机制的贡献，
    而保持标量权重和目标端残差不变。
    
    Args:
        **kwargs: 透传给 create_ablation_projector 的参数
    
    Returns:
        AblationProjector: 仅无门控的投影器（保留标量和目标端）
    """
    return create_ablation_projector(ablation_level=4, **kwargs)
