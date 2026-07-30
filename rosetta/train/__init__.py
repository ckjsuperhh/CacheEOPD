"""
Training utilities for RosettaModel
（RosettaModel 的训练工具模块）

本模块是 rosetta.train 包的入口文件（__init__.py），负责将训练相关的核心组件
从子模块中统一导出，方便外部通过 `from rosetta.train import ...` 一站式导入。

核心导出组件：
    - RosettaTrainer: C2C 框架的主训练器，封装了两阶段训练逻辑
        （stage1: sharer 产生 KV-Cache；stage2: receiver 基于融合 KV 生成）
    - ProjectorSaveCallback: 训练回调，用于在 checkpoint 时保存 Projector 权重
        （Projector 负责将 sharer 的 KV 维度投影到 receiver 兼容维度）
    - freeze_model_components: 冻结模型部分参数的工具函数，
        训练时通常冻结 sharer/receiver 的 LLM 主干，只训练 Projector 和 Fuser
    - ChatDataset / InstructCoderChatDataset: 数据集适配器，
        将原始对话数据转换为 C2C 训练所需的格式
    - RosettaDataCollator: 数据整理器（collator），负责将样本 batch 化并构造
        sharer 输入 / receiver 输入 / attention mask 等张量
    - create_instructcoder_dataset: 工厂函数，快捷创建 InstructCoder 格式数据集
    - setup_models: 模型初始化工具，负责加载 sharer、receiver、projector、fuser
        并配置它们的设备放置与参数冻结策略

子模块依赖关系：
    - .dataset_adapters: 数据集适配逻辑（ChatDataset、RosettaDataCollator 等）
    - .model_utils: 模型加载与配置工具（setup_models）
"""

# ===================== 数据集适配器导入 =====================
# 从 dataset_adapters 子模块导入数据集类和数据整理器
# - ChatDataset: 通用对话数据集适配器，将不同来源的对话数据统一为标准格式
# - RosettaDataCollator: 将单个样本组装成训练 batch，处理 padding、
#   attention mask 构造、以及 sharer/receiver 输入的分离
from .dataset_adapters import (
    ChatDataset,
    RosettaDataCollator,
)

# ===================== 模型工具函数导入 =====================
# 从 model_utils 子模块导入模型初始化函数
# - setup_models: 加载并配置 sharer 模型、receiver 模型、Projector 和 Fuser，
#   返回组装好的 RosettaModel 实例，同时处理设备分配和参数冻结
from .model_utils import setup_models

# ===================== 公共 API 列表 =====================
# __all__ 定义了 `from rosetta.train import *` 时导出的符号列表
__all__ = [
    # 主训练器：封装 C2C 两阶段训练循环（stage1 sharer 前向 → stage2 receiver 前向+loss）
    "RosettaTrainer",
    # 训练回调：在每个 checkpoint 保存时额外保存 Projector（投影层）的权重
    "ProjectorSaveCallback",
    # 参数冻结工具：选择性冻结 sharer/receiver 的 LLM 参数，只训练 Projector + Fuser
    "freeze_model_components",
    # InstructCoder 专用对话数据集适配器（适用于 InstructCoder 格式的指令-代码数据）
    "InstructCoderChatDataset",
    # 通用对话数据集适配器（适用于多种对话数据格式）
    "ChatDataset",
    # 数据整理器：将样本 batch 化，构造 sharer/receiver 的输入张量和 attention mask
    "RosettaDataCollator",
    # 工厂函数：快捷创建 InstructCoder 格式的数据集实例
    "create_instructcoder_dataset",
    # 模型初始化：加载 sharer、receiver、projector、fuser 并配置设备与冻结策略
    "setup_models"
] 