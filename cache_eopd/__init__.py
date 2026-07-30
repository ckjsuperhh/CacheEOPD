"""
cache_eopd: C2C(Cache-to-Cache) 融入 EOPD 学生 rollout 的实现包（路线 A：HF Rollout）。

模块组成:
    - fused_kv.py           : 核心模块。teacher/student 前缀前向 → Projector 投影融合 → fused KV
    - prototype_generate.py : 可单卡跑的最小原型脚本，验证 fused KV 形状与生成正确性

与仓库其他部分的关系:
    - rosetta/  : 从 C2C 仓库 vendor 进来的投影器实现（C2CProjector 等）
    - verl/     : EOPD 训练框架；后续把本包接入 verl/workers/rollout/hf_rollout.py
"""

from .fused_kv import FusedKVBuilder, build_layer_mapping

__all__ = ["FusedKVBuilder", "build_layer_mapping"]
