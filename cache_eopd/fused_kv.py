"""
【C2C 核心】Fused-KV 构造模块 —— 路线 A（HF Rollout）的核心逻辑。

职责：给定同一个 prompt，
    1. teacher 前缀前向 → teacher KV-Cache（sharer cache）
    2. student 前缀前向 → student KV-Cache（base cache）
    3. C2CProjector 把 teacher KV 逐层投影到 student 维度并融合 → fused KV-Cache
    4. 返回 fused cache（以及可选的 teacher logits，供 Part II token-importance 使用）

参考实现：rosetta/model/wrapper.py 的 RosettaModel（Stage1 缓存 / Stage2 融合），
但去掉了它的分段(instruction/section)控制流——EOPD rollout 里 prompt 是一整段，
直接做一次前缀前向即可，无需 monkey-patch attention（融合发生在 generate 之前，
fused cache 通过 past_key_values 传入，attention 自然使用它）。
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache

from rosetta.model.projector import C2CProjector, Projector


def build_layer_mapping(num_teacher_layers: int, num_student_layers: int) -> dict[int, int]:
    """构造 student 层 → teacher 层的映射。

    【C2C 核心】teacher/student 层数不同时（如 Qwen3-4B 36 层 → Qwen3-1.7B 28 层），
    每个 student 层取 teacher 中相对深度最接近的层（与 C2C 论文的均匀映射一致）。
    层数相同时退化为恒等映射。
    """
    if num_teacher_layers == num_student_layers:
        return {i: i for i in range(num_student_layers)}
    # 相对深度对齐: student 第 i 层 (深度 i/S) ← teacher 第 round(i/S * T) 层
    mapping = {}
    for i in range(num_student_layers):
        j = round(i * (num_teacher_layers - 1) / max(num_student_layers - 1, 1))
        mapping[i] = j
    return mapping


@dataclass
class FusedKVConfig:
    """fused-KV 构造配置。字段与 C2CProjector 的关键超参对应。"""

    projector_hidden_dim: int = 1024
    projector_intermediate_dim: int = 1024
    projector_num_layers: int = 3
    projector_dropout: float = 0.0
    # 推理/验证阶段用 zero_init=False + 硬门控；训练 projector 时再打开 Gumbel 退火
    zero_init: bool = False
    dtype: torch.dtype = torch.bfloat16
    # 是否同时返回 teacher 对 prompt 的 logits（Part II token importance 需要）
    return_teacher_logits: bool = False


class FusedKVBuilder(nn.Module):
    """【C2C 核心】把 teacher KV 投影融合进 student KV 的构造器。

    用法（rollout 侧）:
        builder = FusedKVBuilder.from_models(teacher, student, cfg)
        fused_cache, extras = builder.build(input_ids, attention_mask, position_ids)
        output = student.generate(input_ids=..., past_key_values=fused_cache, ...)

    注意:
        - teacher 前向永远 no_grad + eval（rollout 阶段不训练 teacher）。
        - projector 是本模块唯一含可训练参数的部分；idea 验证阶段可随机初始化/零初始化，
          正式训练时应加载 C2C 预训练的 projector 权重或与 EOPD 联合训练。
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        projector: Projector,
        layer_mapping: dict[int, int],
        config: FusedKVConfig,
    ):
        super().__init__()
        # teacher/student 不注册为子模块（避免被 FSDP/optimizer 误纳入），只持引用
        object.__setattr__(self, "_teacher_ref", [teacher])
        object.__setattr__(self, "_student_ref", [student])
        self.projector = projector
        self.layer_mapping = layer_mapping
        self.config = config

    @property
    def teacher(self) -> nn.Module:
        return self._teacher_ref[0]

    @property
    def student(self) -> nn.Module:
        return self._student_ref[0]

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_models(
        cls,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[FusedKVConfig] = None,
        projector: Optional[Projector] = None,
    ) -> "FusedKVBuilder":
        """从两个 HF 模型自动读取 KV 形状信息并构造 projector 与层映射。"""
        config = config or FusedKVConfig()
        t_cfg, s_cfg = teacher.config, student.config

        # 【C2C 核心】KV 头数与 head_dim：GQA 模型 KV 头数是 num_key_value_heads 而非 num_attention_heads
        t_kv_heads = getattr(t_cfg, "num_key_value_heads", t_cfg.num_attention_heads)
        s_kv_heads = getattr(s_cfg, "num_key_value_heads", s_cfg.num_attention_heads)
        t_head_dim = getattr(t_cfg, "head_dim", None) or t_cfg.hidden_size // t_cfg.num_attention_heads
        s_head_dim = getattr(s_cfg, "head_dim", None) or s_cfg.hidden_size // s_cfg.num_attention_heads

        if projector is None:
            projector = C2CProjector(
                source_dim=t_head_dim,
                target_dim=s_head_dim,
                source_num_heads=t_kv_heads,
                target_num_heads=s_kv_heads,
                hidden_dim=config.projector_hidden_dim,
                intermediate_dim=config.projector_intermediate_dim,
                num_layers=config.projector_num_layers,
                dropout=config.projector_dropout,
                dtype=config.dtype,
                zero_init=config.zero_init,
            )
        device = next(student.parameters()).device
        projector = projector.to(device=device, dtype=config.dtype)

        layer_mapping = build_layer_mapping(t_cfg.num_hidden_layers, s_cfg.num_hidden_layers)
        return cls(teacher, student, projector, layer_mapping, config)

    # ------------------------------------------------------------------
    # 核心：构造 fused KV
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[DynamicCache, dict]:
        """对 prompt 前缀构造 fused KV-Cache。

        Args:
            input_ids      : (B, L) left-padded prompt token ids
            attention_mask : (B, L)
            position_ids   : (B, L)，可为 None（由模型内部推导）
        Returns:
            fused_cache : DynamicCache，student 维度，包含融合后的前缀 KV
            extras      : {"teacher_logits": (B, L, V_t) 或 None}
        """
        # ---- (a)【C2C 核心】teacher 前缀前向，取 sharer cache ----
        self.teacher.eval()
        t_device = next(self.teacher.parameters()).device
        t_out = self.teacher(
            input_ids=input_ids.to(t_device),
            attention_mask=attention_mask.to(t_device),
            position_ids=position_ids.to(t_device) if position_ids is not None else None,
            use_cache=True,
            past_key_values=DynamicCache(),
        )
        teacher_cache = t_out.past_key_values

        # ---- (b)【C2C 核心】student 前缀前向，取 base cache ----
        was_training = self.student.training
        self.student.eval()
        s_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=DynamicCache(),
        )
        student_cache = s_out.past_key_values
        if was_training:
            self.student.train()

        # ---- (c)【C2C 核心】逐层投影融合: fused = student + gate * w * proj(teacher) ----
        fused_cache = DynamicCache()
        proj_param = next(self.projector.parameters())
        proj_dtype, proj_device = proj_param.dtype, proj_param.device
        for s_layer, t_layer in self.layer_mapping.items():
            t_key, t_value = _get_layer_kv(teacher_cache, t_layer)   # (B, Ht, L, Dt_head)
            s_key, s_value = _get_layer_kv(student_cache, s_layer)   # (B, Hs, L, Ds_head)
            # teacher 可能在另一张卡上（显存分离部署），先搬到 projector/student 所在卡
            # C2CProjector.forward 约定输入 (B, H, N, D)（内部自行展平/转置）
            fused_key, fused_value = self.projector(
                (t_key.to(device=proj_device, dtype=proj_dtype),
                 t_value.to(device=proj_device, dtype=proj_dtype)),
                (s_key.to(device=proj_device, dtype=proj_dtype),
                 s_value.to(device=proj_device, dtype=proj_dtype)),
            )
            fused_cache.update(fused_key.to(s_key.dtype), fused_value.to(s_value.dtype), s_layer)

        extras = {
            "teacher_logits": t_out.logits if self.config.return_teacher_logits else None,
        }
        return fused_cache, extras


def _get_layer_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """兼容不同 transformers 版本的 DynamicCache 取层接口。

    旧版: cache.key_cache[i] / cache.value_cache[i]
    新版(>=4.54): cache.layers[i].keys / .values；同时 cache[i] 均返回 (key, value)
    """
    try:
        key, value = cache[layer_idx]
        return key, value
    except (TypeError, IndexError, KeyError):
        pass
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    raise TypeError(f"Unsupported cache type: {type(cache)}")
