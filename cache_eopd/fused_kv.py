"""
【C2C 核心】Fused-KV 构造模块 —— 路线 A（HF Rollout）的核心逻辑。

职责：给定同一个 prompt，
    1. teacher 前缀前向 → teacher KV-Cache（sharer cache）
    2. student 前缀前向 → student KV-Cache（base cache）
    3. C2CProjector 把 teacher KV 逐层投影到 student 维度并融合 → fused KV-Cache
    4. 返回 fused cache（以及可选的 teacher logits，供 Part II token-importance 使用）

参考实现：rosetta/model/wrapper.py 的 RosettaModel（Stage1 缓存 / Stage2 融合）。
无需 monkey-patch attention：融合发生在 generate 之前，fused cache 通过
past_key_values 传入，attention 自然使用它。

【与 C2C 原版对齐的三条约定】（第一轮训练因为违反它们导致 GSM8K 掉 6pp）：
  1. **每个 student 层一个独立 projector**（C2C script/train/SFT_train.py:618
     `num_projectors = slm_num_layers`，`projector_idx = target_layer_idx`）。
     28 层共用一个 MLP 只能学到"所有层的平均妥协"——第 0 层和第 27 层的 KV
     几何完全不同。
  2. **只融合前 L-1 个 token**（C2C rosetta/utils/evaluate.py:839-840：
     instruction_index 覆盖 L-1 个位置、response_index 是最后 1 个）。
     最后一位由 decode loop 首步用 student 自己的 KV 重算。
  3. **门控保持可学习**。C2C 的 key/value_gate_logit 是训练出来的，配合
     per-layer projector 就是 28 组独立的层级开关，模型自己决定哪层该收
     teacher 的信息。把它焊死成常开 = 无差别灌入，正是"把不该融合的融合了"。
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache

from rosetta.model.projector import C2CProjector, Projector, load_projector, save_projector


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

    # 【C2C 对齐】per-layer projector 后总参数 = 28 × 单个，hidden 1024 会到 ~1.4G，
    # 故默认降到 512（单个 ~13M ×28 ≈ 370M，bf16 约 0.7GB，可训）
    projector_hidden_dim: int = 512
    projector_intermediate_dim: int = 512
    projector_num_layers: int = 3
    projector_dropout: float = 0.0
    # 推理/验证阶段用 zero_init=False + 硬门控；训练 projector 时再打开 Gumbel 退火
    zero_init: bool = False
    dtype: torch.dtype = torch.bfloat16
    # 是否同时返回 teacher 对 prompt 的 logits（Part II token importance 需要）
    return_teacher_logits: bool = False
    # 【C2C 对齐 1】每个 student 层一个独立 projector（SFT_train.py:618）
    per_layer_projector: bool = True
    # 【C2C 对齐 2】末位 token 不融合，由 decode 首步用 student 自身 KV 重算
    #（evaluate.py:839-840 的 instruction_index / response_index 划分）
    keep_last_token_unfused: bool = True
    # 【阶段六】融合强度系数。projector 输出 = student + gate*w*proj(teacher)，
    # 这里再对「增量」乘一个 scale：fused = student + fusion_scale * (proj_out - student)。
    # scale=1.0 与原版完全等价；<1 削弱融合，>1 加强。用于「只保留纠偏效益、别把对的带歪」：
    # 在评测侧直接扫不同 scale 即可（无需重训），找让 step200 增益最大、代价最小的取值。
    fusion_scale: float = 1.0


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
        projectors: nn.ModuleList,
        layer_mapping: dict[int, int],
        config: FusedKVConfig,
    ):
        super().__init__()
        # teacher/student 不注册为子模块（避免被 FSDP/optimizer 误纳入），只持引用
        object.__setattr__(self, "_teacher_ref", [teacher])
        object.__setattr__(self, "_student_ref", [student])
        # 【C2C 对齐 1】per-layer 时长度 = student 层数；共享模式下长度为 1
        self.projectors = projectors
        self.layer_mapping = layer_mapping
        self.config = config

    @property
    def teacher(self) -> nn.Module:
        return self._teacher_ref[0]

    @property
    def student(self) -> nn.Module:
        return self._student_ref[0]

    def projector_for(self, s_layer: int) -> Projector:
        """取第 s_layer 层对应的 projector（共享模式下恒为第 0 个）。"""
        return self.projectors[s_layer] if len(self.projectors) > 1 else self.projectors[0]

    @property
    def projector(self) -> Projector:
        """兼容旧调用：取第 0 个。仅用于读超参/门控之类的全局操作。"""
        return self.projectors[0]

    def _proj_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        """取投影权重的 dtype/device。

        【坑】不能用 next(self.projectors.parameters())：门控标量被单独提到 fp32 后，
        参数遍历顺序不保证先命中投影权重，抓到 gate 就会把 KV 转成 fp32，撞上
        "mat1 and mat2 must have the same dtype"。这里显式取投影输入层的权重。
        """
        w = self.projectors[0].key_in.weight
        return w.dtype, w.device

    def gate_params_to_fp32(self) -> None:
        """把门控标量单独提到 fp32。

        【坑】projector 整体是 bf16，而 gate_logit 是个**标量**：bf16 只有 8 位尾数，
        1.0 附近的 ulp 是 2^-7 ≈ 0.0078。优化器每步给它的更新量在 1e-3 量级，
        `param += update` 直接被舍入回原值 —— 实测 30 步后 logit 精确停在 1.0000，
        看起来像"梯度为零"，其实是精度吞了更新，门控完全学不动。
        这两个标量只参与 sigmoid((logit+noise)/T) 和 (logit>0)，与 KV 张量的 dtype
        无关，提到 fp32 没有任何副作用。
        """
        for p in self.projectors:
            p.key_gate_logit.data = p.key_gate_logit.data.float()
            p.value_gate_logit.data = p.value_gate_logit.data.float()

    def set_gate_logit(self, value: float) -> None:
        """把所有 projector 的门控 logit 设为同一初值。

        C2CProjector 默认 gate_logit=0 → 推理时硬门控 (0>0)=False → 融合被静默
        全关。训练前设成正值（如 +1.0）保证一开始融合是打开的，之后由训练自己
        决定要不要关掉某些层——注意与「fill_(3.0) 焊死不训」不同，这里只是初值。
        """
        with torch.no_grad():
            for p in self.projectors:
                p.key_gate_logit.fill_(value)
                p.value_gate_logit.fill_(value)

    def update_temperature(self, step: int) -> None:
        """对所有 projector 同步 Gumbel 门控温度退火。"""
        for p in self.projectors:
            p.update_temperature(step)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_models(
        cls,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[FusedKVConfig] = None,
        projector: Optional[nn.Module] = None,
    ) -> "FusedKVBuilder":
        """从两个 HF 模型自动读取 KV 形状信息并构造 projector 与层映射。

        projector 可以是单个 Projector（共享模式）或 nn.ModuleList（per-layer，
        由 load_projector_ckpt 返回）；为 None 时按 config.per_layer_projector 新建。
        """
        config = config or FusedKVConfig()
        t_cfg, s_cfg = teacher.config, student.config

        # 【C2C 核心】KV 头数与 head_dim：GQA 模型 KV 头数是 num_key_value_heads 而非 num_attention_heads
        t_kv_heads = getattr(t_cfg, "num_key_value_heads", t_cfg.num_attention_heads)
        s_kv_heads = getattr(s_cfg, "num_key_value_heads", s_cfg.num_attention_heads)
        t_head_dim = getattr(t_cfg, "head_dim", None) or t_cfg.hidden_size // t_cfg.num_attention_heads
        s_head_dim = getattr(s_cfg, "head_dim", None) or s_cfg.hidden_size // s_cfg.num_attention_heads

        # 【C2C 对齐 1】per_layer=True 时每个 student 层各造一个 projector
        n_proj = s_cfg.num_hidden_layers if config.per_layer_projector else 1
        if projector is None:
            projectors = nn.ModuleList([
                C2CProjector(
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
                for _ in range(n_proj)
            ])
        elif isinstance(projector, nn.ModuleList):
            projectors = projector
        else:
            projectors = nn.ModuleList([projector])

        device = next(student.parameters()).device
        projectors = projectors.to(device=device, dtype=config.dtype)

        layer_mapping = build_layer_mapping(t_cfg.num_hidden_layers, s_cfg.num_hidden_layers)
        return cls(teacher, student, projectors, layer_mapping, config)

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

        【C2C 对齐 2】默认只融合前 L-1 个位置，末位保留 student 自身 KV
        （对应 evaluate.py:839-840 把最后 1 个 token 划给 response_index=-1）。
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
        L = input_ids.size(1)
        fuse_len = max(1, L - 1) if self.config.keep_last_token_unfused else L
        fused_cache = DynamicCache()
        proj_dtype, proj_device = self._proj_dtype_device()
        for s_layer, t_layer in self.layer_mapping.items():
            t_key, t_value = _get_layer_kv(teacher_cache, t_layer)   # (B, Ht, L, Dt_head)
            s_key, s_value = _get_layer_kv(student_cache, s_layer)   # (B, Hs, L, Ds_head)
            # teacher 可能在另一张卡上（显存分离部署），先搬到 projector/student 所在卡
            # C2CProjector.forward 约定输入 (B, H, N, D)（内部自行展平/转置）
            s_key_f = s_key[:, :, :fuse_len, :].to(device=proj_device, dtype=proj_dtype)
            s_val_f = s_value[:, :, :fuse_len, :].to(device=proj_device, dtype=proj_dtype)
            raw_key, raw_val = self.projector_for(s_layer)(
                (t_key[:, :, :fuse_len, :].to(device=proj_device, dtype=proj_dtype),
                 t_value[:, :, :fuse_len, :].to(device=proj_device, dtype=proj_dtype)),
                (s_key_f, s_val_f),
            )
            # 【阶段六】fused = student + fusion_scale * (proj_out - student)，
            # proj_out 本身已含 student，故 (proj_out - student) 就是「投影增量」。
            scale = self.config.fusion_scale
            fused_key = (s_key_f + scale * (raw_key - s_key_f)).to(dtype=s_key.dtype, device=s_key.device)
            fused_value = (s_val_f + scale * (raw_val - s_val_f)).to(dtype=s_value.dtype, device=s_value.device)
            if fuse_len < L:
                # 末位拼回 student 自身的 KV，保持 cache 长度仍为 L
                fused_key = torch.cat([fused_key, s_key[:, :, fuse_len:, :]], dim=2)
                fused_value = torch.cat([fused_value, s_value[:, :, fuse_len:, :]], dim=2)
            fused_cache.update(fused_key, fused_value, s_layer)

        extras = {
            "teacher_logits": t_out.logits if self.config.return_teacher_logits else None,
        }
        return fused_cache, extras


    # ------------------------------------------------------------------
    # 训练版：对 projector 可微的 fused 前缀 cache
    # ------------------------------------------------------------------
    def build_trainable(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        prefix_len: int,
        need_teacher_logits: bool = True,
    ) -> tuple[DynamicCache, Optional[torch.Tensor]]:
        """【C2C 核心】预训练 projector 时用的可微 fused 构造。

        与 rollout 用的 build() 不同：这里只让 *projector* 处于 autograd 图中，
        teacher/student 全程 no_grad（冻结），其 KV 当作常量 detach。反传路径为：
            loss → student_response_logits → fused_prefix_KV → projector
        这样反向传播只会更新 projector 的参数，符合『冻结 teacher/student，只训 projector』。

        Args:
            input_ids      : (B, T) 整段序列（含 prefix 与后续 response）
            attention_mask : (B, T)
            position_ids   : (B, T) 或 None
            prefix_len     : 前缀长度 P；fused cache 只覆盖 x_0..x_{P-1}，
                             后续 response（x_P..x_{T-2}）在训练循环里用 fused 前缀做 teacher-forcing
            need_teacher_logits:
                             True  = teacher 前向看整段（含 response），返回 logits 作蒸馏目标。
                             False = teacher **只看前缀**，不返回 logits。
                             【C2C 对齐】原版 sharer 只处理 instruction 段（kv_cache_index
                             把 response 标成 -1 不缓存），SFT-CE 训练不需要 teacher logits，
                             用 False 既省一半 teacher 算力，也避免 teacher KV 里混进
                             它自己看过答案后的信息（那会造成信息泄漏）。
        Returns:
            fused_cache   : DynamicCache，对 projector 可微。
                            **长度 = P-1**（keep_last_token_unfused=True 时）或 P。
                            用 trainable_cache_len(P) 取这个长度。
                            这里不像 build() 那样把末位 student KV 拼回来 —— 训练时
                            调用方要把 x_{P-1} 当成 decode 首步喂进去，与 rollout
                            的 crop_cache(cache, L-1) + 喂末位 token 完全同构。
            teacher_logits: (B, T, V) teacher 对整段序列的 logits（detach）；
                            need_teacher_logits=False 时为 None
        """
        t_device = next(self.teacher.parameters()).device
        s_device = next(self.student.parameters()).device
        pdtype, pdevice = self._proj_dtype_device()

        # (a)【C2C 核心】teacher 前向取 sharer cache，no_grad
        # SFT 口径下只喂前缀，避免 teacher 提前看到答案（信息泄漏）
        t_end = input_ids.size(1) if need_teacher_logits else prefix_len
        self.teacher.eval()
        with torch.no_grad():
            t_out = self.teacher(
                input_ids=input_ids[:, :t_end].to(t_device),
                attention_mask=attention_mask[:, :t_end].to(t_device),
                position_ids=position_ids[:, :t_end].to(t_device) if position_ids is not None else None,
                use_cache=True,
                past_key_values=DynamicCache(),
            )
        teacher_cache = t_out.past_key_values
        teacher_logits = t_out.logits.detach().to(s_device) if need_teacher_logits else None

        # (b) student 前缀前向（只取前缀 KV），no_grad
        # 【坑】这里会临时把 student 拨到 eval()；若调用方处于 train()（如
        # train_student_distill 训学生时），必须还原，否则后续 response 前向
        # 在 eval 下跑（dropout 关闭），与训练态不一致。
        was_training = self.student.training
        self.student.eval()
        with torch.no_grad():
            s_out = self.student(
                input_ids=input_ids[:, :prefix_len].to(s_device),
                attention_mask=attention_mask[:, :prefix_len].to(s_device),
                position_ids=position_ids[:, :prefix_len].to(s_device) if position_ids is not None else None,
                use_cache=True,
                past_key_values=DynamicCache(),
            )
        if was_training:
            self.student.train()
        student_cache = s_out.past_key_values

        # (c)【C2C 核心】逐层投影融合 —— 只有 projector 在 autograd 图中
        # 【C2C 对齐 2】与评测同口径：cache 只覆盖前 P-1 位，末位 x_{P-1} 交给调用方
        # 当 decode 首步重新前向（rollout 里 crop_cache(cache, L-1) 就是这么做的）。
        # 训练与推理的融合边界必须一致，否则学到的东西在 rollout 时错位一格。
        fuse_len = self.trainable_cache_len(prefix_len)
        fused_cache = DynamicCache()
        for s_layer, t_layer in self.layer_mapping.items():
            t_key, t_value = _get_layer_kv(teacher_cache, t_layer)
            s_key_full, s_value_full = _get_layer_kv(student_cache, s_layer)
            # 裁到融合长度，再搬到 projector 所在卡；teacher/student 冻结 → detach 当常量
            t_key = t_key[:, :, :fuse_len, :].contiguous().to(device=pdevice, dtype=pdtype).detach()
            t_value = t_value[:, :, :fuse_len, :].contiguous().to(device=pdevice, dtype=pdtype).detach()
            s_key = s_key_full[:, :, :fuse_len, :].contiguous().to(device=pdevice, dtype=pdtype).detach()
            s_value = s_value_full[:, :, :fuse_len, :].contiguous().to(device=pdevice, dtype=pdtype).detach()
            raw_key, raw_val = self.projector_for(s_layer)((t_key, t_value), (s_key, s_value))
            # 【阶段六】fused = student + fusion_scale * (proj_out - student)
            scale = self.config.fusion_scale
            fused_key = (s_key + scale * (raw_key - s_key)).to(dtype=s_key_full.dtype, device=s_key_full.device)
            fused_value = (s_value + scale * (raw_val - s_value)).to(dtype=s_value_full.dtype, device=s_value_full.device)
            fused_cache.update(fused_key, fused_value, s_layer)

        return fused_cache, teacher_logits

    def trainable_cache_len(self, prefix_len: int) -> int:
        """build_trainable 返回的 cache 长度：P-1（末位不融合）或 P。"""
        if self.config.keep_last_token_unfused:
            return max(1, prefix_len - 1)
        return prefix_len

    # ------------------------------------------------------------------
    # 工具：冻结 teacher/student，仅 projector 可训
    # ------------------------------------------------------------------
    def freeze_teacher_student(self) -> None:
        """冻结 teacher 与 student（不更新权重），projector 保持可训。"""
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        for p in self.student.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        self.student.eval()


def save_projector_ckpt(projectors, path: str) -> None:
    """保存 projector 的「构造参数 + 权重」，支持 per-layer 的 ModuleList。

    【坑】rosetta 的 save_projector/load_projector 只序列化 __init__ 参数（JSON），
    完全不保存 state_dict —— load 出来的是按原参数重新构造的新实例。若训练时用了
    zero_init=True，加载回来就是个全零 projector，训练成果静默丢失。
    这里额外存权重，并用 load_projector_ckpt 配套加载。

    【C2C 对齐 1】28 个 per-layer projector 结构完全同构，构造配置只存一份（JSON），
    权重按层顺序存成一个列表，加载时逐个复原。
    """
    plist = list(projectors) if isinstance(projectors, nn.ModuleList) else [projectors]
    save_projector(plist[0], path)  # 写 JSON 构造配置（各层同构，一份足够）
    torch.save(
        {
            "num_projectors": len(plist),
            "state_dicts": [
                {k: v.detach().cpu() for k, v in p.state_dict().items()} for p in plist
            ],
        },
        path + ".weights",
    )


def load_projector_ckpt(path: str) -> nn.ModuleList:
    """与 save_projector_ckpt 配套：先按 JSON 配置构造，再逐层灌入训练好的权重。

    返回 nn.ModuleList（长度 = 训练时的 projector 数），可直接传给
    FusedKVBuilder.from_models(..., projector=...)。旧格式（单个 state_dict）
    也能加载，包成长度 1 的 ModuleList。
    """
    blob = torch.load(path + ".weights", map_location="cpu")
    if not (isinstance(blob, dict) and "state_dicts" in blob):
        # 旧格式：整个文件就是一个 state_dict
        projector = load_projector(path)
        projector.load_state_dict(blob)
        return nn.ModuleList([projector])
    mods = []
    for state in blob["state_dicts"]:
        p = load_projector(path)
        p.load_state_dict(state)
        mods.append(p)
    return nn.ModuleList(mods)


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
