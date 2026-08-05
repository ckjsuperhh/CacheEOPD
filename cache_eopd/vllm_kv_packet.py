"""Serialization helpers for CacheEOPD fused KV packets.

The vLLM API accepts token ids, not a HuggingFace ``past_key_values`` object.
This module is the narrow bridge between the existing C2C builder and the
vLLM KV connector: one request produces one packet containing student-shaped
KV tensors and the connector copies that packet into vLLM's paged cache.

Packets are deliberately explicit and CPU-backed.  This first integration is
for correctness and smoke tests; a shared-memory transport can replace the
file transport without changing the connector metadata contract.
"""

from __future__ import annotations

import os
import hashlib
import tempfile
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from cache_eopd.fused_kv import FusedKVBuilder, _get_layer_kv


PACKET_VERSION = 1


def build_packet(
    builder: FusedKVBuilder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    path: str | None = None,
) -> str:
    """Build and atomically save a single-request fused KV packet.

    vLLM request-level KV transfer metadata is intentionally JSON-friendly, so
    only the packet path is sent through ``SamplingParams.extra_args``.  Batch
    size must be one: the request id is the packet's ownership boundary.
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("CacheEOPD vLLM packets currently require input_ids shape (1, prompt_len)")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")

    fused_cache, _ = builder.build(input_ids, attention_mask, position_ids)
    layers: list[dict[str, torch.Tensor]] = []
    layer_index = 0
    while True:
        try:
            key, value = _get_layer_kv(fused_cache, layer_index)
        except (IndexError, KeyError, AttributeError):
            break
        layers.append({"key": key[0].detach().cpu(), "value": value[0].detach().cpu()})
        layer_index += 1
        if layer_index > 512:
            raise RuntimeError("refusing to serialize more than 512 KV layers")
    if not layers:
        raise RuntimeError("fused KV builder returned an empty cache")

    packet: dict[str, Any] = {
        "version": PACKET_VERSION,
        "input_ids": input_ids[0].detach().cpu().to(torch.int32),
        "prompt_len": int(input_ids.size(1)),
        "layers": layers,
        "dtype": str(layers[0]["key"].dtype),
    }

    if path is None:
        fd, path = tempfile.mkstemp(prefix="cacheeopd-kv-", suffix=".pt")
        os.close(fd)
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(packet, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_packet(path: str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate a packet produced by :func:`build_packet`."""
    try:
        packet = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        packet = torch.load(path, map_location=map_location)
    if not isinstance(packet, dict) or packet.get("version") != PACKET_VERSION:
        raise ValueError(f"Unsupported CacheEOPD KV packet: {path}")
    if not isinstance(packet.get("layers"), list) or not packet["layers"]:
        raise ValueError(f"KV packet has no layers: {path}")
    prompt_len = int(packet.get("prompt_len", -1))
    input_ids = packet.get("input_ids")
    if input_ids is None or input_ids.ndim != 1 or input_ids.numel() != prompt_len:
        raise ValueError(f"KV packet has inconsistent prompt metadata: {path}")
    for index, layer in enumerate(packet["layers"]):
        if not isinstance(layer, dict) or "key" not in layer or "value" not in layer:
            raise ValueError(f"KV packet layer {index} is malformed: {path}")
        if layer["key"].shape != layer["value"].shape:
            raise ValueError(f"KV packet layer {index} key/value shapes differ: {path}")
        if layer["key"].ndim != 3 or layer["key"].shape[1] != prompt_len:
            raise ValueError(f"KV packet layer {index} has an unsupported shape: {path}")
    return packet


def request_params(packet_path: str) -> dict[str, Any]:
    """Return the JSON-friendly vLLM request metadata for a packet."""
    packet = load_packet(packet_path)
    token_hash = hashlib.sha256(packet["input_ids"].numpy().tobytes()).hexdigest()
    return {
        "packet_path": os.path.abspath(packet_path),
        "prompt_len": int(packet["prompt_len"]),
        "cached_len": max(0, int(packet["prompt_len"]) - 1),
        "packet_version": PACKET_VERSION,
        "input_ids_sha256": token_hash,
    }


@torch.no_grad()
def build_packet_from_models(
    student_path: str,
    teacher_path: str,
    input_ids: torch.Tensor,
    *,
    device: str = "cuda",
    teacher_device: str | None = None,
    projector_path: str | None = None,
    fuser_dir: str | None = None,
    layer_mapping: str = "last_aligned",
    fusion_scale: float = 1.0,
    path: str | None = None,
) -> str:
    """Load HF models, construct C2C KV, and save one vLLM packet."""
    if projector_path and fuser_dir:
        raise ValueError("projector_path and fuser_dir are mutually exclusive")
    input_ids = input_ids.to(device)
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    if teacher_device == "auto":
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        ).eval()
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(teacher_device or device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    from cache_eopd.fused_kv import FusedKVConfig, load_official_fuser_projectors, load_projector_ckpt

    projectors = None
    if projector_path:
        projectors = load_projector_ckpt(projector_path)
    elif fuser_dir:
        projectors = load_official_fuser_projectors(fuser_dir)
    builder = FusedKVBuilder.from_models(
        teacher,
        student,
        FusedKVConfig(
            dtype=torch.bfloat16,
            layer_mapping_strategy=layer_mapping,
            fusion_scale=fusion_scale,
        ),
        projector=projectors,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    return build_packet(builder, input_ids, attention_mask, position_ids, path=path)
