"""vLLM V1 connector for precomputed CacheEOPD fused KV packets.

This connector implements the vLLM KV-transfer contract rather than passing a
HuggingFace cache through an incompatible API.  The scheduler advertises the
first ``L-1`` packet tokens as already-computed; the worker writes those tokens
into the allocated paged KV blocks before vLLM evaluates the final prompt token.

The packet provider is intentionally separate from this file.  That keeps the
worker-side cache layout code testable and leaves room for the next step:
generating teacher KV and projecting it inside the vLLM worker after the
student prefill, with no change to request metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import os
import re
from typing import Any

import torch

try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
    )
except ImportError:
    try:
        from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase_V1

        class KVConnectorMetadata:
            pass
    except ImportError:
        class KVConnectorBase_V1:
            pass

        class KVConnectorMetadata:
            pass

from cache_eopd.vllm_kv_packet import load_packet


logger = logging.getLogger(__name__)


@dataclass
class CacheEOPDConnectorMetadata(KVConnectorMetadata):
    requests: list[dict[str, Any]] = field(default_factory=list)


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _packet_params(request: Any) -> dict[str, Any] | None:
    sampling_params = _get_attr(request, "sampling_params")
    extra_args = _get_attr(sampling_params, "extra_args", default={}) or {}
    params = extra_args.get("kv_transfer_params")
    if params is None:
        params = extra_args.get("cache_eopd")
    if params is None:
        params = _get_attr(request, "kv_transfer_params")
    if not isinstance(params, dict) or not params.get("packet_path"):
        return None
    return params


def _request_id(request: Any) -> str:
    value = _get_attr(request, "request_id", "req_id", "id")
    if value is None:
        raise ValueError("vLLM request has no request id")
    return str(value)


def _block_ids(blocks: Any) -> list[int]:
    if blocks is None:
        return []
    if isinstance(blocks, torch.Tensor):
        blocks = blocks.detach().cpu().tolist()
    while isinstance(blocks, (list, tuple)) and blocks and isinstance(blocks[0], (list, tuple)):
        blocks = blocks[0]
    return [int(block) for block in blocks]


class CacheEOPDConnector(KVConnectorBase_V1):
    """Synchronous single-request packet connector for vLLM V1."""

    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None):
        try:
            super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        except TypeError:
            try:
                super().__init__(vllm_config, role)
            except TypeError:
                try:
                    super().__init__(vllm_config)
                except TypeError:
                    super().__init__()
        self.vllm_config = vllm_config
        self.block_size = int(getattr(getattr(vllm_config, "cache_config", None), "block_size", 16))
        self.kv_caches: dict[str, torch.Tensor] = {}
        self._request_state: dict[str, dict[str, Any]] = {}
        self._metadata = CacheEOPDConnectorMetadata([])

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = kv_caches

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> int:
        params = _packet_params(request)
        if params is None:
            return 0, False
        prompt_len = int(params.get("prompt_len", 0))
        if prompt_len <= 1:
            return 0, False
        aligned_cached_len = (prompt_len - 1) // self.block_size * self.block_size
        external_tokens = aligned_cached_len - int(num_computed_tokens)
        return max(0, external_tokens), False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int = 0, *args, **kwargs
    ) -> None:
        params = _packet_params(request)
        if params is None:
            return
        request_id = _request_id(request)
        if num_external_tokens <= 0:
            return
        self._request_state[request_id] = {
            "request_id": request_id,
            "packet_path": os.path.abspath(str(params["packet_path"])),
            "prompt_len": int(params["prompt_len"]),
            "cached_len": int(num_external_tokens),
            "input_ids_sha256": params.get("input_ids_sha256"),
        }

    def build_connector_meta(self, scheduler_output: Any) -> CacheEOPDConnectorMetadata:
        records = []
        for request in _get_attr(scheduler_output, "scheduled_new_reqs", default=[]) or []:
            request_id = _request_id(request)
            state = self._request_state.pop(request_id, None)
            if state is None:
                continue
            state["block_ids"] = _block_ids(_get_attr(request, "block_ids", default=[]))
            records.append(state)
        return CacheEOPDConnectorMetadata(records)

    def bind_connector_metadata(self, metadata: Any) -> None:
        try:
            super().bind_connector_metadata(metadata)
        except AttributeError:
            pass
        if isinstance(metadata, CacheEOPDConnectorMetadata):
            self._metadata = metadata
        elif isinstance(metadata, dict):
            self._metadata = CacheEOPDConnectorMetadata(list(metadata.get("requests", [])))
        else:
            raise TypeError(f"Unsupported CacheEOPD connector metadata: {type(metadata)}")

    set_connector_metadata = bind_connector_metadata

    def start_load_kv(self, forward_context: Any, *args, **kwargs) -> None:
        if not self._metadata.requests:
            return
        for record in self._metadata.requests:
            self._inject_record(record)
        self._metadata = CacheEOPDConnectorMetadata([])

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, *args, **kwargs) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def request_finished(self, request: Any, block_ids: Any = None, *args, **kwargs) -> tuple[bool, None]:
        self._request_state.pop(_request_id(request), None)
        return False, None

    def _inject_record(self, record: dict[str, Any]) -> None:
        packet = load_packet(str(record["packet_path"]), map_location="cpu")
        prompt_len = int(record["prompt_len"])
        cached_len = int(record.get("cached_len", max(0, prompt_len - 1)))
        if int(packet["prompt_len"]) != prompt_len:
            raise ValueError("CacheEOPD packet prompt length disagrees with request metadata")
        expected_hash = record.get("input_ids_sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(packet["input_ids"].numpy().tobytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError("CacheEOPD packet token hash disagrees with request metadata")
        if cached_len < 0 or cached_len > prompt_len:
            raise ValueError("CacheEOPD packet cached length is invalid")
        block_ids = _block_ids(record.get("block_ids"))
        if len(block_ids) * self.block_size < cached_len:
            raise ValueError(
                f"CacheEOPD packet needs {cached_len} slots but only "
                f"{len(block_ids) * self.block_size} were allocated"
            )
        def layer_key(item: tuple[str, torch.Tensor]) -> tuple[int, str]:
            match = re.search(r"(?:layers?|blocks?)\.(\d+)", item[0])
            return (int(match.group(1)) if match else 10**9, item[0])

        ordered_caches = sorted(self.kv_caches.items(), key=layer_key)
        layers = packet["layers"]
        if len(ordered_caches) < len(layers):
            raise ValueError(f"vLLM exposes {len(ordered_caches)} KV layers, packet has {len(layers)}")
        for layer_index, layer in enumerate(layers):
            self._write_layer(ordered_caches[layer_index][1], layer, block_ids, cached_len)

    def _write_layer(
        self,
        cache: torch.Tensor,
        layer: dict[str, torch.Tensor],
        block_ids: list[int],
        num_tokens: int,
    ) -> None:
        key = layer["key"]
        value = layer["value"]
        if key.shape[1] < num_tokens or key.shape != value.shape:
            raise ValueError("CacheEOPD packet layer shape does not match prompt length")
        if cache.ndim != 5 or cache.shape[0] != 2:
            raise ValueError(
                "Unsupported vLLM KV cache layout; expected "
                "[2, num_blocks, block_size, num_kv_heads, head_dim]"
            )
        target = cache
        key = key.to(device=target.device, dtype=target.dtype)
        value = value.to(device=target.device, dtype=target.dtype)
        for token_index in range(num_tokens):
            block_index, offset = divmod(token_index, self.block_size)
            block_id = block_ids[block_index]
            target[0, block_id, offset].copy_(key[:, token_index, :])
            target[1, block_id, offset].copy_(value[:, token_index, :])


__all__ = ["CacheEOPDConnector", "CacheEOPDConnectorMetadata"]
