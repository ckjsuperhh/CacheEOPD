"""CPU smoke test for CacheEOPD's vLLM paged-cache mapping."""

from __future__ import annotations

import hashlib
import os
import tempfile
from types import SimpleNamespace

import torch

from cache_eopd.vllm_kv_connector import CacheEOPDConnector, CacheEOPDConnectorMetadata

try:
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
except ImportError:
    KVConnectorRole = None


def main() -> None:
    prompt_len = 5
    block_size = 4
    input_ids = torch.arange(prompt_len, dtype=torch.int32)
    layers = []
    for layer_index in range(2):
        key = torch.arange(2 * prompt_len * 3, dtype=torch.float32).reshape(2, prompt_len, 3) + layer_index * 1000
        layers.append({"key": key, "value": key + 100})
    packet = {"version": 1, "input_ids": input_ids, "prompt_len": prompt_len, "layers": layers}

    with tempfile.TemporaryDirectory() as directory:
        packet_path = os.path.join(directory, "packet.pt")
        torch.save(packet, packet_path)
        record = {
            "request_id": "smoke",
            "packet_path": packet_path,
            "prompt_len": prompt_len,
            "cached_len": prompt_len - 1,
            "input_ids_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
            "block_ids": [1, 3],
        }
        cache = {
            "model.layers.1.attn": torch.zeros(2, 4, block_size, 2, 3),
            "model.layers.0.attn": torch.zeros(2, 4, block_size, 2, 3),
        }
        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=block_size),
            kv_transfer_config=SimpleNamespace(),
        )
        role = KVConnectorRole.WORKER if KVConnectorRole is not None else "worker"
        connector = CacheEOPDConnector(config, role)
        connector.register_kv_caches(cache)
        connector.bind_connector_metadata(CacheEOPDConnectorMetadata([record]))
        connector.start_load_kv(None)

        for layer_index in range(2):
            target = cache[f"model.layers.{layer_index}.attn"]
            source = layers[layer_index]["key"]
            for token_index in range(prompt_len - 1):
                block_index, offset = divmod(token_index, block_size)
                assert torch.equal(target[0, [1, 3][block_index], offset], source[:, token_index])
        assert torch.count_nonzero(cache["model.layers.0.attn"][:, 3, 0]) == 0
    print("VLLM_CONNECTOR_OK")


if __name__ == "__main__":
    main()
