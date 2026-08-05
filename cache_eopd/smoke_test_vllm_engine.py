"""End-to-end vLLM engine smoke using a student self-KV packet."""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_eopd.fused_kv import _get_layer_kv
from cache_eopd.vllm_kv_packet import load_packet, request_params


def make_packet(model_path: str, packet_path: str, device: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    inputs = tokenizer("Compute 2 + 2. Answer briefly.", return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=inputs.attention_mask.to(device), use_cache=True)
    layers = []
    layer_index = 0
    while True:
        try:
            key, value = _get_layer_kv(output.past_key_values, layer_index)
        except (IndexError, KeyError, AttributeError):
            break
        layers.append({"key": key[0].cpu(), "value": value[0].cpu()})
        layer_index += 1
    packet = {
        "version": 1,
        "input_ids": input_ids[0].cpu().to(torch.int32),
        "prompt_len": int(input_ids.size(1)),
        "layers": layers,
    }
    torch.save(packet, packet_path)
    del model, output
    torch.cuda.empty_cache()
    return input_ids[0].cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--packet", default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cacheeopd-vllm-smoke-") as directory:
        if args.packet:
            packet_path = args.packet
            prompt_ids = load_packet(packet_path)["input_ids"].tolist()
        else:
            packet_path = os.path.join(directory, "packet.pt")
            prompt_ids = make_packet(args.model, packet_path, args.device)
        from vllm import LLM, SamplingParams
        from vllm.config.kv_transfer import KVTransferConfig

        transfer_config = KVTransferConfig(
            kv_connector="CacheEOPDConnector",
            kv_role="kv_both",
            kv_connector_module_path="cache_eopd.vllm_kv_connector",
        )
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=64,
            max_num_seqs=1,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.30,
            enforce_eager=True,
            kv_transfer_config=transfer_config,
            disable_hybrid_kv_cache_manager=True,
        )
        sampling_params = SamplingParams(
            max_tokens=4,
            temperature=0,
            extra_args={"kv_transfer_params": request_params(packet_path)},
        )
        outputs = llm.generate([{"prompt_token_ids": prompt_ids}], sampling_params)
        text = outputs[0].outputs[0].text
        print(f"VLLM_ENGINE_OK generated={text!r}")


if __name__ == "__main__":
    main()
