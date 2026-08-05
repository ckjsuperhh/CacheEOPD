"""Prepare a CacheEOPD fused-KV packet for a vLLM request."""

from __future__ import annotations

import argparse
import json

import torch

from cache_eopd.vllm_kv_packet import build_packet_from_models, request_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--input-ids", required=True, help="torch.save tensor with shape (1, prompt_len)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--projector-path", default=None)
    parser.add_argument("--fuser-dir", default=None)
    parser.add_argument("--layer-mapping", default="last_aligned")
    parser.add_argument("--fusion-scale", type=float, default=1.0)
    args = parser.parse_args()

    try:
        input_ids = torch.load(args.input_ids, map_location=args.device, weights_only=True)
    except TypeError:
        input_ids = torch.load(args.input_ids, map_location=args.device)
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("--input-ids must contain a torch.Tensor")
    path = build_packet_from_models(
        args.student,
        args.teacher,
        input_ids,
        device=args.device,
        teacher_device=args.teacher_device,
        projector_path=args.projector_path,
        fuser_dir=args.fuser_dir,
        layer_mapping=args.layer_mapping,
        fusion_scale=args.fusion_scale,
        path=args.output,
    )
    print(json.dumps(request_params(path), sort_keys=True))


if __name__ == "__main__":
    main()
