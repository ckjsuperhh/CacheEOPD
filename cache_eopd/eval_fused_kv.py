"""
【C2C 核心】fused-KV 的量化评测（不看文本，看数值）。

原型脚本只能肉眼看生成质量，这里给出可复现的数值指标，用于回答三个问题：

Q1 注入链路是否等价？
    self-KV 路径 vs 原生 generate：逐 token 一致率应为 1.0
Q2 zero_init 时融合是否恒等？
    fused(zero_init) vs self-KV：逐 token 一致率应为 1.0，KV 最大绝对差应为 0
Q3 融合到底改变了什么？
    - KV 层面：每层 fused vs student 的相对 L2 偏移 ‖fused-student‖/‖student‖
    - 分布层面：student 在 prompt 末位的 next-token 分布，融合前后的 KL 散度
    - 教师对齐：融合后 student 分布是否更接近 teacher 分布
      （C2C 的期望方向：KL(student_fused ‖ teacher) < KL(student ‖ teacher)）
      注意：projector 未训练时这个指标不会变好，它是训练后的验收指标。

用法:
    cd ~/CacheEOPD && PYTHONPATH=. python -m cache_eopd.eval_fused_kv \
        --student ~/taopd-baseline/modelweights/Qwen3-1.7B \
        --teacher ~/taopd-baseline/modelweights/Qwen3-4B \
        --device cuda:3 --teacher-device auto
"""

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cache_eopd.fused_kv import FusedKVBuilder, FusedKVConfig, _get_layer_kv
from cache_eopd.fused_kv import load_projector_ckpt

PROMPTS = [
    "What is 17 * 23? Think step by step.",
    "Name the capital of France.",
    "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
    "Explain why the sky appears blue in two sentences.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--teacher-device", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None, help="把指标写成 JSON 的路径")
    p.add_argument("--projector-path", default=None,
                   help="若提供，额外评估一个『预训练后』的 projector（验收 KL-to-teacher 是否下降）")
    p.add_argument("--layer-mapping", choices=["last_aligned", "k_nearest", "relative_depth"],
                   default="relative_depth",
                   help="teacher→student 层映射；旧 projector checkpoint 使用 relative_depth")
    return p.parse_args()


def load(path, dtype, device, exclude_gpus=None):
    if device == "auto":
        max_memory = None
        if exclude_gpus:
            max_memory = {i: ("1MiB" if i in exclude_gpus else "7GiB") for i in range(6)}
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="eager",
            device_map="auto", max_memory=max_memory).eval()
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()


@torch.no_grad()
def next_token_logits_with_cache(model, input_ids, attention_mask, position_ids, cache):
    """给定前缀 cache（含全部前缀），返回 prompt 末位的 next-token logits。

    做法：把 cache 裁到 L-1，再喂末 token —— 与 decode loop 首步完全一致。
    """
    L = input_ids.shape[1]
    cropped = DynamicCache()
    idx = 0
    while True:
        try:
            k, v = _get_layer_kv(cache, idx)
        except (IndexError, AttributeError):
            break
        if k is None:
            break
        cropped.update(k[:, :, : L - 1, :].contiguous(), v[:, :, : L - 1, :].contiguous(), idx)
        idx += 1
    out = model(
        input_ids=input_ids[:, -1:], attention_mask=attention_mask,
        position_ids=position_ids[:, -1:], past_key_values=cropped, use_cache=True,
    )
    return out.logits[:, -1, :].float()


@torch.no_grad()
def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.student)
    tok.padding_side = "left"
    student = load(args.student, dtype, args.device)
    # teacher 用 auto 分片时，避开 VLLM 卡(0) 与 student 卡，避免显存打架
    student_gpu = 0
    if args.device.startswith("cuda:"):
        student_gpu = int(args.device.split(":")[1])
    exclude = {0, student_gpu} if args.teacher_device == "auto" else None
    teacher = load(args.teacher, dtype, args.teacher_device or args.device, exclude_gpus=exclude)

    texts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                     tokenize=False, add_generation_prompt=True) for q in PROMPTS]
    enc = tok(texts, return_tensors="pt", padding=True).to(args.device)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

    # ---- student 自身 KV 与末位分布 ----
    s_cache = student(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=position_ids, use_cache=True,
                      past_key_values=DynamicCache()).past_key_values
    logits_self = next_token_logits_with_cache(
        student, input_ids, attention_mask, position_ids, s_cache)

    # ---- teacher 末位分布（词表可能不同，仅在同 tokenizer 时可比）----
    t_dev = next(teacher.parameters()).device
    t_logits = teacher(input_ids=input_ids.to(t_dev), attention_mask=attention_mask.to(t_dev),
                       position_ids=position_ids.to(t_dev)).logits[:, -1, :].float().to(args.device)

    metrics = {}

    # 构造待评估的 projector 集合：zero_init / random_init / (可选) 预训练后
    builders = {}
    builders["zero_init"] = FusedKVBuilder.from_models(
        teacher, student, FusedKVConfig(
            dtype=dtype, zero_init=True,
            layer_mapping_strategy=args.layer_mapping))
    builders["random_init"] = FusedKVBuilder.from_models(
        teacher, student, FusedKVConfig(
            dtype=dtype, zero_init=False,
            layer_mapping_strategy=args.layer_mapping))
    if args.projector_path:
        trained = load_projector_ckpt(args.projector_path)
        builders["pretrained"] = FusedKVBuilder.from_models(
            teacher, student, FusedKVConfig(
                dtype=dtype, zero_init=False,
                layer_mapping_strategy=args.layer_mapping),
            projector=trained)

    for tag, builder in builders.items():
        fused_cache, _ = builder.build(input_ids, attention_mask, position_ids)

        # 指标 1: 每层 KV 的相对 L2 偏移
        rel_k, rel_v, max_abs = [], [], 0.0
        for i in range(student.config.num_hidden_layers):
            fk, fv = _get_layer_kv(fused_cache, i)
            sk, sv = _get_layer_kv(s_cache, i)
            fk, fv, sk, sv = fk.float(), fv.float(), sk.float(), sv.float()
            rel_k.append(((fk - sk).norm() / sk.norm().clamp(min=1e-9)).item())
            rel_v.append(((fv - sv).norm() / sv.norm().clamp(min=1e-9)).item())
            max_abs = max(max_abs, (fk - sk).abs().max().item(), (fv - sv).abs().max().item())

        # 指标 2: 末位 next-token 分布的变化
        logits_fused = next_token_logits_with_cache(
            student, input_ids, attention_mask, position_ids, fused_cache)
        kl_shift = F.kl_div(F.log_softmax(logits_fused, -1),
                            F.log_softmax(logits_self, -1),
                            log_target=True, reduction="batchmean").item()
        agree = (logits_fused.argmax(-1) == logits_self.argmax(-1)).float().mean().item()

        # 指标 3: 与 teacher 的距离（同 tokenizer 才有意义）
        kl_to_teacher = None
        if t_logits.shape[-1] == logits_self.shape[-1]:
            kl_before = F.kl_div(F.log_softmax(logits_self, -1), F.log_softmax(t_logits, -1),
                                 log_target=True, reduction="batchmean").item()
            kl_after = F.kl_div(F.log_softmax(logits_fused, -1), F.log_softmax(t_logits, -1),
                                log_target=True, reduction="batchmean").item()
            kl_to_teacher = {"before": kl_before, "after": kl_after,
                             "improved": kl_after < kl_before}

        metrics[tag] = {
            "kv_rel_l2_key_mean": sum(rel_k) / len(rel_k),
            "kv_rel_l2_value_mean": sum(rel_v) / len(rel_v),
            "kv_max_abs_diff": max_abs,
            "next_token_kl_shift": kl_shift,
            "next_token_argmax_agreement": agree,
            "kl_to_teacher": kl_to_teacher,
        }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\n[判据]")
    z = metrics["zero_init"]
    print(f"  zero_init KV 最大绝对差 = {z['kv_max_abs_diff']:.3e} (应为 0)")
    print(f"  zero_init argmax 一致率 = {z['next_token_argmax_agreement']:.3f} (应为 1.000)")
    r = metrics["random_init"]
    print(f"  random_init KV 相对偏移 key={r['kv_rel_l2_key_mean']:.4f} "
          f"value={r['kv_rel_l2_value_mean']:.4f} (>0 说明融合确实生效)")
    if "pretrained" in metrics:
        pr = metrics["pretrained"]
        prk = pr["kl_to_teacher"]
        print(f"  [验收] pretrained KV 相对偏移 key={pr['kv_rel_l2_key_mean']:.4f} "
              f"value={pr['kv_rel_l2_value_mean']:.4f}")
        if prk is not None:
            arrow = "↓ 改善" if prk["improved"] else "↑ 未改善"
            print(f"  [验收] pretrained KL→teacher: before={prk['before']:.4f} "
                  f"after={prk['after']:.4f} {arrow}")
        else:
            print("  [验收] pretrained: 词表不一致，跳过 KL→teacher（需同 tokenizer）")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
