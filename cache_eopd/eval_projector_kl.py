"""
【C2C 核心】projector 验收：在「整段 response 位置」上测 student 分布逼近 teacher 的程度。

这是训练目标本身（不是末位单点 KL，那个判别力太低）。对每个样本：
    KL(student_fused ‖ teacher) over response positions
越小说明 student 在 fused KV 下的输出分布越接近 teacher。

对比四种设置，证明 projector 预训练有效：
    baseline   : student 不用任何 teacher 信息（自身 KV 前向）
    zero       : fused 但 projector 零初始化（融合恒等于 student 自身，应≈baseline）
    random     : fused 但 projector 随机初始化（未训练）
    pretrained : fused 且用训练后的 projector（应明显低于 baseline / random）

用法：
    /home/knhdu/anaconda3/envs/rosetta/bin/python -m cache_eopd.eval_projector_kl \
        --teacher ~/taopd-baseline/modelweights/Qwen3-4B \
        --student ~/taopd-baseline/modelweights/Qwen3-1.7B \
        --data-path ~/taopd-baseline/data/DAPO-Math-17k-dedup/dapo_math_17k_dedup_slime.jsonl \
        --device cuda:1 --teacher-device auto \
        --projector-path ./ckpt_projector/projector_step400.pt --num-samples 50
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cache_eopd.fused_kv import FusedKVBuilder, FusedKVConfig, load_projector_ckpt
from cache_eopd.train_projector import token_mean_kl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--data-field", default="prompt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--teacher-device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--min-len", type=int, default=24)
    p.add_argument("--prefix-ratio", type=float, default=0.5)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--holdout", type=int, default=200,
                   help="只用数据文件前 N 条（与 train_projector.py 的 --holdout 一致），"
                        "确保验收样本训练时没见过")
    p.add_argument("--projector-layers", type=int, default=3)
    p.add_argument("--projector-path", default=None)
    p.add_argument("--layer-mapping", choices=["last_aligned", "k_nearest", "relative_depth"],
                   default="relative_depth",
                   help="teacher→student 层映射；旧 projector checkpoint 使用 relative_depth")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(path, dtype, device, exclude=None):
    if device == "auto":
        mm = {i: ("1MiB" if i in exclude else "7GiB") for i in range(6)} if exclude else None
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="eager",
            device_map="auto", max_memory=mm).eval()
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    student = load_model(args.student, dtype, args.device)
    sg = int(args.device.split(":")[1]) if args.device.startswith("cuda:") else 0
    teacher = load_model(args.teacher, dtype, args.teacher_device,
                         {0, sg} if args.teacher_device == "auto" else None)
    tok = AutoTokenizer.from_pretrained(args.student)

    # ---- 读取并筛选样本 ----
    texts = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(str(obj.get(args.data_field, obj.get("prompt", ""))))
            except json.JSONDecodeError:
                texts.append(line)
    # 只取训练时被排除的前 holdout 条，保证是真正的 held-out
    texts = texts[: args.holdout]
    samples = [tok(t, return_tensors="pt", add_special_tokens=True)["input_ids"][0] for t in texts]
    samples = [s for s in samples if args.min_len <= s.numel() <= args.max_len]
    samples = samples[: args.num_samples]
    print(f"[eval] 用 {len(samples)} 条 held-out 样本，max-len={args.max_len}，prefix_ratio={args.prefix_ratio}")

    s_device = next(student.parameters()).device

    # ---- 构造各变体的 builder ----
    def make_builder(projector=None, zero=False):
        cfg = FusedKVConfig(
            dtype=dtype,
            zero_init=zero,
            projector_num_layers=args.projector_layers,
            layer_mapping_strategy=args.layer_mapping,
        )
        b = FusedKVBuilder.from_models(teacher, student, cfg, projector=projector)
        # zero/random 两个对照组没训练过，gate_logit=0 会让硬门控关掉融合、
        # 三者都退化成 student 自身，所以强制常开。pretrained 例外：它的门控是
        # 训练出来的（哪层该融合是学到的结论），必须原样保留。
        if projector is None:
            for p in b.projectors:
                p.use_gumbel = False
            b.set_gate_logit(3.0)
        return b

    builders = {
        "baseline": None,
        "zero": make_builder(zero=True),
        "random": make_builder(zero=False),
    }
    if args.projector_path:
        trained = load_projector_ckpt(args.projector_path)
        builders["pretrained"] = make_builder(projector=trained)

    for b in builders.values():
        if b is not None:
            b.freeze_teacher_student()
            b.projectors.eval()  # 验收用硬门控，与 rollout 推理一致

    tot = {k: 0.0 for k in builders}
    cnt = 0
    for ids in samples:
        T = ids.numel()
        P = max(1, int(T * args.prefix_ratio))
        R = T - 1 - P
        if R < 1:
            continue
        input_ids = ids.unsqueeze(0).to(s_device)
        am = torch.ones(1, T, dtype=torch.long, device=s_device)
        pos = torch.arange(T, device=s_device).unsqueeze(0)

        # teacher 全序列 logits（蒸馏目标）
        t_out = teacher(input_ids=input_ids, attention_mask=am, position_ids=pos,
                        use_cache=True, past_key_values=DynamicCache())
        tgt = t_out.logits[:, P:T - 1, :]  # (1, R, V)

        # baseline: student 自身前向（无 teacher 信息）
        s_self = student(input_ids=input_ids, attention_mask=am, position_ids=pos,
                         use_cache=True, past_key_values=DynamicCache())
        base_logits = s_self.logits[:, P:T - 1, :]
        tot["baseline"] += token_mean_kl(base_logits, tgt).item()

        # fused 变体
        for k, b in builders.items():
            if k == "baseline":
                continue
            fused, _ = b.build_trainable(input_ids, am, pos, P)
            resp = student(input_ids=input_ids[:, P:T - 1],
                           attention_mask=am[:, :T - 1],
                           position_ids=pos[:, P:T - 1],
                           past_key_values=fused, use_cache=True)
            tot[k] += token_mean_kl(resp.logits, tgt).item()
        cnt += 1

    print(f"\n[结果] 样本数={cnt}，指标 = per-token KL(student ‖ teacher) over response（nats，越小越好）")
    for k in builders:
        print(f"  {k:10s} : {tot[k] / max(1, cnt):.4f}")
    if "pretrained" in tot and cnt > 0:
        bl, rd, pr = tot["baseline"] / cnt, tot["random"] / cnt, tot["pretrained"] / cnt
        print(f"\n[判据]")
        print(f"  pretrained 相对 baseline 下降: {(bl - pr) / bl * 100:.1f}%  "
              f"(baseline={bl:.4f} → pretrained={pr:.4f})")
        print(f"  pretrained 相对 random    下降: {(rd - pr) / rd * 100:.1f}%  "
              f"(random={rd:.4f} → pretrained={pr:.4f})")
        ok = pr < bl and pr < rd
        print(f"  结论: {'PASS ✅ projector 预训练有效' if ok else 'FAIL ❌ 未观察到改善'}")


if __name__ == "__main__":
    main()
