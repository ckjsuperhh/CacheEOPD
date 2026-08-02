"""
【C2C 核心】下游验收：把预训练 projector 接进 rollout，测数学题正确率。

为什么需要这个脚本：
    eval_projector_kl.py 证明的是「student 在 fused KV 下的分布更接近 teacher」，
    但那个指标恰好就是训练目标本身——它只能说明优化成功了，不能说明这件事有用。
    projector 完全可能只学到 teacher 的说话风格/置信度校准，把 KL 拉低却没搬运
    任何解题知识。真正的硬证据是下游任务指标，也就是这里测的正确率。

三组对照：
    baseline : student 原生 generate（不接触 teacher）
    zero     : 零初始化 projector 融合（数学上恒等于 baseline）
               —— 这是**harness 自检**，不是实验组。它必须与 baseline 几乎一致，
               否则说明带 cache 的 decode 路径本身有 bug，pretrained 的数字就不可信。
    pretrained: 训练后的 projector 融合（实验组）

用法（apex-llm）：
    PYTHONPATH=/home/kejiechen/CacheEOPD \
    /home/knhdu/anaconda3/envs/rosetta/bin/python -m cache_eopd.eval_math_acc \
        --teacher /home/kejiechen/taopd-baseline/modelweights/Qwen3-4B \
        --student /home/kejiechen/taopd-baseline/modelweights/Qwen3-1.7B \
        --data-path /home/kejiechen/taopd-baseline/data/GSM8K-COT/gsm8k_cot_slime_300_seed41717.jsonl \
        --device cuda:1 --teacher-device auto \
        --projector-path ./ckpt_projector_v5/projector_final.pt \
        --num-samples 150 --max-new-tokens 512
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cache_eopd.fused_kv import (
    FusedKVBuilder,
    FusedKVConfig,
    _get_layer_kv,
    load_projector_ckpt,
)
from cache_eopd.train_student_distill import (
    load_official_fuser_projectors,
    load_teacher_sharded,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--teacher-device", default="auto")
    p.add_argument("--teacher-gpus", default=None,
                   help="--teacher-device auto 时允许 teacher 占用的卡，逗号分隔（如 '4,5'）。"
                        "不填则自动排除 GPU0(VLLM) 与 student 卡 —— 但那会把被别的作业"
                        "占满的卡也算进来导致 OOM，卡紧张时务必显式指定。")
    p.add_argument("--teacher-mem-per-gpu", default="7GiB")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--projector-path", default=None)
    p.add_argument("--fuser-dir", default=None,
                   help="官方 C2C fuser 的 final 目录（含 projector_{idx}.pt/.json）")
    p.add_argument("--num-samples", type=int, default=150)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--chat", action="store_true", default=True,
                   help="用 chat template 包装 prompt（Qwen3 是 instruct 模型，默认开）")
    p.add_argument("--no-chat", dest="chat", action="store_false")
    p.add_argument("--sanity", type=int, default=3,
                   help="先用 N 条样本自检：zero 融合的生成必须与原生 generate 一致（0=跳过）")
    p.add_argument("--out", default=None, help="逐条结果写入 jsonl，便于事后人工核查")
    p.add_argument("--fusion-scale", type=float, default=1.0,
                   help="融合强度系数：fused = student + scale*(proj_out - student)。"
                        "1.0 等价原版；<1 削弱融合、>1 加强。可在不重训的前提下直接扫，"
                        "找『纠偏效益最大、把对的带歪最小』的取值。")
    p.add_argument("--layer-mapping", choices=["last_aligned", "k_nearest", "relative_depth"],
                   default="relative_depth",
                   help="teacher→student 层映射；旧 projector checkpoint 使用 relative_depth")
    return p.parse_args()


# ----------------------------------------------------------------------
# 答案抽取与比对
# ----------------------------------------------------------------------
def extract_boxed(text: str):
    """抽取最后一个 \\boxed{...} 的内容（需要括号配对，不能用正则贪婪匹配）。"""
    key = r"\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    buf = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    return "".join(buf).strip() if depth == 0 else None


def extract_answer(text: str):
    """按 boxed → 'Answer:' → 最后一个数字 的优先级抽取模型答案。"""
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    m = re.findall(r"Answer:\s*\$?([^\n$]+)", text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1] if nums else None


def normalize(ans):
    """数值归一化：去掉 $ , % 空格与末尾 .0，使 '1,200' / '1200.0' / '$1200' 等价。"""
    if ans is None:
        return None
    s = str(ans).strip().strip("$").replace(",", "").replace(" ", "").rstrip("%")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.lower()


def is_correct(pred, label):
    p, g = normalize(pred), normalize(label)
    return p is not None and p == g


# ----------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------
def crop_cache(cache, length):
    """把 DynamicCache 裁到前 length 个位置（末位由 decode loop 首步重算）。"""
    new_cache = DynamicCache()
    idx = 0
    while True:
        try:
            k, v = _get_layer_kv(cache, idx)
        except (IndexError, AttributeError, KeyError):
            break
        if k is None:
            break
        new_cache.update(k[:, :, :length, :].contiguous(),
                         v[:, :, :length, :].contiguous(), idx)
        idx += 1
    return new_cache


@torch.no_grad()
def greedy_decode_with_cache(student, cache, input_ids, attention_mask,
                             position_ids, max_new_tokens, eos_ids):
    """带外部前缀 KV 的 greedy decode。与 c2c_hf_rollout._decode_with_cache 同构，
    这里独立实现以避免引入 verl 依赖。"""
    L = input_ids.size(1)
    cache = crop_cache(cache, L - 1)
    cur = input_ids[:, -1:]
    cur_pos = position_ids[:, -1:]
    mask = attention_mask
    out_toks = []
    for _ in range(max_new_tokens):
        out = student(input_ids=cur, attention_mask=mask, position_ids=cur_pos,
                      past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        tok = nxt.item()
        out_toks.append(tok)
        if tok in eos_ids:
            break
        cur = nxt
        cur_pos = cur_pos + 1
        mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype, device=mask.device)], dim=-1)
    return out_toks


@torch.no_grad()
def baseline_generate(student, input_ids, attention_mask, max_new_tokens, eos_ids, pad_id):
    """原生 greedy generate，作为不接触 teacher 的对照。"""
    out = student.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=pad_id, eos_token_id=list(eos_ids),
    )
    return out[0, input_ids.size(1):].tolist()


def main():
    args = parse_args()
    if args.projector_path and args.fuser_dir:
        raise ValueError("--projector-path 与 --fuser-dir 只能二选一")
    dtype = getattr(torch, args.dtype)

    def load(path, device, allow=None):
        if device == "auto":
            n = torch.cuda.device_count()
            mm = {i: (args.teacher_mem_per_gpu if i in allow else "1MiB")
                  for i in range(n)} if allow else None
            return AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, attn_implementation="eager",
                device_map="auto", max_memory=mm).eval()
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()

    student = load(args.student, args.device)
    sg = int(args.device.split(":")[1]) if args.device.startswith("cuda:") else 0
    allow = None
    if args.teacher_device == "auto":
        allow = ({int(x) for x in args.teacher_gpus.split(",")} if args.teacher_gpus
                 else set(range(torch.cuda.device_count())) - {0, sg})
        print(f"[load] teacher 分片可用卡 = {sorted(allow)}", flush=True)
    if args.teacher_device == "auto":
        teacher_devices = sorted(allow or set())
        teacher = load_teacher_sharded(args.teacher, dtype, teacher_devices, attn_impl="eager")
    else:
        teacher = load(args.teacher, args.teacher_device, allow)
    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    s_device = next(student.parameters()).device

    eos_ids = {tok.eos_token_id}
    for extra in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(extra)
        if tid is not None and tid >= 0:
            eos_ids.add(tid)

    # ---- 数据 ----
    rows = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[: args.num_samples]
    print(f"[data] {len(rows)} 题 | chat_template={args.chat} | "
          f"max_new_tokens={args.max_new_tokens}", flush=True)

    def encode(prompt_text):
        if args.chat:
            msgs = [{"role": "user", "content": prompt_text}]
            # Qwen3 默认开启 thinking 模式会产生超长 <think> 块，这里关掉
            try:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        else:
            text = prompt_text
        ids = tok(text, return_tensors="pt", add_special_tokens=not args.chat)["input_ids"]
        return ids[:, -args.max_prompt_len:].to(s_device)

    # ---- builder ----
    def make_builder(projector=None, zero=False):
        cfg = FusedKVConfig(
            dtype=dtype,
            zero_init=zero,
            fusion_scale=args.fusion_scale,
            layer_mapping_strategy=args.layer_mapping,
        )
        b = FusedKVBuilder.from_models(teacher, student, cfg, projector=projector)
        b.freeze_teacher_student()
        b.projectors.eval()  # 硬门控，与 rollout 推理一致
        if zero:
            # zero 组是 harness 自检：投影输出恒为 0，必须让门控开着才真的走完
            # 融合路径（fused = target + gate*w*0 == target），否则自检形同虚设。
            b.set_gate_logit(3.0)
        return b

    def fused_gen(builder, ids, am, pos, max_new=None):
        cache, _ = builder.build(ids, am, pos)
        return greedy_decode_with_cache(student, cache, ids, am, pos,
                                        max_new or args.max_new_tokens, eos_ids)

    # ---- harness 自检 ----
    # 三路对比，把「融合机制是否正确」与「decode 路径的 bf16 数值噪声」分离开：
    #   A 原生 generate
    #   B 手写 decode loop + student 自身 KV（不过 projector）
    #   C 手写 decode loop + zero-init 融合 KV（过 projector，但投影输出恒为 0）
    # B vs C 必须**逐 token 完全一致**——两者唯一差别是 projector，而零初始化下
    #   fused = target + gate*scalar*0 = target，是严格恒等。不一致 ⇒ 融合机制有 bug。
    # A vs B 只反映 decode 路径的数值差异：full-prompt 一次前向 vs 带 cache 逐位重算，
    #   bf16 下 attention 分块方式不同，near-tie 处 argmax 可能翻转，属预期噪声。
    if args.sanity:
        print(f"\n[自检] {args.sanity} 条 × 64 tokens", flush=True)
        zb = make_builder(zero=True)
        bc_match = ab_match = 0
        for r in rows[: args.sanity]:
            ids = encode(r["prompt"])
            am = torch.ones_like(ids)
            pos = torch.arange(ids.size(1), device=s_device).unsqueeze(0)
            a = baseline_generate(student, ids, am, 64, eos_ids, tok.pad_token_id)
            with torch.no_grad():
                self_cache = student(input_ids=ids, attention_mask=am, position_ids=pos,
                                     use_cache=True,
                                     past_key_values=DynamicCache()).past_key_values
            b = greedy_decode_with_cache(student, self_cache, ids, am, pos, 64, eos_ids)
            c = fused_gen(zb, ids, am, pos, max_new=64)

            def cmp(x, y):
                n = min(len(x), len(y))
                return (x[:n] == y[:n]), next((i for i in range(n) if x[i] != y[i]), n)

            bc_ok, bc_i = cmp(b, c)
            ab_ok, ab_i = cmp(a, b)
            bc_match += bc_ok
            ab_match += ab_ok
            if not bc_ok:
                print(f"  ✗ B≠C @token{bc_i} —— 融合机制有问题", flush=True)
            elif not ab_ok:
                print(f"  · A≠B @token{ab_i}（decode 路径数值噪声，B==C 说明融合无恙）", flush=True)
        print(f"[自检] 融合机制 B==C: {bc_match}/{args.sanity}"
              f"{' ✅' if bc_match == args.sanity else ' ❌ 融合有 bug，下面的数字不可信'}"
              f" | decode 数值 A==B: {ab_match}/{args.sanity}"
              f"{'' if ab_match == args.sanity else '（bf16 噪声，baseline/pretrained 两臂同样受影响，不影响对比公平性）'}",
              flush=True)
        del zb

    # ---- 主评测 ----
    # 【公平性】baseline 也走手写 decode loop（用 student 自身 KV），不用原生 generate。
    # 否则两臂 decode 路径不同，bf16 噪声会混进对比；现在唯一差别就是 projector。
    @torch.no_grad()
    def self_kv_gen(ids, am, pos):
        cache = student(input_ids=ids, attention_mask=am, position_ids=pos,
                        use_cache=True, past_key_values=DynamicCache()).past_key_values
        return greedy_decode_with_cache(student, cache, ids, am, pos,
                                        args.max_new_tokens, eos_ids)

    arms = {"baseline": None}
    if args.projector_path:
        arms["pretrained"] = make_builder(projector=load_projector_ckpt(args.projector_path))
    elif args.fuser_dir:
        arms["official_fuser"] = make_builder(
            projector=load_official_fuser_projectors(args.fuser_dir, s_device)
        )

    stats = {k: {"correct": 0, "total": 0, "no_answer": 0, "gen_len": 0} for k in arms}
    out_f = open(args.out, "w") if args.out else None

    print(f"\n[评测] 开始 ...", flush=True)
    for i, r in enumerate(rows):
        ids = encode(r["prompt"])
        am = torch.ones_like(ids)
        pos = torch.arange(ids.size(1), device=s_device).unsqueeze(0)
        label = r["label"]
        rec = {"idx": i, "label": label}

        for name, builder in arms.items():
            toks = (self_kv_gen(ids, am, pos) if builder is None
                    else fused_gen(builder, ids, am, pos))
            text = tok.decode(toks, skip_special_tokens=True)
            pred = extract_answer(text)
            ok = is_correct(pred, label)
            st = stats[name]
            st["correct"] += ok
            st["total"] += 1
            st["no_answer"] += pred is None
            st["gen_len"] += len(toks)
            rec[name] = {"pred": pred, "correct": bool(ok), "len": len(toks), "text": text}

        if out_f:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
        if (i + 1) % 10 == 0:
            cur = " | ".join(f"{k} {v['correct']}/{v['total']}={v['correct']/v['total']:.1%}"
                             for k, v in stats.items())
            print(f"  [{i+1}/{len(rows)}] {cur}", flush=True)

    print(f"\n[结果] 数学正确率（greedy，{len(rows)} 题）")
    for k, v in stats.items():
        n = max(1, v["total"])
        print(f"  {k:12s} acc {v['correct']}/{v['total']} = {v['correct']/n:.1%}   "
              f"未抽到答案 {v['no_answer']}   平均生成长度 {v['gen_len']/n:.0f}")

    experiment_arm = "pretrained" if "pretrained" in stats else "official_fuser"
    if experiment_arm in stats:
        b, p = stats["baseline"], stats[experiment_arm]
        n = max(1, b["total"])
        ba, pa = b["correct"] / n, p["correct"] / n
        print(f"\n[判据] pretrained - baseline = {(pa - ba) * 100:+.1f} 个百分点")
        # 配对样本下的二项标准误，用于判断差异是否可能只是噪声
        se = (ba * (1 - ba) / n) ** 0.5 * 100
        print(f"       单臂标准误 ≈ {se:.1f} 个百分点（n={n}）"
              f"；差异小于约 {2*se:.1f} 时不宜下结论")
        print(f"  结论: {'融合提升正确率 ✅' if pa > ba else ('持平' if pa == ba else '融合降低正确率 ❌')}")
    if out_f:
        out_f.close()
        print(f"\n[明细] {args.out}")


if __name__ == "__main__":
    main()
