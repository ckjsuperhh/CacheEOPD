"""
【C2C 核心】路线 A 最小验证原型（不依赖 verl / ray，单卡即可跑）。

跑三路对比，验证 fused-KV 注入链路是否打通：
    A. baseline    : student 直接 generate（EOPD 现状）
    B. self-kv     : student 先前缀前向拿自己的 KV，再带 past_key_values 续写
                     —— 用来验证「切末 token + past_key_values」这条路径本身正确
                     （B 的输出在 greedy 下应与 A 完全一致，否则说明注入方式有 bug）
    C. fused-kv    : teacher KV 经 Projector 投影融合进 student KV 后续写（真正的 C2C）

用法:
    python -m cache_eopd.prototype_generate \
        --student ~/taopd-baseline/modelweights/Qwen3-1.7B \
        --teacher ~/taopd-baseline/modelweights/Qwen3-4B \
        --prompt "What is 17 * 23? Think step by step."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cache_eopd.fused_kv import FusedKVBuilder, FusedKVConfig, _get_layer_kv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True, help="student(receiver) 模型路径")
    p.add_argument("--teacher", required=True, help="teacher(sharer) 模型路径")
    p.add_argument("--prompt", default="What is 17 * 23? Think step by step.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda", help="student 所在设备（fused KV 与生成都在这里）")
    p.add_argument("--teacher-device", default=None,
                   help="teacher 所在设备，默认与 student 同卡；显存紧张时可放另一张卡")
    p.add_argument("--zero-init", action="store_true", help="projector 输出层零初始化（此时 C 应等于 B）")
    return p.parse_args()


def load_model(path, dtype, device):
    # device 传 "auto" 时用 accelerate 的 device_map 跨卡分片（大 teacher 单卡放不下时用）
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, attn_implementation="eager", device_map="auto")
        return model.eval()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager")
    return model.to(device).eval()


@torch.no_grad()
def generate_with_cache(model, input_ids, attention_mask, cache, max_new_tokens, eos_id, pad_id):
    """【C2C 核心】带外部前缀 KV 的续写。

    HF generate 若同时收到完整 input_ids 和 past_key_values 会在新版里做 cache 长度校验，
    行为随版本变化。这里手写自回归 decode loop（方案 2），语义明确、跨版本稳定：
        每步只喂 1 个 token，attention_mask 覆盖 [前缀 + 已生成]，position 连续递增。
    """
    B, L = input_ids.shape
    device = input_ids.device
    # 前缀最后一个 token 作为 decode 的第一个输入（其 KV 尚未在 cache 中被“消费”，
    # 但它的 KV 已在 cache 里——所以实际喂入的是 cache 已含前缀、由末 token 触发下一步预测）
    # 因此这里直接用前缀最后一位的 logits 采下一个 token 更省一次前向：
    # 简化起见仍走标准 loop：喂末 token，同时把 cache 裁掉末 token 的那一格。
    cache = _crop_cache(cache, L - 1)
    cur = input_ids[:, -1:]
    # left-padding 下真实位置 = 累计 mask - 1
    pos = attention_mask.long().cumsum(-1) - 1
    cur_pos = pos[:, -1:]

    generated = []
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    mask = attention_mask
    for _ in range(max_new_tokens):
        out = model(
            input_ids=cur,
            attention_mask=mask,
            position_ids=cur_pos,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)  # greedy，便于 A/B 对齐比较
        next_tok = torch.where(finished.unsqueeze(-1), torch.full_like(next_tok, pad_id), next_tok)
        generated.append(next_tok)
        finished = finished | (next_tok.squeeze(-1) == eos_id)
        if finished.all():
            break
        cur = next_tok
        cur_pos = cur_pos + 1
        mask = torch.cat([mask, (~finished).long().unsqueeze(-1)], dim=-1)
    return torch.cat(generated, dim=-1)


def _crop_cache(cache, length):
    """把 cache 裁到前 `length` 个位置（丢弃末 token 那一格，由 decode loop 重新计算）。"""
    new_cache = DynamicCache()
    idx = 0
    while True:
        try:
            k, v = _get_layer_kv(cache, idx)
        except (IndexError, AttributeError):
            break
        if k is None:
            break
        new_cache.update(k[:, :, :length, :].contiguous(), v[:, :, :length, :].contiguous(), idx)
        idx += 1
        if idx > 200:
            break
    return new_cache


@torch.no_grad()
def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.student)
    student = load_model(args.student, dtype, args.device)
    teacher = load_model(args.teacher, dtype, args.teacher_device or args.device)

    print(f"[info] student layers={student.config.num_hidden_layers} "
          f"kv_heads={getattr(student.config, 'num_key_value_heads', None)} "
          f"head_dim={getattr(student.config, 'head_dim', None)}")
    print(f"[info] teacher layers={teacher.config.num_hidden_layers} "
          f"kv_heads={getattr(teacher.config, 'num_key_value_heads', None)} "
          f"head_dim={getattr(teacher.config, 'head_dim', None)}")

    msgs = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt", padding=True).to(args.device)
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    # ---- A. baseline ----
    out_a = student.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        do_sample=False, max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id, pad_token_id=pad_id,
    )
    resp_a = out_a[:, input_ids.shape[1]:]

    # ---- B. self-kv（链路自检）----
    s_cache = student(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=True, past_key_values=DynamicCache()).past_key_values
    resp_b = generate_with_cache(student, input_ids, attention_mask, s_cache,
                                 args.max_new_tokens, eos_id, pad_id)

    # ---- C. fused-kv（C2C）----
    cfg = FusedKVConfig(dtype=dtype, zero_init=args.zero_init)
    builder = FusedKVBuilder.from_models(teacher, student, cfg)
    fused_cache, _ = builder.build(input_ids, attention_mask)
    fk, fv = _get_layer_kv(fused_cache, 0)
    print(f"[info] fused cache layer0 key shape={tuple(fk.shape)} value shape={tuple(fv.shape)}")
    resp_c = generate_with_cache(student, input_ids, attention_mask, fused_cache,
                                 args.max_new_tokens, eos_id, pad_id)

    print("\n=== A. student baseline (generate) ===")
    print(tok.decode(resp_a[0], skip_special_tokens=True))
    print("\n=== B. student + self KV (链路自检，应≈A) ===")
    print(tok.decode(resp_b[0], skip_special_tokens=True))
    print("\n=== C. student + C2C fused KV ===")
    print(tok.decode(resp_c[0], skip_special_tokens=True))

    n = min(resp_a.shape[1], resp_b.shape[1])
    same_ab = (resp_a[:, :n] == resp_b[:, :n]).all().item()
    print(f"\n[check] B≡A 前 {n} token 完全一致: {same_ab}  (False 说明 KV 注入路径有 bug)")

    if args.zero_init:
        # 【C2C 核心】zero_init 下 projected=0，融合式 fused = student + gate*w*0 = student，
        # 因此 C 必须严格等于 B。这是融合公式与层映射实现是否正确的判决性检验。
        m = min(resp_b.shape[1], resp_c.shape[1])
        same_bc = (resp_b[:, :m] == resp_c[:, :m]).all().item()
        print(f"[check] zero_init 下 C≡B 前 {m} token 完全一致: {same_bc}  "
              f"(False 说明融合公式/层映射有 bug)")


if __name__ == "__main__":
    main()
