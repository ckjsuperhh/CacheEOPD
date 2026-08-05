"""HF EOPD training with optional C2C teacher-KV rollout guidance."""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from cache_eopd.fused_kv import FusedKVBuilder, FusedKVConfig, _get_layer_kv
from cache_eopd.train_student_distill import load_official_fuser_projectors, load_teacher_sharded


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--fuser-dir", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--teacher-gpus", default="1")
    parser.add_argument("--proj-device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-prompt-len", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--entropy-threshold", type=float, default=0.8)
    parser.add_argument("--soft-kd-coef", type=float, default=1.0)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--mode", choices=["fused", "mixed", "anneal"], required=True)
    parser.add_argument("--fused-prob", type=float, default=0.5)
    parser.add_argument("--anneal-start-prob", type=float, default=1.0)
    parser.add_argument("--anneal-end-prob", type=float, default=0.0)
    parser.add_argument("--anneal-steps", type=int, default=None)
    parser.add_argument("--anneal-schedule", choices=["linear", "quadratic", "sqrt"], default="linear")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def log_probs_for_labels(logits, labels):
    return F.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def render_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def crop_cache(cache, length):
    cropped = DynamicCache()
    layer = 0
    while True:
        try:
            key, value = _get_layer_kv(cache, layer)
        except (IndexError, AttributeError):
            break
        if key is None:
            break
        cropped.update(key[:, :, :length, :].contiguous(), value[:, :, :length, :].contiguous(), layer)
        layer += 1
    return cropped


def sample_token(logits, tokenizer, do_sample=True):
    if not do_sample:
        return logits.argmax(-1, keepdim=True)
    return torch.multinomial(torch.softmax(logits.float(), dim=-1), num_samples=1)


def probability(args, step):
    if args.mode == "fused":
        return 1.0
    if args.mode == "mixed":
        return args.fused_prob
    total = max(1, args.anneal_steps or args.steps)
    progress = min(1.0, step / total)
    if args.anneal_schedule == "quadratic":
        progress = progress**2
    elif args.anneal_schedule == "sqrt":
        progress = progress**0.5
    return args.anneal_start_prob + progress * (args.anneal_end_prob - args.anneal_start_prob)


@torch.no_grad()
def student_plain_rollout(student, prompt, max_new_tokens, eos_ids, pad_id):
    generated = student.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=list(eos_ids),
        pad_token_id=pad_id,
    )
    response = generated[:, prompt.shape[1]:]
    if response.shape[1] == 0:
        return response, None
    sequence = torch.cat([prompt, response], dim=1)
    output = student(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False)
    logits = output.logits[:, prompt.shape[1] - 1:-1]
    return response, log_probs_for_labels(logits, response).detach()


@torch.no_grad()
def student_fused_rollout(student, builder, prompt, max_new_tokens, eos_ids, pad_id):
    prompt_len = prompt.shape[1]
    attention_mask = torch.ones_like(prompt)
    position_ids = torch.arange(prompt_len, device=prompt.device).unsqueeze(0)
    fused_cache, _ = builder.build(prompt, attention_mask, position_ids)
    cache = crop_cache(fused_cache, prompt_len - 1)
    current = prompt[:, -1:]
    current_position = position_ids[:, -1:]
    mask = attention_mask
    response_tokens = []
    old_log_probs = []
    finished = False
    for _ in range(max_new_tokens):
        output = student(
            input_ids=current,
            attention_mask=mask,
            position_ids=current_position,
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        next_token = sample_token(logits, None, do_sample=True)
        old_log_probs.append(log_probs_for_labels(logits.unsqueeze(1), next_token))
        response_tokens.append(next_token)
        if int(next_token.item()) in eos_ids:
            finished = True
        if finished:
            break
        current = next_token
        current_position = current_position + 1
        mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
    if not response_tokens:
        return torch.empty((1, 0), dtype=prompt.dtype, device=prompt.device), None
    response = torch.cat(response_tokens, dim=1)
    return response, torch.cat(old_log_probs, dim=1).detach()


def student_fused_response_logits(student, builder, prompt, response):
    prompt_len = prompt.shape[1]
    response_len = response.shape[1]
    attention_mask = torch.ones_like(prompt)
    position_ids = torch.arange(prompt_len, device=prompt.device).unsqueeze(0)
    with torch.no_grad():
        fused_cache, _ = builder.build(prompt, attention_mask, position_ids)
    cache = crop_cache(fused_cache, prompt_len - 1)
    model_inputs = torch.cat([prompt[:, -1:], response[:, :-1]], dim=1)
    model_mask = torch.ones(
        (1, prompt_len + response_len - 1), dtype=prompt.dtype, device=prompt.device
    )
    model_positions = torch.arange(
        prompt_len - 1, prompt_len + response_len - 1, device=prompt.device
    ).unsqueeze(0)
    output = student(
        input_ids=model_inputs,
        attention_mask=model_mask,
        position_ids=model_positions,
        past_key_values=cache,
        use_cache=True,
    )
    return output.logits


@torch.no_grad()
def teacher_signals(teacher, sequence, prompt_len, topk):
    device = next(teacher.parameters()).device
    inputs = sequence.to(device)
    output = teacher(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    response_logits = output.logits[:, prompt_len - 1:-1]
    labels = inputs[:, prompt_len:]
    full_log_probs = F.log_softmax(response_logits.float(), dim=-1)
    teacher_log_probs = full_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    teacher_entropy = -(full_log_probs.exp() * full_log_probs).sum(dim=-1)
    topk_logits, topk_indices = torch.topk(response_logits.float(), k=topk, dim=-1)
    teacher_topk_log_probs = F.log_softmax(topk_logits, dim=-1)
    return teacher_log_probs, teacher_entropy, teacher_topk_log_probs, topk_indices


def save_student(student, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    student.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(args.device)
    student_device = torch.device(args.device)
    teacher_gpus = [int(value) for value in args.teacher_gpus.split(",") if value]
    teacher = load_teacher_sharded(args.teacher, dtype, teacher_gpus, attn_impl="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    projectors = load_official_fuser_projectors(args.fuser_dir, args.proj_device or args.device)
    projectors = projectors.to(device=args.proj_device or args.device, dtype=dtype)
    config = FusedKVConfig(
        dtype=dtype,
        per_layer_projector=True,
        keep_last_token_unfused=True,
        layer_mapping_strategy="last_aligned",
    )
    builder = FusedKVBuilder.from_models(teacher, student, config, projector=projectors)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in builder.projectors.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    builder.projectors.eval()
    with open(args.data_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("data file is empty")
    eos_ids = {tokenizer.eos_token_id}
    generation_eos = student.generation_config.eos_token_id
    if isinstance(generation_eos, list):
        eos_ids.update(generation_eos)
    elif generation_eos is not None:
        eos_ids.add(generation_eos)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as log_handle:
        for step in range(1, args.steps + 1):
            row = rows[(step - 1) % len(rows)]
            prompt_ids = render_prompt(tokenizer, row["prompt"])[-args.max_prompt_len:]
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=student_device).unsqueeze(0)
            student.eval()
            use_fused = random.random() < probability(args, step - 1)
            if use_fused:
                response, old_log_probs = student_fused_rollout(
                    student, builder, prompt, args.max_new_tokens, eos_ids, tokenizer.pad_token_id
                )
            else:
                response, old_log_probs = student_plain_rollout(
                    student, prompt, args.max_new_tokens, eos_ids, tokenizer.pad_token_id
                )
            if response.shape[1] == 0:
                continue
            sequence = torch.cat([prompt, response], dim=1)
            prompt_len = prompt.shape[1]
            if old_log_probs is None:
                continue
            teacher_log_probs, teacher_entropy, teacher_topk_log_probs, topk_indices = teacher_signals(
                teacher, sequence, prompt_len, args.topk
            )
            student.train()
            if use_fused:
                response_logits = student_fused_response_logits(student, builder, prompt, response)
            else:
                output = student(input_ids=sequence, attention_mask=torch.ones_like(sequence), use_cache=False)
                response_logits = output.logits[:, prompt_len - 1:-1]
            current_log_probs = log_probs_for_labels(response_logits, response)
            teacher_log_probs = teacher_log_probs.to(student_device)
            teacher_entropy = teacher_entropy.to(student_device)
            teacher_topk_log_probs = teacher_topk_log_probs.to(student_device)
            topk_indices = topk_indices.to(student_device)
            old_log_probs = old_log_probs.to(student_device)
            student_topk_log_probs = F.log_softmax(response_logits.float(), dim=-1).gather(-1, topk_indices)
            advantage = (teacher_log_probs - old_log_probs).detach()
            ratio = torch.exp(current_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
            pg_loss = torch.maximum(-advantage * ratio, -advantage * clipped_ratio).mean()
            soft_mask = (teacher_entropy >= args.entropy_threshold).float()
            teacher_probs = teacher_topk_log_probs.exp()
            kl = (teacher_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)
            soft_kd_loss = (kl * soft_mask).sum() / soft_mask.sum().clamp_min(1.0)
            loss = pg_loss + args.soft_kd_coef * soft_kd_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pg_loss": float(pg_loss.detach().cpu()),
                "soft_kd_loss": float(soft_kd_loss.detach().cpu()),
                "teacher_entropy": float(teacher_entropy.mean().detach().cpu()),
                "soft_kd_token_ratio": float(soft_mask.mean().detach().cpu()),
                "response_len": int(response.shape[1]),
                "fused_rollout": bool(use_fused),
                "fused_probability": probability(args, step - 1),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
            if step % args.log_every == 0:
                print(json.dumps(record), flush=True)
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
            if step % args.save_every == 0 or step == args.steps:
                save_student(student, tokenizer, os.path.join(args.out_dir, f"step{step}"))


if __name__ == "__main__":
    main()
