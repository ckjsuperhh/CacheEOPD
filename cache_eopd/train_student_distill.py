"""
【EOPD 核心】轻量蒸馏对比实验 —— 4B->0.6B 官方 projector。

目的：验证「在学生 rollout 时注入 teacher KV（C2C）做蒸馏训练」是否让模型变好。
这是与 train_projector.py **梯度目标相反**的实验：
  train_projector.py：冻结 teacher+student，只训 projector        （机制探针）
  本脚本：           冻结 teacher+官方 projector，只训 **student**（真正的蒸馏）

两种模式（各跑一次 = 两个学生，再各自 plain 评测对比）：
  --mode fused  : 训时前缀用官方 projector 融 teacher KV（w/ C2C）
  --mode plain  : 训时前缀用学生自身 KV            （w/o C2C，基线）

数据：teacher(Qwen3-4B) 的答对轨迹 jsonl，字段 {prompt, solution}，
      由 gen_teacher_traj.py 产出（teacher 用 Qwen3-4B）。
评测：训完用 eval_math_acc.py 的 baseline 臂（不加载 projector）评两学生的 plain 准确率，
      比 (fused 学生 plain) vs (plain 学生 plain) 即「C2C 训练是否让学生本身变好」。

【关键】官方 fuser 是 rosetta 格式（final/ 下 projector_{idx}.pt + .json）。
        本脚本**直接**从这些文件加载 projector 权重，搬进我们的 FusedKVBuilder
        （同属 vendored C2CProjector），不再调用 load_rosetta_model —— 后者会把
        4B teacher 也按 device_map={"":student.device} 载入 student 所在卡，
        在 apex 这种单卡仅 ~7GB 的环境必 OOM。

用法（apex-llm）：
  PYTHONPATH=/home/kejiechen/CacheEOPD \
  /home/knhdu/anaconda3/envs/rosetta/bin/python -m cache_eopd.train_student_distill \
    --teacher $M/Qwen3-4B --student $M/Qwen3-0.6B \
    --data-path ./data/teacher_traj_gsm8k_4b.jsonl \
    --fuser-dir $M/qwen3_0.6b+qwen3_4b_base_Fuser/final \
    --device cuda:1 --teacher-device auto --teacher-gpus 4,5 --mode fused \
    --steps 300 --lr 1e-5 --grad-accum 8 --out-dir ./ckpt_student_fused
  # --mode plain 时不用融 KV，无需 --fuser-dir
"""

import argparse
import json
import os
import random
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_eopd.fused_kv import FusedKVBuilder, FusedKVConfig, DynamicCache
from rosetta.model.projector import load_projector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--data-path", required=True,
                   help="teacher 轨迹 jsonl（含 prompt / solution）")
    # 官方 fuser：仅 fused 模式需要；直接从 final/ 下 projector_{idx}.pt 加载
    p.add_argument("--fuser-dir", default=None,
                   help="官方 fuser 目录，如 .../qwen3_0.6b+qwen3_4b_base_Fuser/final（fused 模式必填）")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--teacher-device", default="auto")
    p.add_argument("--teacher-gpus", default=None)
    p.add_argument("--teacher-mem-per-gpu", default="7GiB")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-prompt-len", type=int, default=384)
    p.add_argument("--max-answer-len", type=int, default=384)
    p.add_argument("--min-answer-len", type=int, default=8)
    p.add_argument("--holdout", type=int, default=64, help="前 N 条留作验收集（绝不训练）")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5, help="学生 SFT 学习率（比 projector 小）")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--mode", choices=["fused", "plain"], required=True,
                   help="fused=训时融 teacher KV；plain=不融（基线）")
    p.add_argument("--fusion-scale", type=float, default=1.0,
                   help="融合强度（仅 fused 模式生效）")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-dir", default="./ckpt_student")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_official_fuser_projectors(fuser_dir: str, device) -> torch.nn.ModuleList:
    """直接从官方 fuser 的 final/ 目录加载 projector 权重（绕过 load_rosetta_model）。

    final/ 下是标准命名 projector_{idx}.pt（权重）+ projector_{idx}.json（结构配置），
    与 rosetta load_rosetta_model 内部读取方式一致。直接读避免它把 4B teacher 也按
    device_map={"":student.device} 塞进 student 所在卡，导致 apex 这种单卡仅 ~7GB
    的环境 OOM。返回按 idx 升序的 nn.ModuleList[C2CProjector]。
    """
    pt_files = sorted(
        [f for f in os.listdir(fuser_dir) if re.match(r"projector_\d+\.pt$", f)],
        key=lambda f: int(re.search(r"projector_(\d+)\.pt$", f).group(1)),
    )
    if not pt_files:
        raise FileNotFoundError(f"在 {fuser_dir} 没找到 projector_*.pt")
    proj_list = []
    for pt in pt_files:
        idx = int(re.search(r"projector_(\d+)\.pt$", pt).group(1))
        json_cfg = os.path.join(fuser_dir, f"projector_{idx}.json")
        proj = load_projector(json_cfg).to(device)
        state = torch.load(os.path.join(fuser_dir, pt), map_location=device)
        proj.load_state_dict(state, strict=False)
        proj_list.append(proj)
    return torch.nn.ModuleList(proj_list)


def load_model(path, dtype, device, max_memory=None, attn_impl="sdpa"):
    if device == "auto":
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation=attn_impl,
            device_map="auto", max_memory=max_memory).eval()
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation=attn_impl).to(device).eval()


def build_samples(rows, tok, args):
    """{prompt, solution} -> (input_ids, prefix_len)。与 train_projector.py 同口径。"""
    out = []
    for r in rows:
        prompt, solution = r.get("prompt"), r.get("solution")
        if not prompt or not solution:
            continue
        msgs = [{"role": "user", "content": prompt}]
        try:
            instruction = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            instruction = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok(instruction, add_special_tokens=False)["input_ids"]
        a_ids = tok(solution, add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None:
            a_ids = a_ids + [tok.eos_token_id]
        if len(a_ids) < args.min_answer_len:
            continue
        p_ids = p_ids[-args.max_prompt_len:]
        a_ids = a_ids[: args.max_answer_len]
        out.append((torch.tensor(p_ids + a_ids, dtype=torch.long), len(p_ids)))
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = getattr(torch, args.dtype)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 模型 ----
    student_gpu = int(args.device.split(":")[1]) if args.device.startswith("cuda:") else 0
    teacher_max_memory = None
    if args.teacher_device == "auto":
        n_gpu = torch.cuda.device_count()
        if args.teacher_gpus:
            allow = {int(x) for x in args.teacher_gpus.split(",")}
        else:
            allow = {i for i in range(n_gpu)} - {0, student_gpu}
        teacher_max_memory = {i: (args.teacher_mem_per_gpu if i in allow else "1MiB")
                              for i in range(n_gpu)}
    print(f"[load] student  <- {args.student} ({args.device})", flush=True)
    student = load_model(args.student, dtype, args.device, attn_impl="sdpa")
    print(f"[load] teacher  <- {args.teacher} ({args.teacher_device})", flush=True)
    teacher = load_model(args.teacher, dtype, args.teacher_device,
                         max_memory=teacher_max_memory, attn_impl="sdpa")
    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- 构造 builder（per-layer projector），再加载官方 fuser 权重 ----
    cfg = FusedKVConfig(
        dtype=dtype,
        per_layer_projector=True,
        keep_last_token_unfused=True,
        zero_init=False,
        fusion_scale=args.fusion_scale,
    )
    builder = FusedKVBuilder.from_models(teacher, student, cfg)
    n_proj = len(builder.projectors)
    print(f"[proj] builder 有 {n_proj} 层 projector（per-layer）", flush=True)

    if args.mode == "fused":
        assert args.fuser_dir, "fused 模式必须指定 --fuser-dir（官方 fuser 目录）"
        print(f"[load] 直接从磁盘加载官方 fuser projector: {args.fuser_dir}", flush=True)
        official_projectors = load_official_fuser_projectors(args.fuser_dir, student.device)
        assert len(official_projectors) == n_proj, \
            f"projector 层数不符：官方 {len(official_projectors)} vs builder {n_proj}"
        # 【关键】直接替换 builder 的 projector 列表：官方 fuser 的 MLP 维度
        # （如 hidden=1024，见 projector_0.json）与本项目默认（512）不同；若用
        # load_state_dict 复制会因形状不符被 strict=False 静默丢弃 → 投影变随机。
        # 整体替换保证架构与权重完全一致。
        builder.projectors = official_projectors
        for p in builder.projectors:
            p.requires_grad_(False)
            p.eval()
        print(f"[proj] 官方 projector 已加载并冻结：{n_proj} 层", flush=True)
    else:
        # plain 模式不融 KV，projector 冻结即可（不会被调用）
        for p in builder.projectors:
            p.requires_grad_(False)
            p.eval()

    # ---- 冻结 teacher + projector，只训 student ----
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    student.train()  # 唯一可训
    n_stu = sum(q.numel() for q in student.parameters())
    print(f"[stu] 可训参数 {n_stu/1e6:.1f}M（teacher+projector 已冻结）", flush=True)

    # ---- 数据 ----
    rows = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    hold_rows, train_rows = rows[: args.holdout], rows[args.holdout:]
    samples = build_samples(train_rows, tok, args)
    eval_samples = build_samples(hold_rows, tok, args)[:32]
    if not samples:
        raise RuntimeError("没有可用样本")
    random.shuffle(samples)
    avg_p = sum(p for _, p in samples) / len(samples)
    avg_a = sum(x.numel() - p for x, p in samples) / len(samples)
    print(f"[data] 训练 {len(samples)} 条 / holdout {len(eval_samples)} 条 | "
          f"平均题面 {avg_p:.0f} tok，平均轨迹 {avg_a:.0f} tok", flush=True)

    s_device = next(student.parameters()).device

    def pack(ids, P):
        T = ids.numel()
        x = ids.unsqueeze(0).to(s_device)
        am = torch.ones(1, T, dtype=torch.long, device=s_device)
        pos = torch.arange(T, dtype=torch.long, device=s_device).unsqueeze(0)
        return x, am, pos, T, P

    def student_response_ce(x, am, pos, T, P, fused):
        """学生续写 response 的 CE。前缀 KV 来源由 fused 决定；梯度只回 student。"""
        gold = x[:, P:]
        if fused:
            fc, _ = builder.build_trainable(x, am, pos, P, need_teacher_logits=False)
        else:
            with torch.no_grad():
                # 【对齐 fused 分支】前缀 cache 只取前 P-1 位，末位 x_{P-1} 当作 decode
                # 首步重新前向（与 build_trainable 的 keep_last_token_unfused 同口径）。
                # 否则 cache 含 P 位却从 c=P-1 喂起，会把 x_{P-1} 既存进 cache 又当新 token，
                # 位置错一格 → SDPA 序列长度对不上（expanded size 227 vs 226）。
                out = student(input_ids=x[:, :P - 1], attention_mask=am[:, :P - 1],
                              position_ids=pos[:, :P - 1], use_cache=True,
                              past_key_values=DynamicCache())
            fc = out.past_key_values
        c = builder.trainable_cache_len(P)  # P-1（末位留作 decode 首步）
        out = student(input_ids=x[:, c:T - 1], attention_mask=am[:, :T - 1],
                      position_ids=pos[:, c:T - 1], past_key_values=fc, use_cache=True)
        logits = out.logits[:, -gold.size(1):, :]   # (1, R, V)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), gold.reshape(-1))
        return ce

    @torch.no_grad()
    def holdout_plain_ce():
        """纯 plain 验收：看学生**自身**能力（不融 KV），监控是否真在学。"""
        tot, n = 0.0, 0
        for ids, P in eval_samples:
            x, am, pos, T, PP = pack(ids, P)
            if T - PP < 1 or PP < 2:
                continue
            tot += student_response_ce(x, am, pos, T, PP, fused=False).item()
            n += 1
        return tot / max(1, n)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))

    # ---- 训练循环 ----
    print(f"[train] mode={args.mode} steps={args.steps} lr={args.lr} "
          f"grad_accum={args.grad_accum}", flush=True)
    global_step = 0
    idx = 0
    accum_ce, accum_n = 0.0, 0
    while global_step < args.steps:
        ids, P = samples[idx % len(samples)]
        idx += 1
        x, am, pos, T, P = pack(ids, P)
        if T - P < 1 or P < 2:
            continue
        ce = student_response_ce(x, am, pos, T, P, fused=(args.mode == "fused"))
        (ce / args.grad_accum).backward()
        accum_ce += ce.item()
        accum_n += 1
        if accum_n % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"step {global_step:>4d} | CE {accum_ce/accum_n:.4f} | mode {args.mode}",
                      flush=True)
            accum_ce, accum_n = 0.0, 0
            if args.eval_every and global_step % args.eval_every == 0 and eval_samples:
                hce = holdout_plain_ce()
                print(f"  [holdout plain CE] step {global_step} | {hce:.4f}", flush=True)
            if args.save_every and global_step % args.save_every == 0:
                sp = os.path.join(args.out_dir, f"student_step{global_step}")
                student.save_pretrained(sp)
                print(f"[save] {sp}", flush=True)

    student.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"[done] 学生已存至 {args.out_dir}  （下一步用 eval_math_acc.py baseline 臂评 plain 准确率）")


if __name__ == "__main__":
    main()
