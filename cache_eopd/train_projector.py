"""
【C2C 核心】projector 预训练脚本（路线 A / HF Rollout 的融合 projector 单独训练）。

目标：
    冻结 teacher 与 student，只训练 C2CProjector，使得 student 在「融合 KV」下
    的 next-token 输出分布逼近 teacher 在同一位置的输出分布。

    损失（在整段 response 上按 token 平均，而非只看末位）：
        L = mean_{b, t in response} KL( softmax(student_fused_logits[b,t]) ‖ softmax(teacher_logits[b,t]) )
    实现见 token_mean_kl()——F.kl_div 的 batchmean 不能直接用在 (B, R, V) 上。

    其中 student_fused 是把 teacher 前缀 KV 投影融合进 student 前缀 KV 后，
    再让 student 以该 fused 前缀 cache 做 teacher-forcing 得到的分布。

实测效果（Qwen3-4B → Qwen3-1.7B，DAPO-Math-17k，400 步）：
    50 条 held-out 上 per-token KL: baseline 0.4030 → pretrained 0.1812（-55%）

为什么这样设计（对应 EOPD rollout 场景）：
    rollout 时 student 用 fused prompt cache 去生成 response。预训练让这个 fused
    cache 携带 teacher 的「前缀知识」，从而 student 在 response 上的分布更接近 teacher，
    这正是 C2C 想要的知识迁移。

用法示例（apex-llm；必须用 rosetta 环境，Qwen3 需要 transformers>=4.51，且路径不能带 ~）：
    PYTHONPATH=/home/kejiechen/CacheEOPD \
    /home/knhdu/anaconda3/envs/rosetta/bin/python -m cache_eopd.train_projector \
        --teacher /home/kejiechen/taopd-baseline/modelweights/Qwen3-4B \
        --student /home/kejiechen/taopd-baseline/modelweights/Qwen3-1.7B \
        --data-path /home/kejiechen/taopd-baseline/data/DAPO-Math-17k-dedup/dapo_math_17k_dedup_slime.jsonl \
        --data-field prompt --device cuda:1 --teacher-device auto \
        --max-len 160 --prefix-ratio 0.5 --steps 400 --lr 1e-4 --grad-accum 8 \
        --warmup-steps 20 --eval-every 100 --holdout 200 --out-dir ./ckpt_projector

训练中会打印 holdout 上的 fused KL vs baseline；训练结束后用 eval_projector_kl.py
做完整验收（含 zero/random 两个对照组）。
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_eopd.fused_kv import FusedKVBuilder, FusedKVConfig, save_projector_ckpt


def token_mean_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """每 token 平均的 KL(student ‖ teacher)，单位 nats。

    【坑】F.kl_div(reduction="batchmean") 除的是 input.size(0)。若直接传 (B, R, V)，
    除数是 B=1，得到的是「整段 response 上 KL 之和」而非平均——梯度会被放大 R(≈70)倍，
    等效学习率暴涨，projector 一步就被轰离恒等点。必须先展平成 (B*R, V)。
    训练与验收共用此函数，保证两边指标口径一致。
    """
    s = student_logits.reshape(-1, student_logits.size(-1)).float()
    t = teacher_logits.reshape(-1, teacher_logits.size(-1)).float()
    return F.kl_div(F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1),
                    log_target=True, reduction="batchmean")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--data-path", required=True, help="jsonl 数据路径")
    p.add_argument("--data-field", default="prompt",
                   help="jsonl 中作为蒸馏文本用的字段（默认 prompt；也可 text/question 等）")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--teacher-device", default="auto",
                   help="teacher 设备；显存紧张时用 auto 跨卡分片")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-len", type=int, default=200, help="每条样本 token 上限（含 prefix+response）")
    p.add_argument("--min-len", type=int, default=24, help="短于此的样本会被丢弃")
    p.add_argument("--prefix-ratio", type=float, default=0.5,
                   help="prefix 占总长度比例（P = int(T*ratio)）；其余作为 response")
    p.add_argument("--holdout", type=int, default=200,
                   help="数据文件前 N 条留作验收集（训练不使用），避免训练/验收重叠")
    p.add_argument("--steps", type=int, default=400, help="训练步数")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="梯度累积步数（对齐 C2C 的大 batch 训练，降低单样本噪声）")
    p.add_argument("--lr", type=float, default=1e-4, help="学习率（对齐 C2C recipe）")
    p.add_argument("--eval-every", type=int, default=50,
                   help="每隔多少步在 holdout 上算一次 per-token KL（0=关闭）")
    p.add_argument("--eval-samples", type=int, default=16, help="训练中 holdout 评估的样本数")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1, help="每步样本数（原型固定 1，便于正确处理变长）")
    p.add_argument("--out-dir", default="./ckpt_projector")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--projector-hidden", type=int, default=1024)
    p.add_argument("--projector-inter", type=int, default=1024)
    p.add_argument("--projector-layers", type=int, default=3)
    p.add_argument("--zero-init", dest="zero_init", action="store_true", default=True,
                   help="projector 输出层零初始化：融合从『恒等(=student 自身)』出发，"
                        "训练只能改善、不会比 baseline 更差。关闭则用随机初始化。")
    p.add_argument("--no-zero-init", dest="zero_init", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(path, dtype, device, max_memory=None):
    if device == "auto":
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation="eager",
            device_map="auto", max_memory=max_memory).eval()
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation="eager").to(device).eval()


def load_texts(data_path, field):
    """读取 jsonl，抽取每条样本的蒸馏文本。"""
    texts = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 纯文本行
                texts.append(line)
                continue
            if field in obj:
                texts.append(str(obj[field]))
            elif "text" in obj:
                texts.append(str(obj["text"]))
            elif "prompt" in obj:
                texts.append(str(obj["prompt"]))
    return texts


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_f = open(log_path, "w")

    # ---- 模型与 tokenizer ----
    # teacher 用 auto 分片时，避开 student 所在卡与 VLLM 卡，避免显存打架
    student_gpu = 0
    if args.device.startswith("cuda:"):
        student_gpu = int(args.device.split(":")[1])
    teacher_max_memory = None
    if args.teacher_device == "auto":
        # 把 teacher 4B 分片到除 GPU0(VLLM) 与 student 卡之外的卡上
        teacher_max_memory = {
            i: ("1MiB" if i in (0, student_gpu) else "7GiB") for i in range(6)
        }
    print(f"[load] student  <- {args.student} ({args.device})")
    student = load_model(args.student, dtype, args.device)
    print(f"[load] teacher  <- {args.teacher} ({args.teacher_device})")
    teacher = load_model(args.teacher, dtype, args.teacher_device,
                         max_memory=teacher_max_memory)
    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- 数据：预 tokenize 成 1D 张量列表 ----
    print(f"[data] 读取 {args.data_path} (field={args.data_field}) ...")
    raw = load_texts(args.data_path, args.data_field)

    def tokenize(texts, cap):
        out = []
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            if ids.numel() >= args.min_len:
                out.append(ids[:cap])
        return out

    # 【关键】按文件顺序切分：前 holdout 条只用于验收，训练绝不碰，
    # 否则 train/eval 同分布重叠会让「训练 loss 下降」失去参考意义。
    holdout_texts, train_texts = raw[: args.holdout], raw[args.holdout:]
    samples = tokenize(train_texts, args.max_len)
    eval_samples = tokenize(holdout_texts, args.max_len)[: args.eval_samples]
    if not samples:
        raise RuntimeError("没有长度达标的样本，调小 --min-len 或换数据")
    print(f"[data] 训练样本 {len(samples)} 条 / holdout {len(eval_samples)} 条；max-len={args.max_len}")
    random.seed(args.seed)
    random.shuffle(samples)

    # ---- builder + 冻结 teacher/student，只训 projector ----
    cfg = FusedKVConfig(
        dtype=dtype,
        projector_hidden_dim=args.projector_hidden,
        projector_intermediate_dim=args.projector_inter,
        projector_num_layers=args.projector_layers,
        # 【关键】零初始化时投影输出层为 0 → fused == student 自身（恒等），
        # 训练从 baseline 出发，只能变好；随机初始化会先把 KV 打成噪声再指望学回来。
        zero_init=args.zero_init,
    )
    builder = FusedKVBuilder.from_models(teacher, student, cfg)
    builder.freeze_teacher_student()
    projector = builder.projector
    # 【C2C 核心】门控处理：本应用（把 teacher KV 融合进 student rollout）的目标是
    # 「始终融合」，因此强制门控常开（gate_logit>0）。否则 C2CProjector 的 gate 初始为 0，
    # 推理时硬门控 (gate_logit>0)=False 会把融合完全关掉，训练信号在推理时失效。
    # 关闭 Gumbel，使 train/eval 一致使用硬门控 = 1，梯度只走投影/权重路径。
    projector.use_gumbel = False
    with torch.no_grad():
        projector.key_gate_logit.fill_(3.0)
        projector.value_gate_logit.fill_(3.0)
    projector.train()

    s_device = next(student.parameters()).device
    params = list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup_steps)))

    # ---- holdout 在线验收：与 eval_projector_kl.py 同口径的 per-token KL ----
    def split_sample(ids):
        T = ids.numel()
        P = max(1, int(T * args.prefix_ratio))
        if T - 1 - P < 1:
            return None
        x = ids.unsqueeze(0).to(s_device)
        am = torch.ones(1, T, dtype=torch.long, device=s_device)
        pos = torch.arange(T, dtype=torch.long, device=s_device).unsqueeze(0)
        return x, am, pos, T, P

    @torch.no_grad()
    def holdout_kl(fused: bool):
        """fused=False 时是 baseline（student 自身前向），=True 时用当前 projector。"""
        was_train = projector.training
        projector.eval()
        tot, n = 0.0, 0
        for ids in eval_samples:
            packed = split_sample(ids)
            if packed is None:
                continue
            x, am, pos, T, P = packed
            t_dev = next(teacher.parameters()).device
            t_logits = teacher(input_ids=x.to(t_dev), attention_mask=am.to(t_dev),
                               position_ids=pos.to(t_dev)).logits[:, P:T - 1, :].to(s_device)
            if fused:
                fc, _ = builder.build_trainable(x, am, pos, P)
                s_logits = student(input_ids=x[:, P:T - 1], attention_mask=am[:, :T - 1],
                                   position_ids=pos[:, P:T - 1],
                                   past_key_values=fc, use_cache=True).logits
            else:
                s_logits = student(input_ids=x, attention_mask=am,
                                   position_ids=pos).logits[:, P:T - 1, :]
            tot += token_mean_kl(s_logits, t_logits).item()
            n += 1
        if was_train:
            projector.train()
        return tot / max(1, n)

    baseline_kl = holdout_kl(fused=False) if eval_samples and args.eval_every else float("nan")
    print(f"[holdout] baseline(student 自身) per-token KL = {baseline_kl:.4f}  "
          f"—— 训练后的 fused KL 必须低于它才算有效", flush=True)

    # ---- 训练循环 ----
    print(f"[train] steps={args.steps} lr={args.lr} prefix_ratio={args.prefix_ratio} "
          f"device={args.device} teacher_device={args.teacher_device}")
    global_step = 0
    sample_idx = 0
    accum_loss = 0.0
    accum_count = 0
    running = 0.0  # EMA of loss

    builder.train()  # 容器本身 train() 不影响冻结模型，但确保 projector 处于 train

    while global_step < args.steps:
        # 取一条样本，必要时裁剪到 max-len
        ids = samples[sample_idx % len(samples)]
        sample_idx += 1
        if ids.numel() > args.max_len:
            # 随机截取一段，增加多样性
            start = random.randint(0, ids.numel() - args.max_len)
            ids = ids[start:start + args.max_len]
        T = ids.numel()
        P = max(1, int(T * args.prefix_ratio))
        R = T - 1 - P
        if R < 1:
            continue

        input_ids = ids.unsqueeze(0).to(s_device)                       # (1, T)
        attention_mask = torch.ones(1, T, dtype=torch.long, device=s_device)
        position_ids = torch.arange(T, dtype=torch.long, device=s_device).unsqueeze(0)

        # (1) 可微 fused 前缀 cache + teacher 全序列 logits
        fused_cache, teacher_logits = builder.build_trainable(
            input_ids, attention_mask, position_ids, P)

        # (2) student 以 fused 前缀做 teacher-forcing，得到 response 分布
        resp_ids = input_ids[:, P:T - 1]            # (1, R) = x_P .. x_{T-2}
        resp_pos = position_ids[:, P:T - 1]
        resp_attn = attention_mask[:, :T - 1]        # 长度 P + R = T-1
        # student 权重冻结但 fused 前缀可微：不要包 no_grad，保留反传路径。
        # 注意：必须用 use_cache=True，否则 HF 会忽略传入的 past_key_values（梯度断链）。
        student.train()  # 仅影响 dropout/norm；权重已冻结，不会更新
        s_out = student(
            input_ids=resp_ids,
            attention_mask=resp_attn,
            position_ids=resp_pos,
            past_key_values=fused_cache,
            use_cache=True,
        )
        student_logits = s_out.logits               # (1, R, V)

        # (3) 蒸馏目标：teacher 在 response 位置的 logits
        tgt = teacher_logits[:, P:T - 1, :]          # (1, R, V)

        loss = token_mean_kl(student_logits, tgt)

        # (4) 反向 + 优化（含梯度累积）
        loss = loss / args.grad_accum
        loss.backward()
        accum_loss += loss.item() * args.grad_accum
        accum_count += 1

        if accum_count % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            global_step += 1
            builder.projector.update_temperature(global_step)

            avg = accum_loss / max(1, accum_count)
            running = 0.9 * running + 0.1 * avg if running > 0 else avg
            accum_loss = 0.0
            accum_count = 0

            if global_step % args.log_every == 0:
                lr_now = sched.get_last_lr()[0]
                msg = (f"step {global_step:>5d} | loss {avg:.4f} | "
                       f"ema {running:.4f} | lr {lr_now:.2e}")
                print(msg, flush=True)
                log_f.write(json.dumps({
                    "step": global_step, "loss": avg, "ema_loss": running,
                    "lr": lr_now,
                }) + "\n")
                log_f.flush()

            if args.eval_every and global_step % args.eval_every == 0 and eval_samples:
                hk = holdout_kl(fused=True)
                delta = (baseline_kl - hk) / baseline_kl * 100
                print(f"  [holdout] step {global_step} fused KL {hk:.4f} vs "
                      f"baseline {baseline_kl:.4f}  ({delta:+.1f}%)", flush=True)
                log_f.write(json.dumps({
                    "step": global_step, "holdout_fused_kl": hk,
                    "holdout_baseline_kl": baseline_kl,
                }) + "\n")
                log_f.flush()

            if global_step % args.save_every == 0:
                ckpt = os.path.join(args.out_dir, f"projector_step{global_step}.pt")
                save_projector_ckpt(projector, ckpt)
                print(f"[save] {ckpt}", flush=True)

    # 最终存档
    final = os.path.join(args.out_dir, "projector_final.pt")
    save_projector_ckpt(projector, final)
    print(f"[done] 最终 projector 已保存: {final}")
    print(f"[done] 训练日志: {log_path}")
    log_f.close()


if __name__ == "__main__":
    main()
