"""
【C2C 核心】projector 预训练脚本 —— **C2C 原版 SFT 口径**（v2 重写）。

训练目标（对齐 rosetta/train/dataset_adapters.py:1503-1546 的 ChatDataset）：
    instruction = 题目（chat template 包好，末尾带 generation prompt）
    response    = teacher 答对的解题轨迹
    fused KV    只覆盖 instruction 段（C2C 的 kv_cache_index：instruction=[1,0] 融合，
                response=[-1,0] 不融合）
    loss        = 交叉熵，**只算在 response token 上**（labels 里 instruction 全为 -100）

    L = mean_{t in response} CE( student_fused_logits[t], gold_token[t] )

    也就是：「前缀里塞进 teacher 的 KV 之后，student 能不能把这道题做对」。

为什么推翻 v1（KL 蒸馏）：
    v1 把任意一段文本从中间劈开，让 student 在后半段的分布逼近 teacher 的分布。
    这个信号里既没有「题目/答案」的边界，也没有「什么是正确答案」。projector 学到的
    只是 teacher 的说话风格 —— holdout KL 降了 55%，但 GSM8K 正确率从 86.0% 掉到 80.0%
    （生成很流畅但读错题）。C2C 原版从来不是这么训的。

三条与 C2C 对齐的结构约定（见 fused_kv.py 顶部注释）：
    1. per-layer projector：28 个 student 层各一个独立 projector（SFT_train.py:618）
    2. 只融合前 L-1 个 token，末位由 decode 首步用 student 自身 KV 重算
       —— 训练与 rollout 推理必须同口径，否则学到的东西错位一格
    3. 门控保持可学习（Gumbel-Sigmoid + 温度退火），只把初值设成正数保证一开始
       融合是开的。v1 把 gate 焊死成常开 = 无差别灌入，正是"把不该融合的融合了"。

数据来源：cache_eopd/gen_teacher_traj.py 产出的 jsonl
    {"problem":..., "prompt": 套好模板的题面, "solution": teacher 答对的解题过程, "label":...}

用法（apex-llm）：
    PYTHONPATH=/home/kejiechen/CacheEOPD \
    /home/knhdu/anaconda3/envs/rosetta/bin/python -m cache_eopd.train_projector \
        --teacher /home/kejiechen/taopd-baseline/modelweights/Qwen3-4B \
        --student /home/kejiechen/taopd-baseline/modelweights/Qwen3-1.7B \
        --data-path ./data/teacher_traj_gsm8k1500.jsonl \
        --device cuda:1 --teacher-device auto \
        --steps 600 --lr 1e-4 --grad-accum 8 --anneal-steps 600 \
        --eval-every 100 --holdout 64 --out-dir ./ckpt_projector_v6

训练中 holdout 指标是 **teacher-forcing 下 response 的 CE 与 token 准确率**，
比 KL 更贴近下游正确率；最终验收仍以 eval_math_acc.py 的 GSM8K 正确率为准。
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_eopd.fused_kv import FusedKVBuilder, FusedKVConfig, save_projector_ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--data-path", required=True,
                   help="gen_teacher_traj.py 产出的 jsonl（含 prompt / solution）")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--teacher-device", default="auto",
                   help="teacher 设备；显存紧张时用 auto 跨卡分片")
    p.add_argument("--teacher-gpus", default=None,
                   help="--teacher-device auto 时允许 teacher 占用的卡，逗号分隔（如 '4,5'）。"
                        "不填则自动排除 GPU0(VLLM) 与 student 卡——但那会把已被别的作业"
                        "占满的卡也算进来，导致 OOM，所以卡紧张时务必显式指定。")
    p.add_argument("--teacher-mem-per-gpu", default="7GiB")
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager"],
                   help="student/teacher 的 attention 实现。训练要反传过 student，"
                        "eager 会为每层保留完整 L×L 注意力矩阵，长序列下显存翻几倍；"
                        "sdpa 用 flash 路径省显存。评测脚本仍用 eager 保证逐 token 可复现。")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-prompt-len", type=int, default=384, help="题面 token 上限")
    p.add_argument("--max-answer-len", type=int, default=384, help="解题轨迹 token 上限")
    p.add_argument("--min-answer-len", type=int, default=8, help="短于此的轨迹丢弃")
    p.add_argument("--holdout", type=int, default=64,
                   help="文件前 N 条留作验收集，训练绝不使用")
    p.add_argument("--steps", type=int, default=600, help="优化器步数（非样本数）")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="对齐 C2C recipe 的 1e-4")
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--anneal-steps", type=int, default=600,
                   help="Gumbel 门控温度退火总步数（C2C 默认 1360；设成与 steps 同量级）")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-samples", type=int, default=32)
    p.add_argument("--out-dir", default="./ckpt_projector_v6")
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--projector-hidden", type=int, default=512)
    p.add_argument("--projector-inter", type=int, default=512)
    p.add_argument("--projector-layers", type=int, default=3)
    p.add_argument("--gate-init", type=float, default=1.0,
                   help="门控 logit 初值。C2C 原版是 0.0，但那是配合 1929 步退火 + 大数据"
                        "量训出来的。实测本 recipe 下 30 步后 logit 只挪了 ±0.0003、符号纯"
                        "属噪声 —— 而推理走的是硬门控 (logit>0)，等于抛硬币决定每层要不要"
                        "融合，比全开更糟。所以默认 +1.0：起步全开、且仍然可学（配合 "
                        "--gate-lr 足以在训练中被推到负值关掉某层）。这与 v1 的 fill_(3.0)"
                        "+use_gumbel=False『焊死不训』有本质区别。")
    p.add_argument("--gate-lr-mult", type=float, default=20.0,
                   help="门控 logit 的学习率倍率（相对 --lr）。全网只有 2×28=56 个门控标量，"
                        "又被 zero_init 的投影输出层挡住（proj_out=0 时 ∂L/∂gate 恒为 0，"
                        "要等投影学出信号才有梯度），用基础 lr 根本挪不动。放大它才能让"
                        "『哪些层该融合』真正成为学出来的结论而不是初值的残留。设 1.0 可关闭。")
    p.add_argument("--per-layer", dest="per_layer", action="store_true", default=True,
                   help="每层一个独立 projector（C2C 原版；默认开）")
    p.add_argument("--shared-projector", dest="per_layer", action="store_false",
                   help="所有层共用一个 projector（v1 的做法，仅用于消融对比）")
    p.add_argument("--zero-init", dest="zero_init", action="store_true", default=True,
                   help="投影输出层零初始化：融合从恒等(=student 自身)出发，训练只能改善")
    p.add_argument("--no-zero-init", dest="zero_init", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(path, dtype, device, max_memory=None, attn_impl="sdpa"):
    if device == "auto":
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, attn_implementation=attn_impl,
            device_map="auto", max_memory=max_memory).eval()
    return AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, attn_implementation=attn_impl).to(device).eval()


def build_samples(rows, tok, args):
    """把 {prompt, solution} 转成 C2C ChatDataset 那套 (input_ids, prefix_len)。

    【C2C 对齐】instruction 用 add_generation_prompt=True 渲染（末尾带 "<|im_start|>assistant"），
    full = instruction + solution tokens。前 len(instruction) 位是 prefix（融合 + 不计 loss），
    其后是 response（不融合 + 计 loss）——与 ChatDataset 的 labels/kv_cache_index 完全一致。
    这里不显式存 labels，训练循环用 prefix_len 切分即可（等价）。
    """
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
        # 轨迹末尾补 eos，让 student 学会「什么时候停」
        a_ids = tok(solution, add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None:
            a_ids = a_ids + [tok.eos_token_id]
        if len(a_ids) < args.min_answer_len:
            continue
        p_ids = p_ids[-args.max_prompt_len:]   # 题面从后往前截（保留 generation prompt）
        a_ids = a_ids[: args.max_answer_len]
        out.append((torch.tensor(p_ids + a_ids, dtype=torch.long), len(p_ids)))
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = getattr(torch, args.dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_f = open(log_path, "w")

    # ---- 模型 ----
    student_gpu = int(args.device.split(":")[1]) if args.device.startswith("cuda:") else 0
    teacher_max_memory = None
    if args.teacher_device == "auto":
        n_gpu = torch.cuda.device_count()
        if args.teacher_gpus:
            allow = {int(x) for x in args.teacher_gpus.split(",")}
        else:
            # 默认排除 GPU0(VLLM) 与 student 卡
            allow = {i for i in range(n_gpu)} - {0, student_gpu}
        teacher_max_memory = {
            i: (args.teacher_mem_per_gpu if i in allow else "1MiB") for i in range(n_gpu)
        }
        print(f"[load] teacher 分片可用卡 = {sorted(allow)} "
              f"(每卡上限 {args.teacher_mem_per_gpu})", flush=True)
    print(f"[load] student  <- {args.student} ({args.device})", flush=True)
    student = load_model(args.student, dtype, args.device, attn_impl=args.attn_impl)
    print(f"[load] teacher  <- {args.teacher} ({args.teacher_device})", flush=True)
    teacher = load_model(args.teacher, dtype, args.teacher_device,
                         max_memory=teacher_max_memory, attn_impl=args.attn_impl)
    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- 数据 ----
    print(f"[data] 读取 {args.data_path}", flush=True)
    rows = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # 【关键】按文件顺序切分，训练/验收零重叠
    hold_rows, train_rows = rows[: args.holdout], rows[args.holdout:]
    samples = build_samples(train_rows, tok, args)
    eval_samples = build_samples(hold_rows, tok, args)[: args.eval_samples]
    if not samples:
        raise RuntimeError("没有可用样本，检查 jsonl 是否含 prompt/solution 字段")
    avg_p = sum(p for _, p in samples) / len(samples)
    avg_a = sum(x.numel() - p for x, p in samples) / len(samples)
    print(f"[data] 训练 {len(samples)} 条 / holdout {len(eval_samples)} 条 | "
          f"平均题面 {avg_p:.0f} tok，平均轨迹 {avg_a:.0f} tok", flush=True)
    random.shuffle(samples)

    # ---- builder（per-layer projector + 可学习门控）----
    cfg = FusedKVConfig(
        dtype=dtype,
        projector_hidden_dim=args.projector_hidden,
        projector_intermediate_dim=args.projector_inter,
        projector_num_layers=args.projector_layers,
        zero_init=args.zero_init,
        per_layer_projector=args.per_layer,
        keep_last_token_unfused=True,   # 【C2C 对齐 2】与 eval 同口径
    )
    builder = FusedKVBuilder.from_models(teacher, student, cfg)
    builder.freeze_teacher_student()
    n_proj = len(builder.projectors)
    # 【C2C 对齐 3】门控只设初值，保持可学习 + Gumbel 退火，让模型自己决定哪层要融合
    builder.gate_params_to_fp32()   # 必须在 set_gate_logit 之前：bf16 标量存不下小更新
    builder.set_gate_logit(args.gate_init)
    for p in builder.projectors:
        p.use_gumbel = True
        p.anneal_steps = args.anneal_steps
        p.train()
    # 门控标量单独一组、更大 lr、不加 weight_decay
    #（weight_decay 会把 gate_logit 往 0 拽，正好卡在硬门控的翻转点上）
    gate_params = [q for p in builder.projectors
                   for q in (p.key_gate_logit, p.value_gate_logit)]
    gate_ids = {id(q) for q in gate_params}
    body_params = [q for p in builder.projectors for q in p.parameters()
                   if id(q) not in gate_ids]
    params = body_params + gate_params
    n_param = sum(q.numel() for q in params)
    print(f"[proj] {n_proj} 个 projector（per_layer={args.per_layer}）| "
          f"可训参数 {n_param/1e6:.1f}M | gate_init={args.gate_init} | "
          f"gate_lr={args.lr * args.gate_lr_mult:.1e} | "
          f"anneal_steps={args.anneal_steps}", flush=True)

    s_device = next(student.parameters()).device
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": gate_params, "lr": args.lr * args.gate_lr_mult, "weight_decay": 0.0},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))

    def pack(ids, P):
        T = ids.numel()
        x = ids.unsqueeze(0).to(s_device)
        am = torch.ones(1, T, dtype=torch.long, device=s_device)
        pos = torch.arange(T, dtype=torch.long, device=s_device).unsqueeze(0)
        return x, am, pos, T, P

    def response_ce(x, am, pos, T, P, fused: bool):
        """response 段的 CE 与 token 准确率。

        fused=True  : 用 fused 前缀 cache 做 teacher-forcing（实验组）
        fused=False : student 原生整段前向（baseline，不接触 teacher）
        两者预测的目标位置完全相同：x_{P}..x_{T-1}，由 logits[P-1..T-2] 预测。
        """
        gold = x[:, P:]                                       # (1, R)
        if fused:
            # need_teacher_logits=False：SFT 不需要 teacher 分布，teacher 也只看前缀
            fc, _ = builder.build_trainable(x, am, pos, P, need_teacher_logits=False)
            # cache 覆盖 x_0..x_{P-2}（长度 P-1），从 x_{P-1} 开始喂 —— 这一位
            # 正是 rollout 里 decode 首步用 student 自身 KV 重算的那个 token，
            # 训练/推理路径逐位对齐，不存在错位或重复。
            c = builder.trainable_cache_len(P)
            out = student(input_ids=x[:, c:T - 1], attention_mask=am[:, :T - 1],
                          position_ids=pos[:, c:T - 1],
                          past_key_values=fc, use_cache=True)
            logits = out.logits[:, -gold.size(1):, :]          # (1, R, V)
        else:
            logits = student(input_ids=x, attention_mask=am,
                             position_ids=pos).logits[:, P - 1:T - 1, :]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), gold.reshape(-1))
        acc = (logits.argmax(-1) == gold).float().mean()
        return ce, acc

    # ---- holdout 验收 ----
    @torch.no_grad()
    def holdout(fused: bool, force_gate: bool = False):
        """force_gate=True 时临时把门控全部拨开，测「投影本身学得怎么样」。

        为什么需要两个数：推理走硬门控 (logit>0)，gate_init=0 起步时若 logit 还没
        变正，真实口径下融合是关的、fused 恒等于 baseline —— 看不出投影有没有在学。
        强制常开这一路把「门控决策」和「投影质量」拆开，方便判断训练是否在推进。
        """
        for p in builder.projectors:
            p.eval()          # 硬门控，与 rollout 推理一致
        saved = None
        if force_gate:
            saved = [(p.key_gate_logit.clone(), p.value_gate_logit.clone())
                     for p in builder.projectors]
            builder.set_gate_logit(3.0)
        tot_ce = tot_acc = 0.0
        n = 0
        for ids, P in eval_samples:
            x, am, pos, T, PP = pack(ids, P)
            if T - PP < 1 or PP < 2:
                continue
            ce, acc = response_ce(x, am, pos, T, PP, fused)
            tot_ce += ce.item()
            tot_acc += acc.item()
            n += 1
        if saved is not None:
            for p, (k, v) in zip(builder.projectors, saved):
                p.key_gate_logit.copy_(k)
                p.value_gate_logit.copy_(v)
        for p in builder.projectors:
            p.train()
        return tot_ce / max(1, n), tot_acc / max(1, n)

    base_ce, base_acc = holdout(fused=False) if eval_samples else (float("nan"),) * 2
    print(f"[holdout] baseline(student 自身) response CE = {base_ce:.4f} | "
          f"token acc = {base_acc:.3f}  —— fused 必须 CE 更低 / acc 更高才算有效", flush=True)

    # ---- 训练循环 ----
    print(f"[train] steps={args.steps} lr={args.lr} grad_accum={args.grad_accum}", flush=True)
    global_step = 0
    idx = 0
    accum_ce = accum_acc = 0.0
    accum_n = 0
    running = 0.0

    while global_step < args.steps:
        ids, P = samples[idx % len(samples)]
        idx += 1
        x, am, pos, T, P = pack(ids, P)
        if T - P < 1 or P < 2:
            continue

        ce, acc = response_ce(x, am, pos, T, P, fused=True)
        (ce / args.grad_accum).backward()
        accum_ce += ce.item()
        accum_acc += acc.item()
        accum_n += 1

        if accum_n % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            # 【C2C 对齐 3】Gumbel 门控温度退火 1.0 → 0.001，训练末期趋近硬门控
            builder.update_temperature(global_step)

            avg_ce = accum_ce / accum_n
            avg_acc = accum_acc / accum_n
            running = 0.9 * running + 0.1 * avg_ce if running > 0 else avg_ce
            accum_ce = accum_acc = 0.0
            accum_n = 0

            if global_step % args.log_every == 0:
                lr_now = sched.get_last_lr()[0]
                temp = float(builder.projectors[0].gate_temperature)
                gk = [float(p.key_gate_logit) for p in builder.projectors]
                # 【必看】gate_on 是「推理时真正会融合的层数」——推理走硬门控 (logit>0)。
                # 如果它一路停在 0，说明训练结束后融合会被全关，pretrained 将等于 baseline。
                gate_on = sum(1 for g in gk if g > 0)
                print(f"step {global_step:>5d} | CE {avg_ce:.4f} | ema {running:.4f} | "
                      f"acc {avg_acc:.3f} | lr {lr_now:.2e} | gate_T {temp:.3f} | "
                      f"gate_on {gate_on}/{n_proj} (mean {sum(gk)/len(gk):+.4f})", flush=True)
                log_f.write(json.dumps({
                    "step": global_step, "ce": avg_ce, "ema_ce": running,
                    "token_acc": avg_acc, "lr": lr_now, "gate_temp": temp,
                    "gate_on": gate_on, "gate_key_mean": sum(gk) / len(gk),
                }) + "\n")
                log_f.flush()

            if args.eval_every and global_step % args.eval_every == 0 and eval_samples:
                hce, hacc = holdout(fused=True)                      # 真实推理口径（硬门控）
                oce, oacc = holdout(fused=True, force_gate=True)     # 门控强制常开
                print(f"  [holdout] step {global_step} | 推理口径 CE {hce:.4f} acc {hacc:.3f} "
                      f"({(base_ce-hce)/base_ce*100:+.1f}% CE) | 门控常开 CE {oce:.4f} "
                      f"acc {oacc:.3f} ({(base_ce-oce)/base_ce*100:+.1f}% CE) | "
                      f"base CE {base_ce:.4f} acc {base_acc:.3f}", flush=True)
                log_f.write(json.dumps({
                    "step": global_step, "holdout_fused_ce": hce, "holdout_base_ce": base_ce,
                    "holdout_fused_acc": hacc, "holdout_base_acc": base_acc,
                    "holdout_forcegate_ce": oce, "holdout_forcegate_acc": oacc,
                }) + "\n")
                log_f.flush()

            if global_step % args.save_every == 0:
                ckpt = os.path.join(args.out_dir, f"projector_step{global_step}.pt")
                save_projector_ckpt(builder.projectors, ckpt)
                print(f"[save] {ckpt}", flush=True)

    final = os.path.join(args.out_dir, "projector_final.pt")
    save_projector_ckpt(builder.projectors, final)
    # 门控最终取值：>0 的层才会在推理时真正融合，是判断"学到了什么"的直接证据
    gates = [(float(p.key_gate_logit), float(p.value_gate_logit)) for p in builder.projectors]
    n_on_k = sum(1 for k, _ in gates if k > 0)
    n_on_v = sum(1 for _, v in gates if v > 0)
    print(f"[gate] 推理时开启融合的层数: key {n_on_k}/{n_proj}, value {n_on_v}/{n_proj}")
    print(f"[gate] logits = {[(round(k,2), round(v,2)) for k, v in gates]}")
    print(f"[done] 最终 projector 已保存: {final}")
    print(f"[done] 训练日志: {log_path}")
    log_f.close()


if __name__ == "__main__":
    main()
