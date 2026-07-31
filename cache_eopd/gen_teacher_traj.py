"""
【C2C 核心】造「teacher 解题轨迹」训练集 —— projector 预训练的数据前置步骤。

为什么需要这个脚本：
    C2C 原版训练 projector 用的是 SFT：instruction = 题目，response = **真实答案**，
    loss 只算在 response token 上（rosetta/train/dataset_adapters.py:1503-1546）。
    也就是说 projector 学的是「prefix 里有了 teacher 的 KV 之后，能不能把这道题做对」。

    第一轮训练我用的是「把一段文本从中间劈开、后半段对齐 teacher logits」，
    信号里既没有正确答案也没有题目/答案边界，projector 只学到了 teacher 的说话风格
    —— GSM8K 正确率反而从 86.0% 掉到 80.0%（生成很流畅但读错题）。

    DAPO / GSM8K 这两个数据集都只有 {prompt, label}，没有解题过程。
    所以这里让 teacher 自己把题做一遍，**只保留答对的轨迹**，
    这就是「teacher 关于解题的那部分知识」的文本载体。

数据源选择（重要）：
    DAPO-Math-17k 是竞赛级难度，teacher 只能做对 18.8%（实测 16 题留 3 条），
    且题型与 GSM8K 评测集差距很大。GSM8K 官方 train split（7473 题）本地已缓存，
    与评测用的 300 题（全部来自 test split）**零重叠**，是更合适的训练源。
    默认用 gsm8k；--source dapo 可切回。

输出格式（jsonl，每行）：
    {"problem": 原始题干, "prompt": 套好模板的题面, "solution": teacher 的解题过程, "label": 标准答案}

用法：
    python -m cache_eopd.gen_teacher_traj \
        --teacher /path/Qwen3-4B --source gsm8k \
        --num-problems 1000 --batch-size 16 --max-new-tokens 512 \
        --out ./data/teacher_traj_gsm8k1000.jsonl
"""

import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_eopd.eval_math_acc import extract_answer, is_correct

# GSM8K-COT 评测集使用的题面模板。训练数据必须和评测**逐字节一致**，
# 否则 prompt 的 KV 就跟评测时不同，projector 学到的东西对不上号。
#
# 【坑 ×2】评测 jsonl 里这段模板有两处数据生成时留下的 bug，都必须原样复刻：
#   1. `{{}}` 双花括号 —— str.format 没转义干净。所以这里不能用 .format()
#      （会折叠成 `{}`），改用 replace 填题干。
#   2. `\boxed` 里的 `\b` 在 JSON 里是**退格符转义**，json.loads 后真的变成了
#      0x08 字符 —— 模型看到的是 "\x08oxed{{}}" 而不是 "\\boxed{{}}"。
#      下面显式写 \x08 而不是 "\b"，免得以后被当成笔误"修好"。
#   baseline 86.0% 与 pretrained 80.0% 都是在这个 prompt 下测的，两臂一致所以
#   对比有效；这里保持不动，只求训练/评测同源。
GSM8K_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \x08oxed{{}}.\n\n"
    "Problem: <PROBLEM>\n\nSolution:"
)


def render_prompt(problem: str) -> str:
    return GSM8K_TEMPLATE.replace("<PROBLEM>", problem)

# DAPO 题面外面裹了一层固定的指令头尾，需要剥掉只留题干
DAPO_HEAD = re.compile(
    r"^Solve the following math problem step by step\..*?\n\n", re.DOTALL)
DAPO_TAIL = re.compile(r"\n\nRemember to put your answer.*$", re.DOTALL)


def strip_wrapper(prompt: str) -> str:
    """把数据集自带的指令包装剥掉，只保留纯题干。"""
    body = DAPO_HEAD.sub("", prompt)
    body = DAPO_TAIL.sub("", body)
    # GSM8K-COT 的格式是 "Problem: xxx\n\nSolution:"，也一并剥掉
    body = re.sub(r"^Problem:\s*", "", body)
    body = re.sub(r"\n\nSolution:\s*$", "", body)
    return body.strip()


def load_problems(args):
    """返回 [{"problem": 纯题干, "label": 标准答案}, ...]。

    gsm8k: 官方 train split（本地缓存，离线可读）。answer 字段形如
           "...推理...\\n#### 72"，`####` 后面就是标准答案。
           注意只用 train split —— 评测的 300 题全部来自 test split，零重叠。
    dapo : 已有的 jsonl，{prompt, label}，需要剥掉指令包装。
    """
    if args.source == "gsm8k":
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        out = []
        for x in ds:
            ans = x["answer"].split("####")[-1].strip()
            out.append({"problem": x["question"].strip(), "label": ans})
        return out

    out = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "prompt" not in d or "label" not in d:
                continue
            out.append({"problem": strip_wrapper(d["prompt"]), "label": d["label"]})
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--source", default="gsm8k", choices=["gsm8k", "dapo"],
                   help="gsm8k=官方 train split(推荐)；dapo=竞赛级 jsonl(teacher 只做对约 19%%)")
    p.add_argument("--data-path", default=None, help="--source dapo 时必填")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto", help="teacher 放哪；显存紧张就用 auto 分片")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--num-problems", type=int, default=1000)
    p.add_argument("--skip", type=int, default=0, help="跳过前 N 条（便于分批续跑）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--log-every", type=int, default=10, help="每 N 个 batch 打一次进度")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    if args.source == "dapo" and not args.data_path:
        raise SystemExit("--source dapo 需要 --data-path")
    print(f"[data] 数据源 = {args.source}")
    records = load_problems(args)
    records = records[args.skip: args.skip + args.num_problems]
    print(f"[data] 取 {len(records)} 条题目（skip={args.skip}）")

    print(f"[model] 加载 teacher {args.teacher} (device={args.device})")
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # 批量生成必须左 padding，否则 padding 会挤在 prompt 和生成之间
    tok.padding_side = "left"

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=dtype,
        device_map=args.device if args.device == "auto" else None,
    )
    if args.device != "auto":
        teacher = teacher.to(args.device)
    teacher.eval()
    t_device = next(teacher.parameters()).device

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fout = open(args.out, "w")

    n_kept = n_total = 0
    t0 = time.time()
    for bi in range(0, len(records), args.batch_size):
        batch = records[bi: bi + args.batch_size]
        problems = [r["problem"] for r in batch]
        prompts = [render_prompt(p) for p in problems]
        # enable_thinking=False：Qwen3 默认会吐 <think> 块，会污染轨迹也吃 token 预算
        texts = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
            for p in prompts
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_prompt_len, add_special_tokens=False)
        enc = {k: v.to(t_device) for k, v in enc.items()}

        with torch.no_grad():
            out = teacher.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=False,                      # 贪心：轨迹要可复现
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].size(1):]

        for rec, problem, prompt, g in zip(batch, problems, prompts, gen):
            n_total += 1
            solution = tok.decode(g, skip_special_tokens=True).strip()
            pred = extract_answer(solution)
            # 【关键】只保留 teacher 答对的轨迹——答错的轨迹会把错误推理教给 projector
            if not is_correct(pred, rec["label"]):
                continue
            n_kept += 1
            fout.write(json.dumps({
                "problem": problem,
                "prompt": prompt,
                "solution": solution,
                "label": rec["label"],
            }, ensure_ascii=False) + "\n")
        fout.flush()

        if (bi // args.batch_size) % args.log_every == 0:
            el = time.time() - t0
            rate = n_kept / max(1, n_total)
            print(f"[gen] {n_total}/{len(records)} 题 | 保留 {n_kept} "
                  f"({rate:.1%}) | {el:.0f}s", flush=True)

    fout.close()
    print(f"\n[done] 共 {n_total} 题，teacher 答对并保留 {n_kept} 条 "
          f"({n_kept / max(1, n_total):.1%})，写入 {args.out}")
    print(f"[done] 耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
