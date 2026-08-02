"""Batch greedy evaluation for a student-only HF checkpoint."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_eopd.eval_math_acc import extract_answer, is_correct


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def render(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_ids = {tokenizer.eos_token_id}
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0:
            eos_ids.add(token_id)
    with open(args.data_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()][: args.num_samples]
    results = []
    correct = 0
    total_len = 0
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        texts = [render(tokenizer, row["prompt"]) for row in chunk]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_len,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=list(eos_ids),
            )
        response_ids = generated[:, input_ids.shape[1] :]
        for offset, row in enumerate(chunk):
            tokens = response_ids[offset].tolist()
            stop = len(tokens)
            for pos, token_id in enumerate(tokens):
                if token_id in eos_ids or token_id == tokenizer.pad_token_id:
                    stop = pos
                    break
            tokens = tokens[:stop]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            pred = extract_answer(text)
            ok = is_correct(pred, row["label"])
            correct += int(ok)
            total_len += len(tokens)
            results.append({
                "idx": start + offset,
                "label": row["label"],
                "baseline": {"pred": pred, "correct": bool(ok), "len": len(tokens), "text": text},
            })
        print(f"[{min(start + len(chunk), len(rows))}/{len(rows)}] {correct}/{len(results)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"accuracy={correct}/{len(results)}={correct / max(1, len(results)):.4f}")
    print(f"mean_len={total_len / max(1, len(results)):.2f}")


if __name__ == "__main__":
    main()
