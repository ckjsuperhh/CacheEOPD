"""Convert the C2C OpenHermes corpus to projector-training JSONL."""

from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset


def normalize_role(role: str) -> str | None:
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def extract_messages(row: dict) -> list[dict[str, str]]:
    raw_messages = row.get("conversations") or row.get("messages") or []
    messages = []
    for raw in raw_messages:
        role = normalize_role(str(raw.get("from", raw.get("role", ""))))
        content = raw.get("value", raw.get("content", ""))
        if role and str(content).strip():
            messages.append({"role": role, "content": str(content).strip()})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="teknium/OpenHermes-2.5")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=500_000)
    parser.add_argument("--max-messages", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    limit = min(len(dataset), args.num_samples)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    kept = 0
    with open(args.out, "w", encoding="utf-8") as output:
        for index in range(limit):
            messages = extract_messages(dataset[index])
            if len(messages) < 2 or len(messages) > args.max_messages:
                continue
            assistant_positions = [
                position for position, message in enumerate(messages)
                if message["role"] == "assistant"
            ]
            if not assistant_positions:
                continue
            assistant_position = assistant_positions[-1]
            context = messages[:assistant_position]
            solution = messages[assistant_position]["content"]
            if not context or context[-1]["role"] != "user":
                continue
            record = {
                "prompt": context[-1]["content"],
                "messages": context,
                "solution": solution,
                "source_index": index,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[done] scanned={limit} kept={kept} output={args.out}")


if __name__ == "__main__":
    main()
