"""Join base, plain SFT, EOPD, and CacheEOPD evaluation JSONL files."""

import argparse
import json
import re
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--plain", required=True)
    parser.add_argument("--eopd", required=True)
    parser.add_argument("--cacheeopd", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def method_result(row, method):
    if method == "cacheeopd":
        return row.get("pretrained", row.get("baseline"))
    return row.get("baseline", row.get("pretrained"))


def problem_from_prompt(prompt):
    match = re.search(r"Problem:\s*(.*?)\n\nSolution:", prompt, re.S)
    return match.group(1).strip() if match else prompt


def stats(rows, left, right):
    counts = Counter()
    for row in rows:
        a = row[left]["correct"]
        b = row[right]["correct"]
        counts[(a, b)] += 1
    return {
        "both_correct": counts[(True, True)],
        "both_wrong": counts[(False, False)],
        f"{left}_correct_{right}_wrong": counts[(True, False)],
        f"{left}_wrong_{right}_correct": counts[(False, True)],
    }


def main():
    args = parse_args()
    data = read_jsonl(args.data)
    plain = {row["idx"]: row for row in read_jsonl(args.plain)}
    eopd = {row["idx"]: row for row in read_jsonl(args.eopd)}
    cacheeopd = {row["idx"]: row for row in read_jsonl(args.cacheeopd)}
    rows = []
    for idx, data_row in enumerate(data):
        if idx not in plain or idx not in eopd or idx not in cacheeopd:
            continue
        methods = {
            "plain": method_result(plain[idx], "plain"),
            "eopd": method_result(eopd[idx], "eopd"),
            "cacheeopd": method_result(cacheeopd[idx], "cacheeopd"),
        }
        rows.append(
            {
                "idx": idx,
                "problem": problem_from_prompt(data_row["prompt"]),
                "label": data_row["label"],
                **methods,
            }
        )
    for row in rows:
        row["category"] = (
            "all_correct"
            if all(row[name]["correct"] for name in ("plain", "eopd", "cacheeopd"))
            else "all_wrong"
            if not any(row[name]["correct"] for name in ("plain", "eopd", "cacheeopd"))
            else "cacheeopd_correct_eopd_wrong"
            if row["cacheeopd"]["correct"] and not row["eopd"]["correct"]
            else "eopd_correct_cacheeopd_wrong"
            if row["eopd"]["correct"] and not row["cacheeopd"]["correct"]
            else "mixed"
        )
    summaries = {
        "num_rows": len(rows),
        "accuracy": {
            name: {
                "correct": sum(row[name]["correct"] for row in rows),
                "total": len(rows),
                "mean_len": sum(row[name].get("len", 0) for row in rows) / max(len(rows), 1),
            }
            for name in ("plain", "eopd", "cacheeopd")
        },
        "paired": {
            "cacheeopd_vs_eopd": stats(rows, "cacheeopd", "eopd"),
            "cacheeopd_vs_plain": stats(rows, "cacheeopd", "plain"),
            "eopd_vs_plain": stats(rows, "eopd", "plain"),
        },
        "categories": dict(Counter(row["category"] for row in rows)),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
