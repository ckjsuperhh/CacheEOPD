#!/usr/bin/env python3
"""Summarize student-only or paired plain/fused JSONL evaluation files."""

import json
import pathlib
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT.jsonl")
    path = pathlib.Path(sys.argv[1])
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("empty result file")
    keys = [key for key in rows[0] if isinstance(rows[0][key], dict) and "correct" in rows[0][key]]
    print(f"file={path} n={len(rows)}")
    for key in keys:
        correct = sum(bool(row[key]["correct"]) for row in rows)
        print(f"{key}: {correct}/{len(rows)}={correct / len(rows):.4%}")
    if {"baseline", "official_fuser"}.issubset(keys):
        both = fused_only = plain_only = neither = 0
        for row in rows:
            plain = bool(row["baseline"]["correct"])
            fused = bool(row["official_fuser"]["correct"])
            if plain and fused:
                both += 1
            elif fused:
                fused_only += 1
            elif plain:
                plain_only += 1
            else:
                neither += 1
        print(f"both_correct={both}")
        print(f"fused_only_correct={fused_only}")
        print(f"plain_only_correct={plain_only}")
        print(f"both_wrong={neither}")


if __name__ == "__main__":
    main()
