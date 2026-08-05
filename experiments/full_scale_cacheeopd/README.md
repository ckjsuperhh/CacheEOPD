# Full-scale EOPD + CacheEOPD experiments

This directory is a portable experiment handoff for a VM or another GPU host.
It contains no model weights and no machine-specific absolute paths.

## Current evidence

The strict HF `EOPD + CacheEOPD` pipeline is runnable with the official Qwen3-0.6B to
Qwen3-4B fuser, frozen projector weights, student rollout, teacher-side EOPD signals,
and student-only evaluation. The current 300-step pilot uses three strategies:

| Strategy | Rollout policy | Status |
| --- | --- | --- |
| `fused` | Always teacher-KV fused | Completed training; evaluation recorded in `cache_eopd/PROGRESS.md` |
| `mixed` | Fused with probability `0.5` | Completed training; second-seed evaluation was still running at handoff |
| `linear_anneal` | Probability `1.0 -> 0.0` over 300 steps | Completed training; evaluation recorded in `cache_eopd/PROGRESS.md` |

The completed `mixed-41717` student-only curve is:

```text
step 50/100/150/200/250/300: 305/311/316/312/317/310 (out of 500)
mean: 62.37%
```

This is not yet evidence of a stable improvement over EOPD. The pilot also started a
paired plain-vs-fused rollout diagnostic; only after that diagnostic should `adaptive`
selection be promoted to a full training arm.

## Files

- `EXPERIMENT_PLAN.md`: frozen comparison protocol, fairness rules, and scale-up gates.
- `env.example`: paths and resource variables to fill on the VM.
- `run_cacheeopd_hf.sh`: portable strict HF CacheEOPD launcher.
- `run_rollout_gain_eval.sh`: paired plain/fused rollout diagnostic launcher.
- `summarize_eval.py`: summarizes student-only JSONL results and paired rollout records.

## Quick start

```bash
cd CacheEOPD
cp experiments/full_scale_cacheeopd/env.example experiments/full_scale_cacheeopd/.env
${EDITOR:-vi} experiments/full_scale_cacheeopd/.env
bash experiments/full_scale_cacheeopd/run_cacheeopd_hf.sh mixed 41719
```

The launcher requires the repository root on `PYTHONPATH`, an official fuser directory,
the Qwen3 student and teacher paths, and a JSONL dataset with a `prompt` field. It saves
checkpoints every `SAVE_EVERY` steps. Evaluation must load only the student checkpoint;
do not pass a teacher or projector to the student-only evaluator.
