# Experiment plan

## Hypothesis

Teacher KV may improve the state from which the student rolls out, but the benefit is
useful only when it transfers into student weights. The main test is therefore not
teacher-assisted accuracy; it is student-only accuracy after training.

## Frozen comparison

Run only the following two arms with the same student, teacher, prompts, response token
budget, optimizer updates, data order policy, checkpoint interval, and evaluation set.

1. `eopd`: student on-policy rollout plus the official EOPD loss, with ordinary student KV.
2. `cacheeopd`: exactly the same EOPD loss and budget, but its rollout starts from a
   C2C-fused teacher/student KV prefix produced by a frozen projector.

`plain_sft`, mixed-probability injection, annealing, adaptive selection, and any
teacher-assisted evaluation are deliberately outside this full-scale package. They are
pilot ablations, not part of the requested EOPD-versus-CacheEOPD comparison.

## Required reporting

For each seed and checkpoint, report:

- accuracy, answer extraction failures, and mean response length;
- paired `both_correct`, `plain_only_correct`, `fused_only_correct`, and `both_wrong`;
- fused rollout rate and a hard failure count (a CacheEOPD request must never silently
  fall back to ordinary student KV);
- optimizer updates, response tokens, wall-clock time, and peak GPU memory;
- best checkpoint and fixed-step checkpoint separately.

## Scale-up gates

Do not spend full-scale resources until these checks pass:

1. `eval_fused_kv` zero-init sanity passes for the selected model pair.
2. Paired rollout evaluation uses explicit `last_aligned` mapping and student-only final evaluation.
3. Three seeds complete a small pilot without NaN, checkpoint loss, or duplicate runners.
4. The same token/update budget is used for EOPD and CacheEOPD.
5. CacheEOPD has passed an online vLLM rollout smoke using the current synchronized
   student weights. A static KV-packet smoke is necessary but not sufficient.

## Suggested stages

- Smoke: 10 steps, 32 prompts, one GPU for EOPD; CacheEOPD also needs a current-weight
  online injection smoke.
- Pilot: 300 steps, 500 evaluation questions, seeds `41717`, `41718`, `41719`.
- Medium: 1,000--2,000 updates, 2,048-token context, three seeds.
- Large: 4,096-token context and the agreed data scale, only after the medium report.

## Important interpretation

KV fusion adds teacher forward and projector work to rollout. It cannot make an individual
update cheaper. Its only efficiency claim can be fewer updates to reach a target quality;
measure this with time-to-threshold and token-normalized curves, not raw step numbers.
