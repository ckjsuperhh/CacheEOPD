# Experiment plan

## Hypothesis

Teacher KV may improve the state from which the student rolls out, but the benefit is
useful only when it transfers into student weights. The main test is therefore not
teacher-assisted accuracy; it is student-only accuracy after training.

## Frozen comparison

Run all arms with the same student, teacher, official projector, prompts, response token
budget, optimizer updates, data order policy, checkpoint interval, and evaluation set.

1. `plain_sft`: offline teacher trajectory response CE, student KV only.
2. `eopd`: student on-policy rollout plus official EOPD loss without KV injection.
3. `cacheeopd_mixed`: EOPD loss, fused rollout probability `0.5`.
4. `cacheeopd_adaptive`: EOPD loss, choose fused or plain rollout using a frozen scoring rule.

The adaptive arm is conditional. First run the paired rollout diagnostic. During training,
ground-truth labels may be used for offline analysis only, never to select trajectories.
The training selector should use normalized teacher sequence log-probability, EOPD
advantage, and an explicit margin. The selected rollout context must also be used for both
old and current student log-probabilities.

## Required reporting

For each seed and checkpoint, report:

- accuracy, answer extraction failures, and mean response length;
- paired `both_correct`, `plain_only_correct`, `fused_only_correct`, and `both_wrong`;
- fused rollout rate, fallback rate, teacher score margin, and score/correctness correlation;
- optimizer updates, response tokens, wall-clock time, and peak GPU memory;
- best checkpoint and fixed-step checkpoint separately.

## Scale-up gates

Do not spend full-scale resources until these checks pass:

1. `eval_fused_kv` zero-init sanity passes for the selected model pair.
2. Paired rollout evaluation uses explicit `last_aligned` mapping and student-only final evaluation.
3. Three seeds complete a small pilot without NaN, checkpoint loss, or duplicate runners.
4. The same token/update budget is used for EOPD and CacheEOPD.
5. Adaptive selection has positive paired coverage, or it is removed from the main run.

## Suggested stages

- Smoke: 10 steps, 32 prompts, one GPU.
- Pilot: 300 steps, 500 evaluation questions, seeds `41717`, `41718`, `41719`.
- Medium: 1,000--2,000 updates, 2,048-token context, three seeds.
- Large: 4,096-token context and the agreed data scale, only after the medium report.

## Important interpretation

KV fusion adds teacher forward and projector work to rollout. It cannot make an individual
update cheaper. Its only efficiency claim can be fewer updates to reach a target quality;
measure this with time-to-threshold and token-normalized curves, not raw step numbers.
