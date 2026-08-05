# Results snapshot

Snapshot date: 2026-08-05. All counts below use the fixed 500-question GSM8K
student-only evaluator unless stated otherwise.

## Strict EOPD + CacheEOPD

The official projector is frozen. Training uses student rollout, teacher-side EOPD
signals, clipped reverse-KL policy gradient, and entropy-gated top-k forward-KL.

### Mixed probability 0.5

| Seed | Step 50 | 100 | 150 | 200 | 250 | 300 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41717 | 305 | 311 | 316 | 312 | 317 | 310 | 311.8 |
| 41718 | 305 | 303 | 307 | pending | pending | pending | pending |

The first seed is `61.0/62.2/63.2/62.4/63.4/62.0%` across checkpoints. The second
seed's remaining checkpoints were still being evaluated at handoff.

### Other strict CacheEOPD arms

| Strategy / seed | Step 50 | 100 | 150 | 200 | 250 | 300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fused / 41717 | 309 | 295 | 305 | 304 | 303 | pending |
| Fused / 41718 | 307 | 314 | 296 | 300 | 308 | pending |
| Linear anneal / 41717 | 310 | 304 | 308 | 320 | 314 | pending |

The strict pilot does not yet show a stable quality gain over EOPD. It does show that
the end-to-end training, checkpointing, and independent evaluation path is operational.

## Reference curves

The earlier offline teacher-trajectory SFT mixed experiment averaged approximately:

```text
step 50/100/150/200/250/300: 61.4/61.8/62.3/62.3/63.3/63.4%
```

The previously recorded two-seed EOPD reference curve was:

```text
step 50/100/150/200/250/300: 62.0/63.1/62.5/61.8/64.2/64.2%
```

These curves are useful diagnostics, not a substitute for a fresh matched-budget run.
Raw step counts are not comparable when gradient accumulation, response tokens, or
rollout counts differ.

## Rollout-gain diagnostic

A paired 500-question plain-vs-fused greedy rollout diagnostic is running from the base
student. It records the two answers for each question and will report:

```text
both_correct, fused_only_correct, plain_only_correct, both_wrong
```

The zero-init path sanity check passed `5/5`. The diagnostic is the decision gate for an
adaptive rollout arm. Ground-truth labels are used only for this offline analysis; an
actual training selector must use frozen teacher scores or another label-free signal.
