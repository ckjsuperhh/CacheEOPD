# EOPD vs CacheEOPD: paper-scale protocol

This directory contains only the two comparison arms needed for the main result:
standard EOPD and CacheEOPD. Both use Qwen3-1.7B-Base as student, Qwen3-8B as
teacher, MATH training data, the same EOPD loss, and vLLM rollout. CacheEOPD is
identical except that its rollout requests carry C2C-fused teacher/student KV.

The fixed EOPD settings are taken from the local paper reproduction record
`EOPD复现.md`: teacher top-k 16, entropy threshold 0.8, soft-KD coefficient 1.0,
batch/mini-batch 128/32, three epochs, and final evaluation with temperature 1.0,
top-p 0.8, 8192 generated tokens, and 8 samples per problem.

## 1. Install

```bash
cd CacheEOPD
bash experiments/full_scale_cacheeopd/setup_conda_env.sh
conda activate cacheeopd
```

The default installation includes vLLM and `math-verify`. CacheEOPD requires
vLLM 0.13 or later because it uses its V1 KV connector API. Set
`INSTALL_FLASH_ATTN=1` only when the host compiler/CUDA toolchain can build
FlashAttention. Set `TORCH_INDEX_URL=` if PyTorch is already managed by the VM.

## 2. Prepare MATH

```bash
python examples/data_preprocess/math_dataset.py \
  --local_save_dir /data/cacheeopd/math
```

Set `TRAIN_FILE` and `VAL_FILE` in `.env` to the generated `train.parquet` and
`test.parquet`. These parquet rows provide `prompt` and rule-based ground truth
for EOPD's teacher-side supervision.

## 3. Prepare and train the projector

C2C trained its projectors on `teknium/OpenHermes-2.5`. Convert that corpus into
the prefix/response JSONL format used by this repository, then freeze the trained
projector before either EOPD run.

```bash
python -m cache_eopd.prepare_c2c_projector_data \
  --dataset teknium/OpenHermes-2.5 \
  --split train \
  --num-samples 500000 \
  --out /data/cacheeopd/openhermes_c2c_projector.jsonl

cp experiments/full_scale_cacheeopd/env.example \
  experiments/full_scale_cacheeopd/.env
# Fill every absolute path in .env before continuing.
bash experiments/full_scale_cacheeopd/train_projector_8b_to_1p7b.sh
```

The projector script follows the C2C recipe: all student/teacher weights stay
frozen; it trains a per-student-layer C2C projector with a three-layer 1024-wide
MLP, `1e-4` learning rate, `0.01` weight decay, gradient accumulation 8,
last-aligned layer mapping, and a learned annealed gate. It writes
`projector_step*.pt` and `projector_final.pt`; set `PROJECTOR_PATH` to the chosen
checkpoint. The first 5,000 converted rows are held out from projector updates.

## 4. Run the two baselines

```bash
bash experiments/full_scale_cacheeopd/run_eopd_cacheeopd_vllm.sh eopd 42
bash experiments/full_scale_cacheeopd/run_eopd_cacheeopd_vllm.sh cacheeopd 42
```

Run them separately or on equivalent allocations. Do not change model paths,
MATH files, GPU count, seed, context limits, optimizer budget, checkpoint cadence,
or EOPD hyperparameters between the two arms. The only intentional difference is
`rollout.c2c.enable=True` plus the frozen projector in CacheEOPD.

### Current CacheEOPD vLLM status

The EOPD command is ready to run. The CacheEOPD command intentionally **does not
yet constitute a full training run**: the checked-in vLLM connector correctly imports
an already-built fused KV packet and has passed an engine-level smoke test, but the
training server does not yet build a fresh packet from the **current synchronized
student weights** for each rollout request. The launcher therefore hard-fails when
packet metadata is absent rather than silently running ordinary EOPD.

This is an implementation boundary, not an experimental result. A valid online
implementation must preserve this ordering for every request:

```text
current student weights -> student prefix KV
teacher prefix KV + frozen C2C projector -> fused student-shaped prefix KV
fused prefix KV -> vLLM paged cache -> last prompt token prefill -> student rollout
```

Do not use a packet built from the initial Hugging Face checkpoint for a multi-step
run: after the first optimizer update it no longer represents the student that is
being trained. Complete the online model-runner integration and its current-weight
smoke before invoking the CacheEOPD arm at paper scale.

## 5. Evaluate

Merge an FSDP checkpoint to Hugging Face format, then run the paper protocol on
MATH500, AMC23, Minerva, OlympiadBench, AIME24, and AIME25: zero-shot prompts,
temperature 1.0, top-p 0.8, max new tokens 8192, and eight independent samples
per problem. Report both Avg@8 and Pass@8. Evaluation loads the merged student
checkpoint only: it must not load the teacher or projector and must not inject
teacher KV.

The repository includes preprocessors for MATH500, AIME24, and AIME25 under
`examples/data_preprocess/`. Add the remaining benchmark parquet files in the
same `{prompt, reward_model: {ground_truth}}` schema before evaluation.

## Files

- `setup_conda_env.sh`: Conda plus vLLM environment setup.
- `run_eopd_cacheeopd_vllm.sh`: the only two training arms.
- `train_projector_8b_to_1p7b.sh`: frozen C2C projector training wrapper.
- `env.example`: all required paths and fixed comparison parameters.
- `EXPERIMENT_PLAN.md`: fairness and reporting gates.
