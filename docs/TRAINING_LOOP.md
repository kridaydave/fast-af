# Minimal diffusion training loop

`scripts/train_diffusion.py` is a CPU/CUDA training pipeline for an existing
`training-mixture/v1` bundle. It is deliberately separate from `bench/`, the
decode/cache experiments, and evaluation: the only metric it computes is masked
cross-entropy on the bundle's own validation split.

## Usage

```bash
# Tiny synthetic fixture used by the tests (no network, a few seconds)
python -m pytest tests/test_train_diffusion.py -q

# Real bundle, small CPU run; writes checkpoints and config.json under --out
python scripts/train_diffusion.py \
  --bundle /home/kriday/Code/AI/fast-af-boi/artifacts/training-v1/final \
  --out /tmp/train-run --steps 100 --warmup 10 \
  --batch-size 8 --checkpoint-every 50 --eval-every 50

# GPU run (validated on a Colab T4); fp32, no AMP
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run \
  --device cuda --steps 100 --warmup 10 --batch-size 8 \
  --checkpoint-every 50 --eval-every 50

# CUDA fp16 autocast with fp32 master weights
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run \
  --device cuda --amp --grad-accum 8 --batch-size 8

# P1 guard: exits 2 listing missing prerequisites (D1 is not implemented)
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run --p1

# Resume from an exact checkpoint; the config hash must match
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run \
  --device cuda --resume /tmp/train-run/checkpoint-000050.pt
```

The bundle is never modified. Shards are opened read-only with `np.memmap`.

## What it does

- **Manifest validation.** Checks the `training-mixture/v1` schema, canonical
  manifest fingerprint, `<u2` dtype, shard existence and exact byte size, and
  optionally SHA-256 shard hashes (`--verify-hashes`). The tokenizer manifest is
  fingerprint-checked and supplies the vocabulary size.
- **Shard loading.** `ShardLoader` concatenates nothing: it iterates rows of one
  packed shard at a time, yielding contiguous `[batch, seq_len]` int64 batches.
  Each epoch uses a shard permutation derived only from `(seed, epoch)`. The
  resumable cursor is `(epoch, position, sequence)`; the last batch of a shard
  may be partial. Target IDs are checked to be in `[1, vocab)` and free of
  MASK/PAD/UNK/BOS.
- **Dynamic masking.** `diffusion_mask` draws a per-sequence rate uniformly
  from `[mask_min, mask_max]` (default `0.10–0.90`), then replaces each token
  independently with that probability. EOS and reserved control ids are never
  masked. A row with at least one eligible token is guaranteed one masked token
  so the loss is defined; stored targets are never modified. Masking is driven
  by the global torch RNG, whose state is checkpointed.
- **Objective.** Mean cross-entropy over masked positions only, token-weighted
  across the global batch. A row may be heavily masked; an empty batch-wide
  mask is rejected.
- **Model.** A small bidirectional pre-norm transformer with tied token
  embeddings and learned absolute positions, plus a final LayerNorm and
  vocabulary projection. This is a pipeline smoke model, not the spec's GDN
  hybrid or a block-diffusion decoder. `profile=d1` is rejected explicitly.
- **Schedule and optimizer.** Linear warmup, then cosine decay to 10% of peak
  LR. AdamW with recipe-style groups: weight decay on matrices only, not norms,
  biases, embeddings, or positions. Optional CUDA AMP and gradient accumulation.
- **Checkpoints.** Atomic writes (sibling `.tmp`, fsync, rename) plus a SHA-256
  sidecar containing model, optimizer, scheduler, AMP scaler, loader cursor,
  step, tokens seen, config hash, corpus/tokenizer provenance (manifest,
  tokenizer, and shard hashes), library versions, git commit, GPU, and
  Python/NumPy/PyTorch/CUDA RNG states. `checkpoint-last.txt` names the newest
  checkpoint. Resuming rejects a different config hash, corpus provenance, or
  cursor/token inconsistency.

## Assumptions

- A bundle is a directory with `manifest.json`, `tokenizer_manifest.json`, and
  `.bin` shards as produced by `scripts/build_training_mix.py`.
- `MASK=0`, `PAD=2`, `UNK=3`, `BOS=4` are control IDs; `EOS=1` is a legal
  target. Stored shards contain no control IDs other than EOS.
- A per-sequence rate with per-token Bernoulli corruption is an adequate
  dynamic-diffusion objective for pipeline validation. The spec's 32-token
  blocks, 8-16 denoising rounds, and few-round consistency training are not
  implemented here.
- CPU execution with one torch thread is the reproducible default. Determinism
  covers the same machine, library versions, and config; the T4 smoke used
  fp32. GPU bitwise reproducibility is not guaranteed.
- Validation is a fixed batch count (default 8) from the validation split with
  a separate generator, or the whole split with `--eval-full`; the loss is
  token-weighted.

## Recipe alignment (draft `TRAINING_RECIPE_T4_PILOT.md`, 2026-11-13)

Implemented against the draft:

- Classical masking is now per-token Bernoulli with one rate per sequence drawn
  from `U(0.10, 0.90)`; EOS and reserved ids are excluded. The old
  exact-count/min-1 scheduler is gone. `mask_mode=classical` is recorded in the
  run config; blockwise masking is rejected as an unimplemented gap.
- Gradient accumulation (`--grad-accum`) and CUDA AMP (`--amp`) are wired into
  the training loop. AMP uses fp16 autocast with fp32 master weights and a
  `GradScaler`; its state is checkpointed.
- Optimizer groups follow the recipe: AdamW `betas=(0.9, 0.95)`, `eps=1e-8`,
  weight decay only on matrices.
- Validation can consume the whole split (`--eval-full`) and is token-weighted.
- Checkpoints bind manifest, tokenizer manifest, tokenizer model, and shard
  hashes; they carry library versions, git commit, GPU, step/tokens, and RNG
  states. Saves are fsynced and have SHA-256 sidecars; resume validates all of
  it plus the loader cursor and token/step consistency.
- `--p1` is a hard guard. It refuses to run without `profile=d1`, `device=cuda`,
  `amp=true`, `eval_full=true`, `verify_hashes=true`, and a 64-sequence global
  batch. `profile=d1` itself raises `NotImplementedError` because the D1
  architecture is not implemented.

Remaining gaps, explicitly not claimed as D1-ready:

- The model is still the ~2.2M-param smoke transformer, not the D1 100M GDN
  hybrid (16 blocks, d_model 640, 12 GDN + 4 GQA attention, SwiGLU, RMSNorm).
- CUDA AMP, gradient accumulation, and the new masking were not re-run on a GPU
  after this alignment; the last GPU smoke predates them. No P1 throughput,
  loss, or convergence claim is made.
- No blockwise-aligned masking, denoiser-round unrolling, fp16 master-weight
  duplication beyond AMP, production checkpoint retention, full-corpus run,
  gPPL/MBPP, or decode-policy integration.
- The draft recipe may still change; this code follows the 2026-11-13 draft.

## Validation

- `python -m pytest tests/test_train_diffusion.py -q` → **16 passed** on the dev
  box. Coverage: seeded epoch coverage and loader resume; per-token Bernoulli
  masking with EOS/reserved exclusion and configurable bounds; masked-only loss;
  loss decreases on a tiny fixture; exact CPU resume; config-hash, provenance,
  cursor, and token-consistency rejection; optimizer groups; gradient
  accumulation; full-split validation; checkpoint sidecar/tamper detection; CLI
  smoke and P1 guard; and a real `training-v1/final` manifest/shard/tokenizer
  load plus one finite forward.
- Tiny CPU fixture (640 train tokens, 320 validation tokens, 117,312 params):
  first losses `[66.143, 64.489, 65.180, 62.762, 57.006]`, last
  `[2.580, 2.234, 2.419, 2.492, 2.101]`, validation `[3.535, 3.350]`.
- Full suite: **109 passed, 1 failed**. The failure is pre-existing and
  environment-specific: `tests/test_token_baseline.py` invokes `sys.executable
  -S`, and in this T3/AppImage environment `sys.executable` is the T3 binary,
  which rejects `-S`. It is unrelated to the training files.

### Colab T4 GPU smoke (2026-11-13, before the recipe-alignment changes)

Run through the Colab CLI on a `Tesla T4` (sm75), `torch 2.14.1+cu130`, fp32,
one micro-batch of 8 sequences. It used the old exact-count masker; the current
Bernoulli masker, AMP, and gradient accumulation have not been re-run on GPU.
The uploaded corpus was a one-train-shard subset of `training-v1/final` plus its
full validation shard: **10,485,760 train tokens and 983,040 validation tokens**,
not the full 99,983,360-token corpus.

The model was the pipeline's small bidirectional transformer, **2,213,632
params** (32,000 vocab, 1024 context, d_model 64, 2 layers, 4 heads). This is
not the D1 100M architecture, and the run makes no D1 performance claim.

- 100 steps, 819,200 tokens, 9.71 s, about 84,300 tokens/s (single T4, fp32,
  including data loading and evaluation).
- Training loss `44.873 → 18.055`; validation loss `20.029 → 18.032`.
- Checkpoint/resume: loader cursor, scheduler, step (100) and tokens seen
  (819,200) matched exactly. Loss trajectories differed by at most `3.81e-6`
  and final model tensors by at most `4.53e-5`. An independent fresh repeat of
  the same run differed by `3.81e-6` in loss and `8.15e-5` in weights, so the
  residual is CUDA kernel nondeterminism, not a resume-state error. CUDA RNG
  state was present in the checkpoint.
- Throughput is not comparable to the D1 recipe's P0 gate: the model is ~2.2M
  params, not ~100M, and the corpus is a single shard.

## Limitations

- The D1 100M architecture is not implemented. The model is a ~2.2M-param smoke
  transformer; `--p1` and `profile=d1` refuse to run rather than mislabel it.
- AMP and gradient accumulation are implemented but were not re-validated on a
  GPU after this alignment. The last GPU run was fp32 with the old masker.
- No distributed training, fused optimizer, or production checkpoint retention.
- GPU bitwise reproducibility is not guaranteed; CPU resume is exact in tests.
- No block-diffusion decode objective, remasking schedule, or denoiser-step
  sampling; this trains a masked-token predictor, not the full D1 recipe.
- Validation is a sample by default; `--eval-full` covers one full pass but is
  not run at 100M-token scale here.
- Resume determinism is verified on CPU exactly and on one T4 (pre-alignment) to
  floating-point tolerance; multi-GPU and other CUDA devices were not tested.
- The loader holds one memmap per shard at a time and validates target ranges on
  every batch, which is intentionally simple rather than throughput-optimal.
- Training throughput and convergence at 100M-token scale were not measured.