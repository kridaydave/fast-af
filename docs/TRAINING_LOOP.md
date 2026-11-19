# Minimal diffusion training loop

`scripts/train_diffusion.py` is a CPU/CUDA training pipeline for an existing
`training-mixture/v1` bundle. It is deliberately separate from `bench/`, the
decode/cache experiments, and evaluation: the only metric it computes is masked
cross-entropy on the bundle's own validation split.

As of 2026-11-14, `--profile d1` implements the forward-GDN/GQA hybrid and
blockwise objective. The default remains the original small smoke transformer.
See [D1 architecture and usage](D1_ARCHITECTURE.md) for the exact model, cache
contract, accelerated dependency, validation commands and pending GPU gates.
The notes below about the earlier integration describe the smoke pipeline;
D1 changes are specified in that document. Current checkpoints use format 4,
and validation logs use `val_masked_ce` rather than `val_loss`.

## Usage

```bash
# Tiny synthetic fixture used by the tests (no network, a few seconds)
python -m pytest tests/test_train_diffusion.py -q

# Real bundle, small CPU run; writes checkpoints and config.json under --out
python scripts/train_diffusion.py \
  --bundle /home/kriday/Code/AI/fast-af-boi/artifacts/training-v1/final \
  --out /tmp/train-run --steps 100 --warmup 10 \
  --batch-size 8 --checkpoint-every 50 --eval-every 50

# GPU usage example; current implementation has no verified GPU run
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run \
  --device cuda --steps 100 --warmup 10 --batch-size 8 \
  --checkpoint-every 50 --eval-every 50

# CUDA fp16 autocast with fp32 master weights
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run \
  --device cuda --amp --grad-accum 8 --batch-size 8

# P1 guard: exits 2; GPU validation and measured P0 are still pending
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run --p1

# Resume from an exact checkpoint; the config hash must match
python scripts/train_diffusion.py --bundle ... --out /tmp/train-run \
  --device cuda --resume /tmp/train-run/checkpoint-000050.pt
```

The bundle is never modified. Shards are opened read-only with `np.memmap`.

## What it does

- **Manifest validation.** Checks the `training-mixture/v1` schema, canonical
  manifest fingerprint, `<u2` dtype, shard existence and exact byte size,
  positive whole-sequence dimensions, sequence and split totals, fixed special
  IDs, and optionally SHA-256 shard hashes (`--verify-hashes`). The tokenizer
  manifest is fingerprint-checked and supplies the vocabulary size.
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
- **Checkpoints.** Atomic checkpoint writes (sibling `.tmp`, fsync, rename)
  contain model, optimizer, scheduler, AMP scaler, loader cursor,
  attempted step, tokens consumed, config hash, corpus/tokenizer provenance (manifest,
  tokenizer, and shard hashes), library versions, git commit, GPU, and
  Python/NumPy/PyTorch/CUDA RNG states. A SHA-256 sidecar verifies the file.
  `checkpoint-last.txt` names the newest
  checkpoint. Resuming rejects a different config hash, corpus provenance, or
  cursor/token/step inconsistency, including partial batches and completed epochs.
  Format 4 rejects earlier checkpoints because the architecture, counters
  and metric contract changed. A failed integrity or progress check always rejects resume.

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
  covers the same machine, library versions, and config. GPU bitwise
  reproducibility is not guaranteed, and this integration ran no GPU checks.
- Validation is a fixed batch count (default 8) from the validation split with
  a separate generator, or the whole split with `--eval-full`; the loss is
  token-weighted.

## Recipe alignment (draft `TRAINING_RECIPE_T4_PILOT.md`, 2026-11-13)

Implemented against the draft:

- Classical masking is now per-token Bernoulli with one rate per sequence drawn
  from `U(0.10, 0.90)`; EOS and reserved ids are excluded. The old
  exact-count scheduler is gone; the first eligible token is still forced when
  Bernoulli draws select none, a documented short-row bias.
  `mask_mode=classical` is recorded in the run config. D1 uses the separate blockwise objective described in
  `D1_ARCHITECTURE.md`.
- Gradient accumulation (`--grad-accum`) and CUDA AMP (`--amp`) are wired into
  the training loop. AMP uses fp16 autocast with fp32 master weights and a
  `GradScaler`; its state is checkpointed.
- Optimizer groups follow the recipe: AdamW `betas=(0.9, 0.95)`, `eps=1e-8`,
  weight decay only on matrices.
- Validation can consume the whole split (`--eval-full`) and is token-weighted.
- Checkpoints bind manifest, tokenizer manifest, tokenizer model, and shard
  hashes; they carry library versions, git commit, GPU, step/tokens, and RNG
  states. Saves are fsynced and have SHA-256 sidecars; resume validates
  the corpus hashes, loader cursor, and exact token/step consistency. Library
  versions and git commit are recorded, not compatibility-checked on resume.
- `--p1` is a hard guard. It refuses to run without `profile=d1`, `device=cuda`,
  `amp=true`, `eval_full=true`, `verify_hashes=true`, and a 64-sequence global
  batch. `profile=d1` now constructs the real hybrid. P1 stays blocked pending
  actual GPU checks and a measured P0 probe.

Remaining gaps, explicitly not claimed as D1-ready:

- The default model remains the ~2.2M-param smoke transformer. D1 is an
  implemented, separate 109M profile; its GPU validation remains pending.
- No GPU run was verified in this integration. CUDA AMP, accumulation, and
  masking still need an actual GPU run. No P1 throughput, loss, or convergence
  claim is made.
- D1 now implements blockwise masking and a development decoder. Denoiser-round
  training unrolling, production checkpoint retention, full-corpus training,
  gPPL/MBPP and frozen decode-policy integration are still absent.
- The draft recipe may still change; this code follows the 2026-11-13 draft.

## Validation

Current integration evidence is in `TRAINING_INTEGRATION_REVIEW.md`. Checks use
`/usr/bin/python -m pytest`, including real-bundle hash verification and a CPU
forward. Regression coverage includes exact resume with partial shard batches,
accumulation across epochs, invalid dimensions and metadata, corrupted cursors,
non-finite loss and gradient rejection, LR endpoints, and overflow recovery using
the real CPU GradScaler. CPU GradScaler coverage is not CUDA AMP validation.

A separate CPU process check on the real bundle trained six updates of four
sequences each, with accumulation two. A second process was SIGKILLed after
step four and restarted from checkpoint three. Final loss records, model,
optimizer, scheduler, scaler, loader, and torch RNG matched the uninterrupted
run exactly at step six and 24,576 consumed tokens. Only one validation batch
was scored per evaluation in this check.

The source branch reported a pre-alignment Colab T4 smoke, but the branch has
no raw training logs or checkpoints to verify it. That report is historical,
unverified here, and is not evidence for the current implementation or D1.

The source branch attributed its earlier full-suite failure to an AppImage
Python launcher; that original process is unavailable to rerun here.
Here the bare `pytest` launcher points to a deleted Synara AppImage and exits
127. `/usr/bin/python -m pytest` runs the standard-library CLI regression
successfully, including its `sys.executable -S` subprocess. The global launcher
was left unchanged; no test was skipped or weakened to accommodate it.

## Update and overflow accounting

`step` counts attempted global batches and `tokens_seen` counts consumed tokens,
including batches skipped by AMP after overflow. `optimizer_steps` in each log
record counts successful updates. Overflow backs off the scaler and leaves the
model, optimizer, and LR schedule unchanged; the record contains
`optimizer_step_skipped=true` and `grad_norm=null`. FP32 non-finite gradients and
non-finite losses fail before an optimizer update. This follows the
[AMP accumulation and clipping order](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html).

The logged LR is the value used for that update. Without skipped updates, runs
of at least two steps reach 10% of peak on the final update. A one-step run uses
peak LR. Shard tails may produce partial global batches: `batch_size * grad_accum` is a maximum, so budget using actual
token counts. The production bundle supports full micro-batches of 4 or 8;
its 5,480-row tail produces a partial batch at micro-batch 16.

## Limitations

- D1 is implemented as a separate 109M profile. `--p1` remains blocked until
  GPU correctness and P0 measurements are available.
- AMP and gradient accumulation are implemented; CUDA execution is unvalidated
  in this integration because this machine has CPU-only PyTorch.
- No distributed training, fused optimizer, or production checkpoint retention.
- GPU bitwise reproducibility is not guaranteed; CPU resume is exact in tests.
- The smoke profile trains classical masked-token prediction. D1 has blockwise
  training and a fixed-length development decoder, with no benchmark-policy
  integration, looping or quality/convergence claim.
- Validation is a sample by default; `--eval-full` covers one full pass but is
  not run at 100M-token scale here.
- Resume determinism is verified on CPU exactly. CUDA and multi-GPU resume
  were not tested here. Checkpoint versions before 4 are rejected.
- The loader holds one memmap per shard at a time and validates target ranges on
  every batch, which is intentionally simple rather than throughput-optimal.
- Training throughput and convergence at 100M-token scale were not measured.
