# T4 optimization and two-rank D1 evidence

2026-11-15. Base commit: `c4bf81e9328e8b09ed278046a430e6dba3d72674`.
`origin/master` remains `cde26573920c7ae5bd6aab06617952ba282bbde9`.
The pipeline/recipe integration and earlier native-chunk measurements are
already in that history; this change adds optional Tensor Core GDN, fused
AdamW and single-node two-T4 training. No pretraining candidate was launched.

The model is **109,137,632 parameters**, context 1024, global batch 64.
All timing uses a verified real-token byte-prefix of training-v1: 128 training
contexts and 32 validation contexts. Its fingerprint is
`ec6f6545c91ae536ed75e631f5d53f699703279477f913a753c53df51a1d8508`.
The original approximately 100M-token corpus was only read and hashed.
This Python-heavy fixture is not the final general-base data recipe.

## Delivered precision and backend

`chunk-tc` version **3** uses FP16 GEMM operands and FP32 GEMM outputs and
first-order gradients. State, solve, residuals, normalization, decay, master
weights and optimizer state stay FP32. CUDA FP16 autocast and initial loss
scale 8 were used. TF32 was disabled. No BF16, FLA/Triton, or unsupported SM80
kernel was used on T4/SM75. The ordinary `chunk` backend remains selectable.

`source-alignment.json` maps delivered code to `colab-final/measured-scripts`.
The model and numerical/checkpoint code match. The sole subsequent trainer
change makes final CLI summary printing rank-zero-only. It changes neither
training arithmetic nor checkpoint contents. The evidence snapshots and hashes,
not the pre-validation Git head, identify the measured implementation. Unchanged
ancillary scripts are indexed by SHA256 and recoverable from the base commit;
the numerical modules are retained verbatim.

## Single-T4 final validation

`colab-final` is the delivered v3 backend on Tesla T4 / SM75, Torch
2.14.1+cu130, CUDA 13.0 and Python 3.13.16. Reports have explicit source hashes.

- `parity.json`: CPU recurrence versus GPU output, final state and all six
  input gradients at lengths 1, 31, 32, 33, 65, 127, 128, 129 and 257, including
  strong decay and nonzero initial state. Tests cover FP32 inputs and FP16
  value inputs; the Tensor Core backend still internally quantizes GEMM
  operands in both cases. Worst core output relative L2 error: 0.0004134.
- The layer check covers a cached 32-token prefix plus a 33-token continuation,
  convolution/recurrent states, immutable entry cache, input gradients and all
  parameter gradients. Worst layer parameter-gradient relative L2: 0.001880.
- Full 109M model logits and cached logits are compared with CPU. All 254
  named parameter gradients are compared on a masked 32-token block, using the
  CPU dense vocabulary head as the reference. Worst parameter relative L2:
  0.007316. This is not CPU gradient validation at full context 1024.
- `long-cache.json`: independent CPU recurrence versus full GPU and 32 cached
  blocks over **all 1024 tokens**, including all final layer states. CPU/full
  logits relative L2: 0.002053; full/cached logits: 0.001993 against the existing
  0.003 limit. This run took 14.55 seconds.
- `resume.json`: actual subprocess SIGKILL after update 4, restore checksum-
  verified checkpoint 3 and complete update 6. All six optimizer updates
  applied; loader, scheduler, scaler, Torch CPU/CUDA RNG and counters match.
  In this run weights matched with torch.equal; loss and optimizer maximum absolute errors were zero.
  This does not promise long-run or cross-stack bitwise determinism.

`uniform.json` measures four untimed updates followed by three four-update
windows, with global micro-batch 8 / accumulation 8. All 12 timed updates
applied. Aggregate rates, computed from summed counts divided by summed wall
seconds, are **15,308 consumed corpus tokens/s**, **8,436 processed input
tokens/s**, and **241 scored tokens/s**. Peak Torch allocation was
12,842,742,272 bytes, approximately 11.96 GiB. Timing excludes checkpoint and
validation work. The short frontier sample is not a production throughput
confidence interval or a long-run AMP test.

## Investigated failures and discarded candidates

- The new profile in `colab/profile.json` profiles the current native version-2
  training path, not the old pre-optimization baseline. In a full-context
  micro-batch-4 update it measured approximately 0.526 seconds. Copies,
  GEMMs, elementwise operations and triangular solves still dominate. ATen
  and kernel profiler rows overlap; they must not be added as a wall-time sum.
- `colab/experiment.json` contains an early Tensor Core screen. Native
  `bmm(out_dtype=float32)` backward raised "derivative ... not implemented";
  the delivered explicit backward addresses that. Reconstructing fused AdamW
  from existing parameter groups initially inherited `fused=None` and failed;
  the delivered optimizer builds clean groups with `fused=True`.
- CUDA Graphs initially failed capture because boolean selection has dynamic
  output size. A fixed active-block head allowed capture, but produced little
  additional speed. Graph replay allocation statistics are not comparable to
  eager peaks because allocations occur during capture. Graphs were not shipped.
- `colab/v2/batch-screen.json` batches more output products. Its short control,
  candidate and repeat timings did not show a useful consistent gain. It was
  not adopted and its execution status is not a full parity qualification.
- `kaggle-v1`: two real T4s were allocated, Torch 2.11.0+cu130. Native/Tensor
  Core CPU parity and two-rank gradients passed. Native micro-batch 8 then
  OOMed: 13.38 GiB allocated, with a 302 MiB request and 274.81 MiB free.
  Torch warned that mixed-dtype RMSNorm could not dispatch its fused path.
- `kaggle-v2-rejected`: an FP16 residual/norm experiment passed core and layer
  checks but failed cached/full logits. Relative L2 was 0.003039 versus 0.003
  allowed. The tolerance was retained; FP32 residuals and normalization were
  restored in version 3. Those v2 measurements do not qualify delivered code.
- `colab/final/phases.json` records an intermediate repeated resume check
  exceeding its outer 180-second cap. Other v1/v2 resume runs passed; the
  final v3 check was rerun with a 300-second cap and completed in 98.10 seconds.
  The Colab kernel also had a control-plane timeout; SSH through Colab CLI
  retrieved reports. No timed-out phase is treated as a pass.

## Local checks and scope

`local-checks` records **286 passed, 2 skipped**, split into the core suite and
separate HF cache/wiring suites to fit host memory. All tests were run through
`python -m pytest`; the bare `pytest` launcher still references a removed
interpreter and was not used. The two GPU tests skipped locally do not stand
in for the real T4 reports above. Model and new-tool type checks have zero
errors/warnings; Ruff F checks and compilation passed.

The CPU tests independently compare globally token-weighted gradients with
averaged rank gradients, including a zero-scored rank. They also check suffix
head gradients, guards, default checkpoint-hash compatibility and package file
selection. A passing CPU partition test alone is not a two-GPU validation.

`preserved-checkouts.json` records protected heads, statuses and diff hashes.
The dirty project-root master, interrupted checkout and reference-plan checkout
were not edited. There were no subagents or other T3 threads.

## Limits

The timing subset is revisited and cached locally. Training quality, the broad
base data recipe, full-corpus validation latency for v3, generation quality,
GPU bitwise determinism in longer runs, GPU checkpoint upload/retention and
multi-node/elastic training are unvalidated. New configurations still require
P0 and quality requalification. The original 353-second full validation is a
historical native-backend measurement, not a v3 timing claim.

Raw `gpu_seconds` counts participating ranks. Allocation accounting charges
both T4s for the entire Kaggle notebook, including one-rank controls, CPU
reference work, checkpoint work and failed tests. Provider quota has separate
units. Six physical GPU-hours are three runtime hours with both T4s, before
subtracting previous work. `--p1` remains blocked pending the budget controller,
checkpoint retention/upload and frozen evaluation protocol.

Two-T4 final results and the revised allocation ledger are added when that
bounded notebook terminates; `final-status-history.jsonl` records its status.
