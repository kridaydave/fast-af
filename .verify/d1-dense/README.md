# Dense D1 verification

The implementation trains every noisy block from a shared differentiable clean
prefix. See [the design and execution rules](../../docs/D1_SUPERVISION.md).

This record separates local checks, rejected GPU candidates and delivered GPU
measurements. Dense two-T4 throughput and process resume remain unmeasured.
No long pretraining run has been launched. The existing Kaggle two-T4 request
remains a qualification request for the earlier frontier implementation.

## Delivered single-T4 result

Actual Tesla T4 / SM75, Torch 2.14.1+cu130, CUDA 13.0. The model has
109,137,632 parameters and context 1024. Both probes use global batch 64,
FP16 autocast, `chunk-tc`, fused AdamW and the same verified real-token bundle.
Dense timings include deterministic reductions. Each final configuration has
one untimed warmup and three timed successful updates. No timed updates skipped.
These are short throughput screens, not a production P0 or a quality study.

| Supervision | Micro | Consumed tokens/s | Scored targets/s | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Frontier baseline | 8 | 15,219.49 | 245.55 | 11.79 |
| All blocks, selected | 2 | 2,970.78 | 1,473.20 | 7.58 |
| All blocks, memory screen | 4 | 3,186.81 | 1,585.82 | 13.15 |

The selected configuration delivers **5.9997× scored-target throughput**.
Micro 4 reaches 6.4584× but exceeds the existing 85% device-memory safety
threshold on the 14.56 GiB device. Its small rate gain does not justify that
margin for a prototype. Select single-GPU micro 2, global batch 64 via
accumulation 32. A proposed two-GPU screen uses global micro 4, two per GPU,
and accumulation 16; it requires actual two-T4 qualification. Scored counts per consumed token are
about 1.6% for the frontier baseline and 49.6% for dense training. Prefix
computation remains useful; this ratio is not a percentage of wasted FLOPs.

[summary.json](summary.json) derives rates by total scored tokens divided by
total timed seconds, not by averaging rates. It records observed window ranges,
raw source paths and limitations. [hardware.json](colab/hardware.json) records
actual hardware. Seven measured scripts matched delivered sources byte-for-byte.
[source-alignment.json](source-alignment.json) indexes their SHA-256 and Git blob
IDs, so the implementation commit preserves the exact tested sources without
repeating them here. Rejected variants have source patches and blob indexes.

## Correctness and resume

- Final native and Tensor Core parity reports are
  [chunk-parity-final.json](colab/chunk-parity-final.json) and
  [chunk-tc-parity-final.json](colab/chunk-tc-parity-final.json).
  Independent CPU-reference outputs and all **65 parameter tensors** in a small
  hybrid are compared under GPU mixed precision. Worst parameter-gradient
  relative L2 errors are 0.0270005 and 0.0273093, under the fixed 0.05 bound.
- Full 109M-model context-1024 dense/cached logits have relative L2 errors
  0.00167366 and 0.00198060, under the unchanged 0.003 cache bound. This is
  full-model forward/cache parity, not full-model CPU gradient parity.
- [dense-resume-final.json](colab/dense-resume-final.json) covers actual SIGKILL
  after update 4, restart from checkpoint 3 and continuation to update 6 on the
  full 109M model, context 1024, batch 1, AMP scale 8 and fused AdamW. All six
  updates applied. Model weights, optimizer tensors, losses, loader, scheduler,
  scaler and RNG matched exactly in this experiment. This does not promise
  bitwise equivalence across other hardware, libraries or configurations.
- The final [full local suite](local-final-full-suite.log) passed **299 tests**,
  with **2 GPU-only skips**. The dense tests include multi-chunk entry-state
  and input-gradient parity with strong decay, independent-frontier parameter
  gradients, leakage controls, partial batches and accumulation/resume. Basic
  basedpyright found zero errors/warnings in six changed implementation/tools
  files, Ruff F and compilation passed, and Markdown link/diff checks passed.

## Failures retained and resolved

The [first nondeterministic candidate](colab/rejected-nondeterministic) passed
output/gradient parity, but failed the existing optimizer resume tolerance.
For example, optimizer tensor 6's first moment had relative L2 error 0.00481173
and absolute L2 error above its 0.00001 allowance. The delivered dense CUDA
trainer requires deterministic PyTorch reductions and a supported cuBLAS
workspace. The final resume check passed without widening tolerances.

The [32-token clean-scan screen](colab/scan32-screen) passed deterministic
resume but the micro-1 global-64 timing phase hit its external 210-second cap.
No qualified rate is claimed for that attempt. The delivered clean scan uses
128-token chunks and recovers intermediate block-entry states from their solved
updates. Final timing probes used micro 2 and 4 and passed. Colab CLI also had
an output-silence timeout and a connection-loss retry; these are control-plane
errors, separate from the numerical failure. No long pretraining was started.

## Budget implications

[budget-scenarios.json](budget-scenarios.json) keeps the actual 30-hour renewable
Kaggle weekly quota and separates measured single-T4 rates from unmeasured
parallel scaling and overhead. With an illustrative 1.5× effective two-T4 gain
and 30% overhead relative to training time, 30 runtime hours projects to roughly
**370M consumed tokens / 184M scored targets**. This is a planning scenario,
not a two-T4 benchmark or a guarantee of adequate learning.

Dense training does more work per context. The same assumptions put **2B actual
consumed dense tokens near 162 runtime hours**, not the old 31.5-hour frontier
estimate. The gain is more supervised targets per compute budget, not retaining
a misleading corpus-count speed. Hold the compute budget fixed, then compare
held-out learning against frontier training and a conventional autoregressive
baseline before selecting the base. No quality-equivalence conversion between
these objectives is claimed. Dense validation/checkpoint/export overhead remains
unmeasured, and the 30% allowance must be revisited.
