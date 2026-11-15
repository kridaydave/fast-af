# T4 optimization evidence

Budget policy update: the historical six-hour cumulative allocation in this
record is superseded by [the project plan](../../docs/PROJECT_PLAN.md) and
[weekly Kaggle accounting](../d1-t4-parallel/weekly-budget.json). Measurement
values and event durations below are unchanged.

This run optimizes the existing 109,137,632-parameter D1. Model width, layer
layout, context 1024, token masking and global batch 64 are unchanged. No P1 or
long pretraining run was launched. The target is at least 6,000 processed
tokens/s and a P0/P1 plan within six GPU-hours.

## Changes

- Project only scored positions to the 32k vocabulary during D1 training and
  validation. All prefix tokens still pass through the full hybrid, and their
  gradients still flow. Dense decoding logits remain available as before.
- Batch state-independent query/key products and decay work across GDN chunks.
- Unbind chunk views once. Repeated indexing previously made backward zero and
  add full-sized slice-gradient buffers for every chunk.
- Use chunks up to 128 tokens, with smaller power-of-two chunks for short
  inputs. The recurrence and FP32 solve/state stay unchanged.
- Record backend version 2. Version-1 training checkpoints reject resume rather
  than silently changing numerical execution. Model state-dict shapes remain
  unchanged.

Profiling showed efficient CUDA attention was already selected on T4. Explicit
KV-head expansion did not help and was not adopted. No custom CUDA extension,
Triton, TF32, lower-precision recurrent state, detached prefix gradients, or
architecture shrink was needed.

## Exploration and baseline

Baseline commit: `0113590c91ca9457c140c4f3cf2b4a0c1995ff17`. The baseline source
hash in `screen.json` is identical to that commit's D1 model.

`profile.json` records the original CUDA profile. Its backward had 5,328
`aten::add_` calls and 5,760 `aten::bmm` calls in one measured update, with a large
share of time in gradient buffer accumulation and copies. Operator and kernel
rows overlap in the profiler; do not sum their times together.

`screen.json` compares the same four-sequence, full-1024-context workload,
initialization, five-update sequence, optimizer, AMP scale and masking. The
baseline at the start took median 1.125 s/update and the end control 1.144 s.
The 128-token chunk candidate with unbound views and scored-only projection took
0.539 s. `vector-screen.json` adds batched state-independent work; its 128-token
candidate took 0.529 s, repeated at 0.531 s. The 256-token candidate was slower,
0.558 s. These are configuration screens, not production throughput windows.
There was some timing noise, especially immediately after profiler collection.
All original samples are preserved.

The final implementation also avoids excess padding on short prefixes. Timing
screens used fixed length 1024, so they do not measure that additional benefit.
`jobs/` preserves the exact exploratory job source and launcher limits.

## Real T4 checks

Tesla T4, SM75, PyTorch 2.14.1+cu130, CUDA 13.0. TF32 disabled. Every PASS report
records the SHA-256 of the executed model, trainer and checker. Initial results
precede the final evaluation-only delegation. Their trainer and P0 helper are
preserved as `measured-*.py.txt`; source-alignment reports verify that the
training loop, all other trainer code and P0 instrumentation are unchanged.
Fresh `final-checks/` reports record the delivered trainer source. The model and
checker hashes are unchanged throughout. The earlier checker refinements have
now actually been executed on T4.

`phases.json` records exit 0 for:

- `parity.json`: nine lengths, 1 through 257, in FP32 and mixed precision, with
  strong decay, nonzero initial state, all input gradients and cached-state
  checks. Worst core output relative L2 error was 1.34e-6. Full-width layer input
  and all 13 parameter gradients passed. Full-model forward/cache used 96 tokens;
  CPU/GPU relative logits error was 2.08e-6 FP32, 2.11e-3 fp16. The full-model
  masked 32-token backward used the optimized scored-only head against the CPU
  dense reference. All 254 parameter gradients passed; worst relative L2 error
  was 1.06e-5 FP32, 6.73e-3 fp16.
- `resume.json`: real full-model, context-1024 AMP training. Six updates applied,
  kill after step 4, restore checkpoint 3 and finish. Resumed losses matched
  exactly; loader, scheduler, scaler and Torch/CUDA RNG matched. Maximum weight
  error was 2.87e-4 and optimizer error 3.25e-6. Resume is tolerance-based across
  processes, not bitwise.
- `probe.json`: 20 accumulated global-batch updates, 141.8 seconds of work.
  Micro-batch 8, accumulation 8 now fits the same 85% allocation margin.

## Bounded probe

| Objective | Median consumed corpus tokens/s | Median processed tokens/s | Median scored tokens/s |
|---|---:|---:|---:|
| Last frontier, full context | 8,464 | 8,464 | 131.5 |
| Uniform frontier | 14,066 | 7,940 | 233.2 |

Every two-update timing window applied both updates without a skip. Full-context
processed rates ranged from 8,408 to 8,529; uniform-frontier rates from 7,328 to
8,067. The old measured configuration was 3,490 and 3,173 respectively. This is
about 2.43x full-context throughput, comparing the recommended batch configurations,
4/16 before and 8/8 now. It is not an isolated same-batch kernel speedup claim.

Peak allocated memory in full-context windows was 13,092,106,752 bytes,
12.19 GiB. Reserved was 13,537,116,160 bytes, 12.61 GiB. The uniform-frontier
allocator reservation peaked at 13,788,774,400 bytes. All are measured on a
runtime exposing 15,637,086,208 bytes to PyTorch. No batch-16 claim is made.

This probe used a repeated small real-token subset and short windows. It excludes
validation and checkpoint I/O. Production P0 and overhead measurements are
recorded separately below; they determine the budget recommendation.

## Production P0

`p0.json` and `p0.log` record the actual complete training-v1 bundle at fingerprint
`6f62a175ac05648cdef20684fba9038105a906333a0bcea57f81c3f48a172ab4`, with all ten
train shard hashes and the validation/tokenizer hashes checked. Configuration
was seed 1001, micro-batch 8, accumulation 8, context 1024, peak LR 1e-3, four
scheduler warmup updates, AMP initial scale 256. The separate performance warmup
was the first 50 attempts.

| Attempts | Seconds | Consumed tokens/s | Processed tokens/s | Scored tokens/s | Applied |
|---|---:|---:|---:|---:|---:|
| 51 to 100 | 217.071 | 15,095.6 | 7,827.3 | 236.5 | 50 / 50 |
| 101 to 150 | 223.250 | 14,677.7 | 7,861.8 | 231.6 | 50 / 50 |
| 151 to 200 | 217.493 | 15,066.3 | 7,762.7 | 235.0 | 50 / 50 |

All three windows exceed the 6,000 processed-token/s stop gate. The median is
7,827.3. This does not reach the 13,000 green band. Actual P0 work was bounded
at 200 attempted updates / 30 minutes, with a 1,850-second outer process timeout.
It finished in 896.6 seconds of measured job work, 900.5 including process startup.
The final checkpoint write, checksum and summary took 14.61 seconds.

Five AMP skips occurred only in warmup, attempts 19, 21, 24, 38 and 39. All 150
timed updates applied. P0 applied 195 of 200 total attempts and consumed
13,107,200 corpus tokens, processed 6,881,024 and scored 205,801. Loss stayed
finite; skipped gradients did not change parameters or advance the LR scheduler.
Warmup results are not hidden in the throughput denominator.

P0 peak allocated memory was 13,133,616,640 bytes, 12.23 GiB. Reservation reached
15,367,929,856 bytes, 14.31 GiB, including unused allocator segments. The runtime
was close to capacity in reservation while live allocation stayed below the
85% selection criterion. This is one T4 runtime, not an OOM guarantee on every
Colab environment.

## Evaluation bottleneck and replacement

The original full-prefix validation exceeded its 630-second outer cap.
`p0-phases.json` records exit 124 for that phase. It produced no completed whole-
split result, so its exact latency and final loss are not claimed. The first
lower-scale job was blocked by that failed phase; `low-scale.json` is preserved
as a blocked check, not a numerical failure or a pass.

`scripts/evaluate_d1_checkpoint.py` scores the same masks while reusing the clean
prefix cache. It discards noisy candidate states and commits only original clean
blocks, including all-EOS blocks that contribute no loss. Training gradients are
never detached. Its CPU regression compares against full recomputation, covers
partial batches and EOS-only frontiers, and verifies model mode and RNG restoration.

On T4, `cache-parity.json` compares 16 complete 1024-token contexts, all 32
frontiers, using the trained P0 checkpoint. Both paths scored exactly 8,078
positions. Full recomputation CE was 7.6291770095 and cached CE 7.6291828023,
absolute difference 5.79e-6 against a predefined 0.003 tolerance.

`cache-full.json` then completed all 960 validation contexts and 32 frontiers:

- 491,072 scored positions, with the same per-token weighting and mask RNG policy.
- 3,840 score passes and 3,720 clean commit passes.
- 1,935,360 processed tokens including commit work, versus the 16,220,160 that
  full recomputation would process on these contexts.
- 353.09 seconds evaluation work; 365.42 seconds including checkpoint/config
  loading and process overhead.
- Peak allocation 926,572,544 bytes. Validation CE 7.5096933 is a P0 diagnostic,
  not a P1 quality gate, gPPL or MBPP result.

Three such full evaluations project to 18.27 minutes including checkpoint loads.
The replacement is faster than the timeout lower bound, but an exact whole-split
speedup over the old path cannot be calculated from an incomplete baseline.

## Calibrated scale, final integration and source alignment

`low-scale-final.json` reads the actual final P0 scaler state: scale 8, growth
tracker 195. Fresh full-width layer and full-model output/gradient checks passed
against CPU with that starting scale, as did a separate six-update process kill
and resume. Use `--amp-init-scale 8` for the recommended T4 configuration. This
is measured calibration, not a guarantee that future data cannot overflow.

After integrating cached full validation into `train_diffusion.evaluate`,
`final-checks/phases.json` records exit 0 for fresh parity, resume, accumulated
probe and trainer integration. The integration check used 32 real validation
contexts, checked all frontiers, matched the cached evaluator loss exactly,
preserved CPU and CUDA RNG, and restored the model's training mode.

The long production P0 was measured before this last evaluation-only delegation.
`p0-source-alignment.json` compares the entire trainer module AST excluding only
`evaluate`: it is identical, including training, optimizer, masking, loading and
configuration. P0 had `eval_every=0`, so that function was never called. The exact
measured trainer is retained as `measured-train_diffusion.py.txt`. This is why the
long timing remains applicable without claiming the whole trainer file had the
final hash during that run. The final GPU checks do have that final hash.

The P0 overhead reporter also changed to consume the evaluator's returned counts;
a forward hook would miss cached `forward_block` calls. Its P0/config/main
function ASTs are unchanged, as recorded in `p0-helper-source-alignment.json`.
The original P0 helper source is retained. That last reporting adapter has local
type/lint checks; it has not itself been rerun on GPU. A late extra integration-
job update arrived after the process loaded its source and did not execute.
`job-source-provenance.json` distinguishes the executed trainer/RNG check from
that late update. The underlying evaluator and its counts did execute on all
960 contexts. No pass is attributed to the unexecuted addition.

## Six-hour allocation ledger

`six-hour-budget.json` counts all three Colab allocations, including the two
previous backend sessions and this entire optimization session. Allocation time,
including idle setup, is counted rather than only CUDA kernel time.

- Previous sessions: 916 + 1,295 seconds.
- This session: 4,231 seconds, 05:29:10 to 06:39:41 UTC on 2026-11-15.
- Already allocated: 6,442 seconds, 1.7894 GPU-hours, including P0 and all checks.
- Maximum future stage allocations: 4.15 GPU-hours.
- Maximum combined ledger: 5.9394 GPU-hours, leaving 218 seconds under six hours.

The future allocations are 2.50 hours for P1 training including slowdown/restart
margin, 0.25 for optional LR screens, 0.40 for three full validations, 0.75 for
external quality/decode checks and 0.25 for checkpoint I/O, transfer and setup.
Training alone projects to 1.84 to 1.89 hours from the three production windows.
Three full validations project to 0.3045 hours including checkpoint loading.
External generation/MBPP/decode and upload durations remain unmeasured for D1;
their caps must cause an incomplete-gate result if exhausted, not extra spending.
Training margin is already inside the 2.50-hour allocation. Do not add P0 again,
or another blanket contingency, and silently exceed six hours.

This is a feasible capped P0/P1 plan, not an end-to-end measured quality run.
`--p1` remains blocked until the external controller can enforce the global and
per-stage balance, the decode evaluation policy is frozen, and checkpoint
retention/upload is ready. P2 matched twins and a long pretrain are outside the
budget. No P1 or long pretraining run was started.

The VM was terminated after reports were recovered. Temporary SSH credentials
used to copy the read-only corpus were deleted. Full checkpoints were not copied
locally; their checked hashes, sizes and numerical resume evidence remain in the
reports. Protected project-root and interrupted-checkout HEAD/status/diff hashes
remain unchanged.

## Local verification and reproduction

All pytest files were covered across split invocations: 280 passed, two locally
skipped GPU cases. Basic basedpyright found zero errors or warnings in the model,
validator and three GPU/evaluation job scripts. Ruff F checks, compilation and
Git whitespace checks passed. The real native Python CLI was also tested with a
tiny CPU D1 configuration and full cached validation, covering its import path;
that small subprocess is not presented as a full-model or GPU measurement.
The frontier-coverage test still asserts the same visits and RNG preservation,
with its instrumentation moved to the extracted evaluation helper.

The bound tool sequence is recorded in `jobs/` and the raw logs. A reproducible
runtime needs `src/scripts/`, the verified original corpus at `full-bundle/`, and
the small real-token probe bundle at `src/probe-bundle/`. Run parity, resume and
probe with `scripts/check_gdn_t4.py` before `scripts/check_d1_p0.py --phase p0`.
The current `--phase overhead` uses cached validation. For a saved checkpoint:

```sh
python scripts/evaluate_d1_checkpoint.py \
  --checkpoint /path/to/run/checkpoint-000200.pt \
  --bundle /path/to/training-v1/final --device cuda --amp \
  --out /tmp/cached-validation.json
```

`--reference --max-batches 2` selects full recomputation on a bounded subset for
parity. It does not reduce the default full validation. Both evaluator modes
verify the checkpoint checksum, sibling config hash and corpus provenance. The
bare local `pytest` launcher remains broken; use `python -m pytest` as recorded
in the prior integration evidence. No global launcher or other checkout was edited.
