# D1 native chunk backend on a real T4

Recorded from Colab CLI jobs on 2026-11-14, recovered and reviewed 2026-11-15.
The Colab session `d1-t4-gdn-check` has terminated. `colab sessions` on recovery
reported no active server sessions. No long pretraining run was launched.

## Decision and implementation

Use `--gdn-backend chunk` for the next T4 infrastructure run. It implements the
same ordinary forward gated delta rule as the CPU reference, with normalized
queries and keys, positive decay, sigmoid beta, and FP32 recurrent state:

```
S_decay = exp(g_t) * S_previous
u_t = beta_t * (v_t - k_t^T * S_decay)
S_t = S_decay + k_t outer u_t
o_t = q_t^T * S_t / sqrt(key_dim)
```

Within a 32-token chunk, express all residual updates as a unit-lower-triangular
linear system. Solve it with native PyTorch, then propagate state between chunks.
Neutral padding handles partial chunks. Autograd differentiates the solve and
initial state. The implementation preserves the caller's cached state.

| Candidate | Assessment against the recurrence and T4 |
|---|---|
| Tokenwise PyTorch reference | Implements the required recurrence and gradients; retained as the correctness oracle. No comparative timing claim. |
| Pinned modern FLA/Triton | Implements the delta recurrence, but its supported NVIDIA target is SM80+. T4 is SM75. Keep the explicit hardware rejection. |
| Native PyTorch chunk solve | Implements this exact recurrence without Triton, uses supported T4 CUDA operations, and now has measured output, gradient, cache, resume and full-model timing evidence. Selected. |
| Older Triton/FLA pin or custom CUDA port | Possible follow-up work. Neither is validated here; both need the complete backward, strong-decay, cache and AMP checks. Do not assume an old pin is a drop-in solution. |

Hardware support sources: [Triton README](https://github.com/triton-lang/triton/blob/main/README.md),
[NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus).
The chunk algebra was assessed against the [official FLA naive delta rule](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/fla/ops/gated_delta_rule/naive.py)
and the repository's own reference. External implementations are not imported
into this backend. Normalization follows D1's reference exactly.

Backend implementation commit: `bf8b7e15be9c6178e579b3f840772278a1e50bf2`.
The trainer's configurable AMP scale and this evidence are committed separately.
The original integration includes pipeline `dfae0b68f7f3c2d560548a991fd5151d1138ec00`,
recipe `e61cf251f83bb6d34600f17dc31281df303e7111`, and master
`cde26573920c7ae5bd6aab06617952ba282bbde9`.

## Hardware and source identity

All three reports identify Tesla T4, capability 7.5, 15,637,086,208 usable bytes,
Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0. TF32 was disabled. The model was
the actual default 109,137,632-parameter D1, with 12 GDN and four GQA layers.

The reports contain source SHA-256 values:

| File | SHA-256 of executed source |
|---|---|
| `scripts/d1_model.py` | `02916662d472d3c54be195f20b3d13e40468ea75fefd9aceaa474596a81ec755` |
| `scripts/train_diffusion.py` | `3ad3ad7cc6afbfcaed3480a6a93c5715019a2a073670e16047aa3387fe48215a` |
| `scripts/check_gdn_t4.py` | `31b4ac6f4689b3340c73524080f13da1131749e88257935f022675c11f7a6319` |

Model and trainer hashes match the delivered source. The checker has subsequent
changes to retry accounting, configurable bounds, rejection of non-finite
reference values and validation of screen selections. Its current hash differs.
The exact executed checker source was not recovered, so the reports establish
the recorded checks, not a GPU rerun of those later checker edits. All numeric
metrics in the PASS reports were independently checked for finiteness locally.
Raw reports were preserved without rewriting them.

## Numerical results

`phases.json` records exit code 0 for parity, resume and probe, taking 17.0,
111.3 and 381.3 seconds respectively including process overhead.

- `parity.json` and `parity.log`: six sequence lengths, 1, 31, 32, 33, 65, 129,
  in FP32 and mixed precision, including strong decay and a nonzero initial
  state. Outputs, final state and all six input gradients passed against CPU.
  Worst relative output L2 error was 6.05e-7.
- Full-width GDN layer with a 32-token cached prefix and a 33-token continuation:
  output, recurrent/conv state, input gradient and all 13 parameter gradients
  passed in FP32 and fp16 autocast. Prefix cache stayed unchanged.
- Full-model forward and committed-cache checks used 96 tokens. CPU/GPU logits
  relative error was 2.09e-6 FP32 and 2.09e-3 fp16. Cached and committed logits
  passed with unchanged entry caches. Full-model backward used a 32-token masked
  example and checked all 254 parameter gradients. CPU CE was 10.5938692;
  GPU FP32 10.5938702, fp16 10.5938034.
- `resume.json` and the three resume process logs: actual full model, context
  1024, real-token subset, AMP, six updates. Kill exit -9 after step 4, reload
  checkpoint 3, continue through step 6. All six updates applied. Resumed losses
  matched exactly; loader, scheduler, scaler and Torch/CUDA RNG states matched.
  Weights were not bitwise identical. Maximum absolute weight error was
  3.5854e-4, optimizer error 3.3817e-6, within the recorded 1e-3 relative L2
  plus 1e-5 absolute L2 limits. This is tolerance-based cross-process resume.

`layer-amp-diagnostic.json` preserves a failure worth retaining: loss scale 8192
produced non-finite input and parameter gradients on the layer fixture. Scale
256 passed. `--amp-init-scale 256` is therefore recommended, with the normal
GradScaler overflow handling still active. It is not proof that later training
will never overflow. The trainer default remains 65536 for compatibility of the
usual default; set 256 explicitly for this D1 T4 configuration. This new config
field participates in the config hash, so checkpoints made before the field was
introduced reject resume rather than silently changing configuration.

`determinism.json` records two repeatable short runs within a process, with and
without deterministic mode. It does not prove bitwise cross-process behavior.
The original saved checkpoints were not copied locally, so independent checkpoint
reinspection now relies on the executed checker report and process logs.

## Bounded full-model probe

`probe.json` and `probe.log` contain the measured results. The probe used real
training-v1 token prefixes, 128 train and 32 validation sequences at length
1024, with a 32k tokenizer. This is a small repeated single-shard subset, not a
quality test or an audit of a new dataset. Local corpus I/O was cached.

The probe ran 20 optimizer updates in 378.3 seconds: four capacity screens,
two warmups per objective and three two-update timing windows per objective.
No timed updates were skipped. Global batch was 64. AMP used fp16 compute with
FP32 parameters, gradients, state, triangular solve and AdamW moments.

| Objective | Median consumed tokens/s | Median processed tokens/s | Median scored tokens/s |
|---|---:|---:|---:|
| Frontier 31, full 1024 context | 3,489.9 | 3,489.9 | 53.9 |
| Uniform sampled frontier | 6,699.4 | 3,173.3 | 104.3 |

Consumed means all loader tokens, including cropped future tokens. Processed
means the clean prefix plus active block passed to D1. Scored means masked loss
tokens only. Timing includes corruption, transfers, accumulated backward,
clipping, scaler and optimizer. It excludes evaluation and checkpoint writing.
Aggregate uniform-frontier rates across the three windows were 6,506 consumed,
3,141 processed and 101.8 scored tokens/s. Prefix sampling explains much of the
short-window spread; this is not a sustained-rate estimate with a confidence bound.

| Micro-batch screen | Peak allocated GiB | Result |
|---|---:|---|
| 1 | 2.50 | Fits |
| 2 | 4.66 | Fits |
| 4 | 7.40 | Selected, accumulation 16 |
| 8 | 12.69 | Fits, but above 85% allocation margin |

Full-context measured windows peaked at 7.40 GiB allocated. Peak reserved was
13.06 GiB because batch-8 screen allocations remained cached. Do not report that
reserved amount as the live allocation for batch 4, or claim batch 8 was OOM.
Screen times are cold capacity checks at different update states, not a fair
micro-batch throughput comparison. No batch-16 measurement was made.

The probe's wall check was 600 seconds at update boundaries and its update cap
24. Those caps are not a hard kernel preemption guarantee. Actual work stayed
within both. This was not the recipe's formal 200-step P0 or a long pretrain.

## Budget and recommendation

The [revised recipe](../../docs/TRAINING_RECIPE_T4_PILOT.md#8-compute-budget)
uses the measured token denominators. A nominal 100M corpus-token P1 projects to
4.1 hours at the sampled-frontier consumed rate, or 8.0 hours at full context,
before evaluation, checkpoint transfer and interruptions. P0 plus P1 and the
existing evaluation/30% training allowances project to about 12 to 21 T4-hours.
P0 plus P1 plus one paired P2 seed projects to about 66 to 125 hours. P2 is only
an unlooped compute-equivalent extrapolation; the looped model is unmeasured.

The native backend is correct over the tested cases and fits T4. Its 3.2k to
3.5k processed-token rate fails the existing 6k speed gate. Keep `--p1` blocked.
Next, profile the FP32 chunk solve, state propagation and backward; then decide
whether a custom SM75 kernel, smaller pilot or SM80+ hardware is worth pursuing.
Any new kernel needs the same parity suite. Formal P0, LR screens, full-frontier
validation cost, checkpoint retention/upload, frozen decode evaluation and
long-run convergence remain open. No inference TPS, gPPL, MBPP, FLOPS or MFU
result is implied by this training probe.

## Reproduction and local checks

`scripts/build_probe_bundle.py` verifies the source manifest, tokenizer and every
shard, then copies the exact token prefix. It refuses existing destinations.

```sh
python scripts/build_probe_bundle.py \
  --corpus /path/to/training-v1/final --out /tmp/d1-t4/src/probe-bundle \
  --train-sequences 128 --validation-sequences 32
colab new -s d1-t4-check --gpu T4
```

Ship the subset and `scripts/{d1_model,train_diffusion,prepare_corpus,check_gdn_t4}.py`
under `/content/d1-t4/src`, preserving the `scripts/` directory. The original run
used a base64-encoded zip in a `colab exec -f bootstrap.py` script because the CLI
file-upload API failed. Run each command below as a bounded subprocess from a
Colab CLI exec script, recording stdout, stderr and the exit code. Do not run
local CPU fallbacks and label them GPU checks.

```sh
python /content/d1-t4/src/scripts/check_gdn_t4.py --phase parity \
  --root /content/d1-t4 --out /content/d1-t4/parity.json
python /content/d1-t4/src/scripts/check_gdn_t4.py --phase resume \
  --root /content/d1-t4 --out /content/d1-t4/resume.json
python /content/d1-t4/src/scripts/check_gdn_t4.py --phase probe \
  --root /content/d1-t4 --out /content/d1-t4/probe.json \
  --wall-cap 600 --update-cap 24
```

Apply an outer process timeout, recover reports/logs, and stop only the session
created for this work with `colab stop -s d1-t4-check`. The checker requires an
actual Tesla T4 and PASS parity/resume evidence before probing. The final local
checker refinements and subset builder have CPU regression checks; they have not
been rerun on a fresh T4 since recovery.

Final local checks, all exit 0:

- Main pytest selection: 232 passed, two GPU-dependent cases skipped.
- Real model cache tests: 12 passed. Model wiring: four passed.
- Together those split runs cover the full suite: 248 passed, two skipped.
- Changed helper/trainer selection: 67 passed, one locally skipped GPU case.
- Basic basedpyright: model/validator and probe/builder scopes, zero errors or warnings.
- Ruff F checks, Python compilation and Git whitespace checks passed.
- The subset CLI ran against the original corpus, verifying every source hash.
- Protected root and interrupted worktree HEAD/status/diff hashes match their
  prior records. See `preserved-checkouts.json`.

The bare local `pytest` launcher still exits 127 because its interpreter points
at a deleted Synara AppImage. `python -m pytest` executes the suite successfully.
The original historical failing traceback is unavailable, so this reproduces a
local launcher defect without claiming it proves the historical failure's cause.
The launcher and other checkouts were not modified. Exact commands and scopes
are recorded in `checks.json`; logs are alongside it.
