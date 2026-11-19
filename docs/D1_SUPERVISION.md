# D1 supervision efficiency

The original trainer samples one 32-token frontier from a 1,024-token context.
It scores only masked positions there, while recomputing the clean prefix for
that frontier. Prefix computation receives useful gradients, but supplies few
loss targets per forward. The old budget counted entire consumed contexts,
including cropped future tokens. It did not measure autoregressive training
volume or establish model quality.

## Train every block with a shared clean pass

`--d1-supervision all-blocks` keeps the model, recurrence and block-diffusion
generation protocol. Each training batch has two streams:

- A clean stream computes the prefix representations once. Its attention sees
  its own block and earlier clean blocks.
- A noisy stream masks every eligible block independently. A noisy block sees
  strictly earlier clean blocks and its own noisy block. Other noisy blocks and
  its current clean answer are inaccessible.

For GDN, the clean scan exposes the recurrent state immediately before each
block. All noisy blocks then run in one batched recurrence, each starting from
its own clean entry state. The chunk backends retain 128-token clean scans and
recover the intermediate 32-token entry states from the same solved updates.
This avoids a separate scan launch for every block. Causal convolutions start from the corresponding raw
clean projection history. Attention uses the same visibility rules in a
combined clean/noisy matrix. Norms and FFNs act independently at each position.

The clean states remain attached to autograd. Later losses train earlier prefix
representations. No inference-only or detached cache is used in training.

For fixed masks, the noisy outputs and gradient of their summed token losses
match independent original frontier forwards. The new update aggregates all
blocks with token weighting. It has more targets and different gradient
variance than an old sampled-frontier update, so equal steps are not an equal
compute or quality comparison. Validation retains the existing frontier mask
distribution and clean-cache protocol.

## Execution and checkpoint rules

The selectable backends are `torch`, `chunk` and `chunk-tc`. FLA does not expose
this implementation's entry-state path and is rejected. The reference and
native chunk paths run on CPU; actual `chunk-tc` execution still requires CUDA
FP16 autocast, with FP32 recurrent state, solves, residuals and normalization.

Dense CUDA training enables deterministic PyTorch algorithms and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` when the caller has not supplied a value.
Unsupported deterministic operations fail explicitly. The previous PyTorch
setting is restored when training exits. This policy also applies to the dense
benchmark. It addresses an optimizer-state resume mismatch found on a real T4;
the tolerance was not relaxed.

The compatibility default remains `frontier`, preserving existing checkpoint
hashes and benchmark commands. New supervision requires the explicit flag and
has a different config hash. A frontier checkpoint cannot silently resume as an
all-block run. Full state, corruption RNG and loader cursor are retained.

Token counters have these meanings:

- Consumed tokens count the clean corpus context once.
- Processed tokens count both streams, exactly twice consumed tokens in this
  implementation. This includes the final clean block, which has no later
  target, and masked attention work. It is not a FLOP estimate.
- Scored tokens count the eligible masked noisy positions, about half the corpus
  tokens under the current mask distribution. EOS/control ids remain unmasked;
  every eligible row/block gets at least one target.

Compare scored tokens/s and held-out quality at matched elapsed compute. A
higher scored-token rate is a supervision-throughput result, not an equal
multiplier on capability or convergence.

## Validation and timing

Measurements and source fingerprints are recorded in
[the dense verification record](../.verify/d1-dense/README.md). CPU checks cover
independent-frontier output and every parameter gradient, convolution histories
including widths longer than a block, answer/future/noisy-block leakage,
reserved tokens, partitioned losses with an empty-scoring rank, partial batches,
accumulation, frozen validation and exact resume.

The selected single-T4 configuration measured about 6.0× more scored targets/s
than the frontier baseline, at micro 2 and 7.58 GiB peak allocation. Micro 4 was
faster but exceeded the existing 85% memory margin. These are short screens,
not production P0 or evidence of better held-out learning.

The GPU checker requires an actual Tesla T4. It compares mixed-precision dense
outputs and all parameter gradients against an independent CPU-reference model,
and checks the full 109M model at context 1024 against the existing cached
frontier path. Full-model CPU gradient parity at context 1024 is not claimed.
Separate process-resume and bounded training benchmarks use the actual 109M
model and real verified tokens. Two-GPU dense qualification remains separate.

For a bounded timing probe, after parity/resume qualification:

```sh
timeout --kill-after=10s 240s python scripts/benchmark_d1_parallel.py \
  --bundle /path/to/verified/probe-bundle --out dense-t4.json \
  --backend chunk-tc --fused --supervision all-blocks --micro 1 \
  --warmup 2 --windows 2 --window-updates 2 --wall-cap 180
```

Use `scripts/check_d1_dense.py --phase parity` and `--phase resume` with
`--root` and `--out` under an external timeout. Resume expects a verified
`ROOT/src/probe-bundle`. Two-T4 gradient/resume checks accept
`--supervision all-blocks` in `scripts/check_d1_parallel.py`; the benchmark
accepts `--world-size 2` via `torchrun`. For a future two-GPU screen, global
micro 4 gives two sequences per GPU, with accumulation 16 retaining global
batch 64. This is an unqualified configuration, not a two-GPU recommendation
backed by measurement. These commands are qualification tools,
not authorization for a long run. Choose micro-batch from measured peak memory
with a margin, and retain global-batch semantics across devices.

The old 2B-consumed-token timing scenarios describe frontier supervision only.
They cannot be reused for this candidate. Keep Kaggle's renewable 30 runtime
hours/week, then choose consumed and scored budgets from new measured rates,
including setup, validation, checkpoint/export and failed work. Actual two-T4
rates and adequate training volume still require evidence.
