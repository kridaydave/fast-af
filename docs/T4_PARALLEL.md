# D1 on one Kaggle runtime with two T4s

The model remains **109,137,632 parameters**, with 16 layers, context 1024,
32-token blocks and a global batch of 64 sequences. Data volume is separate
from parameter count. The existing Python-heavy training-v1 bundle is an
infrastructure fixture, not the agreed general-base data recipe.

The new optional backend is `chunk-tc`, version 3. It uses FP16 operands for
GDN matrix multiplies with FP32 GEMM outputs and first-order gradients. GDN recurrent state,
triangular solves, decay, residual activations, normalization, master weights
and optimizer state remain FP32. Attention KV and convolution histories follow
the FP16 projection dtype under AMP. Its explicit autograd implementation is needed
because `torch.bmm(out_dtype=torch.float32)` lacks native backward on the tested
stacks. It does not provide higher-order differentiation. The existing `chunk`
backend and CPU recurrence remain available.

This path uses PyTorch CUDA/BLAS operations. It has no FLA/Triton dependency.
It requires CUDA FP16 autocast and support for `bmm` with FP32 output. The final v3 path passed Colab checks on PyTorch 2.14.1+cu130.
Kaggle provides PyTorch 2.11.0+cu130; the final two-T4 qualification
is tracked separately in the measurement record. T4 is SM75;
the pinned FLA/Triton backend still requires SM80 or newer.

An experimental FP16 residual/norm variant failed Kaggle cached/full logits
parity: relative L2 error 0.003039 exceeded the existing 0.003 limit. It was
rejected. Version 3 retains FP32 residuals and feeds FP32 inputs and weights to
RMSNorm, avoiding PyTorch 2.11's mixed-dtype fallback. The cache tolerance was
not loosened. CUDA Graphs and batched GDN output products produced no useful
consistent improvement in their short screens and were left out.

## Two-GPU semantics

Use one process per T4 with `torchrun` and `--world-size 2`. `--batch-size` is
still the **global micro-batch**, before partitioning across ranks:

| Global micro-batch | Per T4 | Accumulation | Global batch |
| --- | --- | --- | --- |
| 8 | 4 | 8 | 64 |
| 16 | 8 | 4 | 64 |

Both ranks read the same deterministic global batches and use identical,
separate corruption generators. Each rank computes only its half of the rows.
This preserves one data cursor and avoids sampler padding or repeated examples.
Counts in reports are global counts once, not the sum of duplicated loader reads.

The objective is weighted by scored tokens across the entire global batch.
A local loss sum is multiplied by `world_size / global_scored_tokens` before
DDP averages gradients. A rank with no scored tokens still participates in
backward with zero loss. DDP synchronizes gradients on the final accumulation
micro-batch. Clipping and AMP overflow handling happen after synchronization;
rejected updates do not advance the learning-rate schedule.

Only rank zero writes checkpoints, configuration and logs. Checkpoints include
both ranks' Torch CPU/CUDA RNG states, the shared corruption generator, loader
cursor, optimizer, scheduler, scaler and counters. Changing world size,
configuration, backend version or data provenance on resume is rejected.
There is no elastic conversion between one- and two-GPU checkpoints.

The global-micro-8 comparison preserves the same eight frontier draws per
update as a single GPU with micro-batch 8. Global micro-batch 16 uses four
frontier draws per update instead. Its token-weighted loss is correct for its
sampled masks, but grouping and gradient statistics change. It needs recipe
and quality requalification before a long run.

## Bounded reproduction

Prepare a small verified byte-prefix of the existing corpus. This reads the
original bundle and writes a new destination:

```bash
python scripts/build_probe_bundle.py \
  --corpus /path/to/training-v1/final --out /path/to/probe-bundle \
  --train-sequences 128 --validation-sequences 32
```

Package it and the sources as a private Kaggle script:

```bash
python scripts/package_kaggle_probe.py \
  --bundle /path/to/probe-bundle --out /path/to/new-kaggle-package \
  --owner YOUR_KAGGLE_USERNAME
kaggle kernels push -p /path/to/new-kaggle-package \
  --accelerator NvidiaTeslaT4 --timeout 1200
```

The package enables two T4s, disables internet, and includes only expected
bundle files and Python sources. Authenticate through Kaggle's supported
credential flow; never put a token in the package. Its job checks real hardware,
CPU output/gradient/cache parity, all 1024 cached tokens, distributed gradients,
bounded timing windows, and process kill/resume. It stops on failed correctness
checks. Temporary multi-GB resume checkpoints are removed after their checks;
reports, logs and checkpoint hashes remain in notebook output.

For a standalone matched-work comparison inside an already allocated runtime:

```bash
python scripts/benchmark_d1_parallel.py \
  --bundle /path/to/probe-bundle --out single.json \
  --backend chunk-tc --fused --micro 8

torchrun --standalone --nproc-per-node=2 scripts/benchmark_d1_parallel.py \
  --bundle /path/to/probe-bundle --out dual.json --world-size 2 \
  --backend chunk-tc --fused --micro 8
```

The defaults are four untimed optimizer updates followed by three windows of
four updates. This is hardware warmup, not learning-rate warmup; the benchmark
uses constant 3e-4 learning rate and initial AMP scale 8. It limits attempts and
checks a wall cap between updates; the notebook also enforces subprocess and
whole-runtime timeouts. It validates shard/tokenizer hashes before measuring.

## Reading the evidence

See [the measurement record](../.verify/d1-t4-parallel/README.md) for results,
source hashes, failed candidates, allocation accounting and limitations.

- **Consumed corpus tokens** count clean packed rows drawn from the loader.
- **Processed input tokens** count the prefix and active block actually run.
- **Scored tokens** count masked targets contributing to the loss.

These are different quantities. Frontier training scores roughly one block's
masked positions per context; consumed corpus tokens are not equivalent to
autoregressive loss tokens or a claim of sufficient pretraining.

Timing windows exclude evaluation and checkpoint writes. The small real-token
subset is revisited and cached locally. A short probe does not replace the
200-attempt P0, long-run AMP stability, full-corpus validation or quality tests.
Kaggle planning uses the [renewable weekly budget](../.verify/d1-t4-parallel/weekly-budget.json):
**30 GPU-enabled session hours per week** for this account, with both T4s
available during those hours. Count active runtime, including GPU-enabled idle
or CPU work, transfers, failures, evaluation and checkpoints, once against that
week's allowance. A six-hour T4×2 run consumes about six quota hours. Do not
halve the 30-hour allowance because two GPUs are present, debit Colab time to
Kaggle, or carry previous weeks' usage into a fresh weekly allowance.

Raw benchmark `gpu_seconds` counts participating devices for compute analysis;
it is a different metric and must not be used as Kaggle quota consumption.
The old cumulative six-hour limit and 2.52-hour balance are withdrawn. Re-read
provider quota at each start and reset, reserve pending work once, and persist
training state through resumable sessions below the documented 12-hour limit.
See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the time scenarios and their assumptions.

`--p1` remains blocked pending weekly runtime enforcement, checkpoint
retention/upload and the frozen evaluation protocol. No long pretraining run
is authorized or launched by these tools.
