# D1 architecture implementation

Implemented 2026-11-14 after the training-pipeline integration review. The
unlooped hybrid, blockwise corruption, cached development decoder and training
path exist. CPU checks are available. CUDA/FLA execution, P0 throughput and P1
validation remain pending; `--p1` still refuses to start.

## Chosen model

`configs/d1.json` is the explicit architecture configuration. The default model
has **109,137,632 registered parameters**, counted on the instantiated model.
The full-width CPU forward/cache check uses that model, not a smaller stand-in.

| Part | Decision |
|---|---|
| Depth and width | 16 blocks, width 640 |
| Mixers | Forward GDN at 12 blocks; attention at blocks 4, 8, 12, 16 |
| Attention | 10 query heads, 2 KV heads, head dimension 64 |
| GDN | 4 heads; key dimension 128, value dimension 256 |
| GDN gates | Positive decay, sigmoid update gate, SiLU output gate |
| Convolution | Causal depthwise convolution, width 4, on projected q/k/v |
| FFN | SwiGLU, hidden dimension 1728 |
| Normalization | Pre-norm RMSNorm, epsilon 1e-5; per-value-head output RMSNorm in GDN |
| Embeddings | 32k vocabulary, tied output; no learned absolute positions or RoPE |
| Initialization | Normal 0.02 matrices; residual outputs scaled by 1/sqrt(2 * depth) |
| Context and block | Pilot context 1024; block size 32 |
| Precision | FP32 parameters, optional CUDA fp16 autocast and gradient scaler |

The GDN layer implements the ordinary forward gated delta rule. Its reference
recurrence is differentiable FP32 PyTorch. There are two accelerated paths. The
`fla` backend calls the existing FLA `chunk_gated_delta_rule` operator and needs
SM80 or newer. The `chunk` backend is plain PyTorch CUDA: chunks up to 128 tokens use an FP32
unit-lower-triangular solve, with smaller chunks for short inputs. Version 2
batches state-independent work and collects chunk gradients without repeated
full-tensor slice accumulation. The exact recurrence survives. Neither path invents a bidirectional GDN or
claims to reproduce the bidirectional-Mamba model.

The attention layers exchange information in both directions inside a block.
Each query can see its own block and earlier blocks, but not later blocks.
Consequently the whole hybrid can use right-hand context inside the active
block even though its recurrent layers scan left to right. Quality of this
particular hybrid remains an experimental question.

Attention still stores a growing KV cache. Only GDN's recurrent and convolution
states have fixed size; the whole hybrid does not have constant-memory caching.

## Training objective and accounting

For each micro-batch, sample one shared frontier uniformly from blocks with
eligible targets. Keep the prefix clean, corrupt a random subset of the active
block using the configured rate, and crop the future completely. Score masked
positions in the active block only, with token-weighted accumulation. EOS and
reserved IDs remain excluded; forcing one eligible mask on otherwise empty
rows retains the existing short-row bias.

This objective is **masked cross-entropy**, not an implemented diffusion NELBO
or sequence-likelihood estimator. Logs use `masked_ce` and `val_masked_ce`.
The returned `losses` list is the training masked-CE history.

One sampled frontier does not score every token in the loaded context. Counters
therefore distinguish:

- `tokens_seen`: all corpus tokens consumed from the loader, including hidden
  future tokens. This remains bound to the resumable loader cursor.
- `tokens_processed`: prefix plus active tokens actually passed to the model.
- `tokens_scored`: masked active targets contributing to the loss.

All counters include attempted batches skipped after AMP overflow. Successful
updates remain separately logged as `optimizer_steps`. Budgeting 100M consumed
tokens is not the same as training on 100M scored tokens. The previous token/hour
estimates cannot be reused without measuring this objective and reporting these
denominators. Reprocessing clean prefixes also affects compute.

Sampled evaluation chooses one deterministic frontier per batch. `--eval-full`
visits every frontier of every validation batch using a separate, fixed RNG.
It averages over the sampled masked targets at those frontiers and restores the
caller's model mode and training RNG. Visiting all frontiers is more expensive
than the old single-forward full-sequence objective.

## Cache contract

`D1LM.forward` computes the uncached block-causal function.
`prefill` consumes whole clean blocks. `forward_block` returns logits and a new
candidate cache, without mutating its input cache. It accepts at most one block
and requires an existing cache to end at a complete block boundary.

Each layer's cache holds either attention K/V tensors or GDN's FP32 delta state
plus all three causal-convolution tails. During denoising, reuse the same clean
prefix cache on every round and discard each noisy candidate cache. Once the
block is finalized, process its clean tokens once more and commit that cache.
This final pass is required and counted.

`D1LM.generate` implements fixed-length greedy confidence decoding for development
at a requested round count, default 8. It supports a partial final prompt block,
preserves its known tokens, excludes reserved output IDs, and reports prefill,
denoising, clean-commit and total pass counts. It restores the caller's model
mode. It does not stop on EOS or replace the frozen benchmark decoder. EOS
termination quality is not established by an objective that never masks EOS.

## Running it

The CPU reference needs the existing PyTorch and data dependencies. It is for
correctness and small development runs, not production throughput. A custom
JSON with reduced dimensions can exercise the same architecture code in tests.

```bash
# Full-width, real model on CPU; forward/cache checks only, no speed claim.
python -m scripts.validate_d1 --device cpu --forward-only --out /tmp/d1-cpu.json

# Original small classical-mask smoke transformer is still available.
python scripts/train_diffusion.py --bundle /path/to/bundle --out /tmp/smoke

# SM80+ GPU host: use an isolated environment and a compatible CUDA PyTorch build.
python -m pip install -r requirements-data.txt -r requirements-d1.txt
python -m scripts.validate_d1 --device cuda --out /tmp/d1-cuda.json
python -m pytest tests/test_d1_model.py::test_fla_cuda_outputs_states_and_gradients_match_torch \
  tests/test_train_diffusion.py::test_d1_cuda_amp_checkpoint_resume -q

# Development training after the hardware checks, not a P1 certification.
python scripts/train_diffusion.py --bundle /path/to/bundle --out /tmp/d1-dev \
  --profile d1 --d1-config configs/d1.json --gdn-backend fla --device cuda --amp \
  --batch-size 4 --grad-accum 16 --mask-min 0.1 --mask-max 0.9 \
  --steps 100 --warmup 10 --eval-full --verify-hashes
```

The FLA source is pinned to `2705ab8ee9a9086bcfa6d079858e4281e01215a1`.
The adapter requires that installation revision, CUDA, and fp16 autocast. It
fails instead of falling back when the dependency or kernel is unavailable.
Modern [Triton supports NVIDIA compute capability 8.0+](https://github.com/triton-lang/triton/blob/main/README.md), while [NVIDIA lists T4 as 7.5](https://developer.nvidia.com/cuda/gpus). The pinned FLA backend therefore rejects T4/SM75 by design. Runtime
versions are recorded and backend-version changes reject resume.

The `chunk` backend needs no Triton. Version 2 is validated on a real T4.
Select it with `--gdn-backend chunk`. It also lets the trainer project only scored
hidden positions to the vocabulary, while keeping full prefix computation and
gradients. This preserves the objective and leaves decoding logits unchanged.
The model still has 109,137,632 parameters. See the
[evidence and exact source hashes](../.verify/d1-t4-speed/README.md).

| Check | Result |
|---|---|
| Recurrence output, final state and all six input gradients | PASS, nine lengths 1 to 257 in FP32 and mixed precision, worst relative output error 1.34e-6 |
| Layer parity, input and all 13 parameter gradients | PASS in FP32 and fp16 |
| Full-model logits and committed cache vs CPU | PASS, relative logits error 2.08e-6 FP32 and 2.11e-3 fp16 |
| Full-model dense-reference loss vs scored-only GPU head, all 254 parameter gradients | PASS, worst gradient relative L2 error 1.06e-5 FP32 and 6.73e-3 fp16 |
| AMP process resume | PASS, six applied updates. Losses, loader, scheduler, scaler and Torch/CUDA RNG match; weight max error 2.87e-4 |
| Short accumulated-batch probe, micro 8 / accumulation 8 | 8,464 processed tokens/s full context, 7,940 uniform frontier; peak allocated 12.19 GiB |

Full-model forward/cache parity used 96 tokens and gradient parity used 32;
resume, throughput and production P0 used context 1024. These scopes should not
be conflated. FP32 recurrent state and solves remain in use under fp16 autocast.
GradScaler may skip an overflowing update without advancing the scheduler; the
production warmup, unlike the short probe, exposed this and its counts are
reported separately. Scale 256 passed initial parity but is not a promise of
zero future overflows. The final calibrated scale is 8; fresh parity and process-resume checks passed
at that scale. Set `--amp-init-scale 8` explicitly for the recommended T4 job.

Backend metadata is now `torch-chunked`, version 2, maximum chunk size 128.
Training resume rejects version-1 checkpoints rather than silently changing
numerical execution. Parameter names and shapes remain unchanged. The earlier
version-1 measurements remain available in `.verify/d1-t4/` as historical evidence.

`scripts/validate_d1.py` defaults to CUDA and exits 2 with a SKIP report when
CUDA is unavailable. It never substitutes CPU. On a selected device it checks
real full-model forward, cache equivalence, prefix immutability and future
isolation, plus a real backward and AdamW update unless `--forward-only` is set.
It uses synthetic vocabulary tokens, measures no P0 throughput, and does not
certify GPU resume. The separate GPU pytest exercises AMP/checkpoint resume.

Checkpoint format is now 4 because the architecture/config, counters and metric
contract changed. Earlier checkpoints are rejected. CPU exact resume is tested
with frontier sampling, partial batches, accumulation and full validation.

## Evidence and remaining gates

The original integration report describes the state at `fda920e`; its statement
that D1 did not exist is historical. Current implementation evidence is stored
separately under `.verify/d1-implementation/` after checks complete.

Version 2 clears the 6,000 processed-token/s stop threshold on T4. Full-frontier validation now reuses clean-prefix caches and takes 5 minutes
53 seconds for all 960 contexts; the production P0 protocol and timings are in the
optimization evidence; the revised recipe uses separate consumed, processed and scored token
counts and a six-GPU-hour ceiling for P0/P1. This does not satisfy the 13,000
processed-token/s green band or establish a quality result.

`--p1` remains blocked for protocol and operational review. Checkpoint retention,
upload and the production wall-time controller still need an executable plan.
The frozen decode evaluation policy, long-context quality, looping, quantized
training, gPPL and MBPP results remain outside these infrastructure checks.

Primary references: [Gated DeltaNet](https://arxiv.org/abs/2412.06464),
[FLA operator](https://github.com/fla-org/flash-linear-attention/blob/2705ab8ee9a9086bcfa6d079858e4281e01215a1/fla/ops/gated_delta_rule/chunk.py),
[block-diffusion cache lifecycle](https://arxiv.org/html/2609.11998v1#S3),
and [PyTorch GQA](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).

Completed CPU evidence includes a real 109,137,632-parameter forward/cache
check, independent recurrence and gradient checks, and exact process-kill/resume
on real corpus contexts with a reduced-width 522,740-parameter D1. The latter
consumed 6,144 corpus tokens, processed 4,032 prefix/active tokens, and scored
92 masked targets. It is an infrastructure check, not a training-quality or
full-width backward result. See `.verify/d1-implementation/README.md` for the
per-check scope, source commits, skipped GPU checks and raw logs.
