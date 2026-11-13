# Smoldlm cached policy: why it is slow, what was changed

Status: diagnosis and low-risk fixes committed; the one structural
optimization ships opt-in and still needs a T4 validation run. Everything
below separates measured evidence from hypotheses.

## The measured problem

Corrected T4 run (`results_t4_corrected.yaml`, sha e5f29ca, T4, 1024 in /
1024 out, 5 runs, median):

| policy    | TPS     | score passes | warm passes | warm window | total window |
|-----------|---------|--------------|-------------|-------------|--------------|
| uncached  | 316.348 | 53           | 0           | n/a         | ~3.2 s       |
| cached    | 166.474 | 48           | 64          | ~1.86 s     | ~6.15 s      |

Arithmetic on those rows: a cached score pass costs ~89 ms while an
uncached full-sequence pass costs ~60 ms. A cached pass feeds 32 tokens and
reads a 1 to 2k cached prefix; an uncached pass runs the full 1 to 2k
sequence through the model, including the lm_head over every position. The
cached pass does strictly less math and is still slower. Both policies
report token_agreement 1009/1024 and 45 fallback commits, so the quality
side of the cached policy is intact.

Conclusion: the cached policy is bound by per-forward fixed overhead, not
by math. A cached run launches ~112 forwards (64 warm + 48 score), each
with 30 transformer layers of tiny fp16 ops; an uncached run launches 53
large ones. On T4, hundreds of small kernels per forward cost more than
the flops they compute.

## What the pinned model source does per forward

From `modeling_smoldlm.py` (revision 64f1177, read from the local HF cache):

- Every attention forward concatenates the whole cached K/V with the
  current chunk, then GQA `repeat_interleave`s the full concatenated KV to
  9 query heads, then `scaled_dot_product_attention(..., is_causal=False)`.
  The full-prefix attention read is the point of the cached policy; the
  cat and repeat are O(prefix) per layer per pass on top of it.
- `update_cache` copies the growing cache with `torch.cat` per layer per
  warm forward (O(prefix^2) across a warm sequence, in bytes not flops).
- Attention is hard-coded `is_causal=False` and the forward takes no
  attention mask. That is why a plain full-prefix warm is NOT equivalent
  to the stock chunked warm: in a single forward, early chunks would
  attend later chunks and their layer-1+ K/V would change. Measured in
  fp64 below.

## Changes in this commit (bench/backends.py, cached sections only)

1. `cache_warm` now runs under `torch.no_grad()`. It only stores K/V; the
   autograd graph it built before was discarded every call. `score_fn`,
   `score_block` and the stock `modeling_smoldlm.generate()` already run
   no_grad, so this aligns the warm path with everything else. Cache
   values are bit-identical (autograd never changes forward values).
2. The kv path no longer builds the full-sequence `seq` tensor every
   denoising pass. It was passed to `_score` and ignored under kv_cache:
   a python list concat of ~2k ids plus an H2D copy per pass for nothing.
   The uncached path still builds and uses it.
3. `tok.tolist()` once per pass instead of `int(tok[i])` per filled slot.
   Each `int()` on a CUDA tensor was a stream sync; up to 32 per pass.
   Values identical. This sits in the shared decode loop, so the uncached
   policy gets the same fix; outputs are unchanged on both paths.
4. Opt-in `warm_batch=True` on `load_smoldlm` (default False): the whole
   prompt is warmed in ONE forward under a block-causal mask, replacing
   the 32 chunk forwards. The mask lets position p attend exactly the
   keys j <= end of p's own 32-token block, which is precisely the
   position set the chunked warm gives it (cache of chunks 0..c-1 plus
   its own chunk). The stock forward takes no mask, so the attention
   forwards are replaced with a verbatim copy of the pinned revision's
   forward that adds one thing: SDPA reads an optional `_warm_mask`,
   set only for the duration of the warm and cleared in a `finally`.
   Block-commit warms stay chunked; only the initial prompt warm batches.

## Evidence for the batched warm equivalence (measured, CPU, fp32/fp64)

- fp64, 360-token prefix: max cache diff between chunked and batched warm
  is 1.5e-14. The semantics are identical; what differs in fp32 is which
  SDPA backend runs (masked vs unmasked dispatch).
- fp32 CPU, full greedy generate, 6 blocks (192 tokens): token_ids are
  exactly equal between the two warm schedules. Pinned by
  `tests/test_smoldlm_cache.py::test_batched_warm_generate_tokens_equal_chunked`.
- fp32 cache tensors: max abs diff 4.8e-6, asserted at atol/rtol 1e-4.

Residual risk: on CUDA fp16 the masked and unmasked SDPA paths are
mem-efficient vs flash and will not be bit-identical. Token agreement may
move a few positions out of 1024. That is why warm_batch is opt-in and
why the validation run should diff sample_token_ids against the previous
corrected run before anyone quotes a speedup.

## Profile results (measured, CPU, fp32, this box)

`scripts/profile_smoldlm_cache.py` (opt-in; CPU numbers rank ops, they do
not predict T4 kernel times):

- prompt 1024 / out 128: chunked warm = 36 forwards at ~210 ms; batched
  warm = 5 forwards, warm phase 7.5 s to 4.4 s. One 1024-token warm
  forward costs ~3.6 s where one 32-token chunk costs ~0.21 s: per-call
  fixed overhead is ~half of every small forward on CPU.
- Microbench per layer at run shapes: cache cat 0.41 ms, GQA repeat
  1.21 ms, SDPA q=32 vs 1184 cached kv 0.79 ms, warm-forward SDPA with
  block-causal mask 18.1 ms vs 15.6 ms without (mask costs ~16%).
- Warm chunk forward with grad on vs off: 190 vs 185 ms (~3%) on CPU.
  On T4 the python-side share of a tiny forward is larger, so expect more
  there, but treat that as a hypothesis until the validation run.

## Rejected or deferred, with reasons

- Batched warm WITHOUT the mask: changes layer-1+ K/V (future-chunk
  leakage). Rejected on semantics; fp64 measurement confirms it would
  differ. The mask is what makes it legal.
- Merging the per-block commit warm into the block's last score pass: a
  score pass still contains MASK tokens at open slots, so its K/V attend
  mask-token positions; the commit warm feeds the fully denoised block.
  Different inputs, different K/V. Rejected (source-derived reasoning).
- `enable_gqa=True` in SDPA to skip the 1.2 ms/layer repeat: likely a few
  percent on T4, but CUDA backend support and numerics cannot be verified
  from this CPU box. Deferred with the profiler as the measuring stick.
- torch.compile / CUDA graphs: the cache grows every block and
  `cache_mode` toggles per call, so shape guards would recompile the
  loop into the ground. Out of scope here.

## How to run the profiler

    python -m scripts.profile_smoldlm_cache --out-len 128          # loop phases
    python -m scripts.profile_smoldlm_cache --microbench           # op costs
    python -m scripts.profile_smoldlm_cache --warm-batch           # batched warm
    python -m scripts.profile_smoldlm_cache --device cuda --sync   # T4 only

`--sync` adds real synchronization and is profiler-only; the benchmark
timing windows in `bench/backends.py` are unchanged (no added syncs, warm
and score computation stay inside the timed window).

## Next T4 validation

Step 1 (this commit, safe fixes only, no numerics change expected):

    python /content/fast-af/bench/__main__.py --only \
        smoldlm-144m,smoldlm-144m-cached --out /content/results_t4_step1.yaml

Expect: cached `sample_token_ids` identical to `results_t4_corrected.yaml`
(no forward math changed), warm_passes still 64/run, warm window and total
window modestly smaller. If tokens differ, something is wrong, stop.

Step 2 (opt-in batched warm) needs `warm_batch: true` on the
smoldlm-144m-cached entry in `bench/config.yaml` (the run.py pass-through
`warm_batch=m.get("warm_batch", False)` is in place). Then rerun the same
command with a new `--out`. Expect: warm_passes 33/run, warm window well
under 1 s, TPS meaningfully up from step 1; token agreement vs uncached
may move a few positions (SDPA backend change, disclosed above).
