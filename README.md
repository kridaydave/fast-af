# EdgeFast-Diff 100M — benchmark week harness

Measures TPS (tokens/s) and HumanEval pass@1 for the D1 roster on a free T4
(batch 1, 1k in / 1k out, median of 5 timed runs). D1 is 100M; baselines are
size-matched to the 100M band, larger models included with the size gap
acknowledged. Roster lives in `bench/config.yaml`.

## Run (Colab free T4)

```bash
# Colab T4 runtime: upload bootstrap_colab.sh, then
BENCH_RUN=smoke bash bootstrap_colab.sh       # install deps, write bench/, smoke-run one model
python -m bench --config bench/config.yaml --out results.yaml
python -m bench --only smollm2-135m,qwen3.5-0.8b          # one model at a time
```

Output `results.yaml`: per model `speed.{median_tps,min_tps,max_tps}` and,
for AR models, `humaneval.pass@1`.

## Backends (`bench/backends.py`)

| backend  | models | status |
|---|---|---|
| hf     | SmolLM2-135M, Qwen3.5-0.8B (+int4, +EAGLE-3), LFM2.5-350M | ready; EAGLE-3 spec-decode not auto-wired, runs as plain int4 AR |
| dllm   | LLaDA-8B (default + Fast-dLLM) | dLLM `generate` is called natively; verify API vs installed dLLM |
| custom | smoldlm-144m | ran on T4 (350 tok/s), but the loop is **uncached**; see KV cache below |

## Honest caveats

- No GPU on the dev box. Benchmarks ran on a Colab free T4 (results in
  `results_t4.yaml`, 2026-11-09); the smoldlm wiring itself was smoke-tested on
  CPU before that.
- The diffusion `CustomBackend` loop is D1's decode-policy prototype
  (block_size 32, 0.9 confidence unmasking, a top-1 commit when a pass
  clears nothing so the loop always makes progress, forced fill on the last
  pass so a block never commits mask tokens). The spec's D1 knobs (0.97
  block early stop, prompt cache, block overlap) are not in this prototype.
- dLLM generate-arg names (`num_steps`, `max_new_tokens`) verified against the
  dLLM version installed by bootstrap_colab.sh; adjust if the API differs.
- Both tiers decode greedily (AR via `do_sample=False`, diffusion via argmax).
  smoldlm's card warns greedy text degenerates (repetition shows up in
  `sample_text`); it does not affect TPS.
- EAGLE-3 (B2) is the known day-of-friction item on T4.

## KV cache: the 350 tok/s baseline is handicapped

`smoldlm`'s own `modeling_smoldlm.py` ships a cached `generate()` (KV cache via
`set_cache_mode`, default 10 steps, temp 0.7). Our `CustomBackend` ran its own
decode loop with **no cache at all**, so every pass re-forwards the whole
prefix + block (1,056-2,048 tokens) instead of just the new tokens.

- Estimated gain from enabling the cache only: **≥2x (700 tok/s conservative,
  ~1,200-1,800 central)**. Estimate, not measured.
- **Wired (2026-11-11):** set `kv_cache: true` on the smoldlm config row.
  The loop then warms the prefix into the model's cache once and scores each
  denoising pass on the block alone at its absolute position, the same
  read/write split stock `generate()` uses. Only this flag changes: steps,
  temperature, and the decode contract stay put, so the cached row vs the
  default row isolates the cache. Still unmeasured on T4 — the flag exists to
  produce that measurement, and the ≥2x number stays an estimate until then.
- Source: `research/hillclimb-speed-reverify.md` (verdict + arithmetic).
