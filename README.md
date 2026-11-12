# EdgeFast-Diff 100M — benchmark week harness

Measures TPS (tokens/s) and HumanEval pass@1 for the D1 roster on a free T4
(batch 1, 1k in / 1k out, median of 5 timed runs). D1 is 100M; baselines are
size-matched to the 100M band, larger models included with the size gap
acknowledged. Roster lives in `bench/config.yaml`.

## Run (Colab free T4)

```bash
# Colab T4 runtime: upload bootstrap_colab.sh, then
BENCH_REPO_DIR=/content/fast-af BENCH_RUN=none bash bootstrap_colab.sh
cd /content/fast-af                           # bootstrap clones; it never rewrites bench/
python -m bench --config bench/config.yaml --out results.yaml
python -m bench --only smollm2-135m,qwen3.5-0.8b          # one model at a time
python -m bench --only smoldlm-144m,smoldlm-144m-cached   # emits token-agreement + speed comparison
```

Output `results.yaml`: machine provenance plus per-model configuration,
speed/pass/cache-warm accounting, stop reasons, full baseline token IDs and,
for AR models, HumanEval pass@1 with raw per-problem completions for re-scoring.

## Backends (`bench/backends.py`)

| backend  | models | status |
|---|---|---|
| hf     | SmolLM2-135M, Qwen3.5-0.8B (+bnb int4), LFM2.5-350M | ready; speed runs force exactly 1,024 output tokens |
| dllm   | LLaDA-8B native | preliminary; Fast-dLLM mode fails loudly until it is genuinely wired |
| custom | smoldlm-144m cached + uncached policies | wired; legacy 350 tok/s row is uncached and lacks current accounting |
| B2 | Qwen3.5-0.8B + DFlash2 | unavailable; harness refuses to label plain int4 as speculative decode |

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
- DFlash2 (B2) remains unwired and has no 0.8B drafter; it is explicitly unavailable.
- TPS uses total model computation after host-side tokenization, including
  AR prefill or diffusion cache warm. Prefill is also reported separately.

## KV cache: the 350 tok/s baseline is handicapped

`smoldlm`'s own `modeling_smoldlm.py` ships a cached `generate()` (KV cache via
`set_cache_mode`, default 10 steps, temp 0.7). Our `CustomBackend` ran its own
decode loop with **no cache at all**, so every pass re-forwards the whole
prefix + block (1,056-2,048 tokens) instead of just the new tokens.

- Estimated gain from enabling the cache only: **≥2x (700 tok/s conservative,
  ~1,200-1,800 central)**. Estimate, not measured.
- **Wired (2026-11-11):** run the `smoldlm-144m-cached` config row.
  The loop then warms the prefix into the model's cache once and scores each
  denoising pass on the block alone at its absolute position, the same
  read/write split stock `generate()` uses. This is not a pure cache-only A/B:
  the legacy uncached full-sequence path lets the prefix attend to the masked
  block, while the cached policy freezes the prefix. Report token agreement
  alongside speed and treat the rows as two decode policies. Still unmeasured
  on T4; the ≥2x number remains an estimate until then.
- Source: `research/hillclimb-speed-reverify.md` (verdict + arithmetic).
