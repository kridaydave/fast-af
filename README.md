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
| custom | smoldlm-144m | wired to the repo's `modeling_smoldlm.py`; loop smoke-tested on CPU, TPS run still needs the T4 |

## Honest caveats

- No GPU on the dev box, so nothing here has run on a T4 yet. The smoldlm
  wiring (load, score_fn, decode loop) was smoke-tested on CPU; the backends
  target the T4.
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
