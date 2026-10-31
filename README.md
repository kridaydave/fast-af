# EdgeFast-Diff 1B — benchmark week harness

Measures TPS (tokens/s) and HumanEval pass@1 for the D1 roster on a free T4
(batch 1, 1k in / 1k out, median of 5 timed runs). D1 is 100M; baselines are
size-matched to the 100M band, larger models included with the size gap
acknowledged. Roster lives in `bench/config.yaml`.

## Run (Colab free T4)

```bash
bash bench/setup_colab.sh
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
| custom | smoldlm-144m | **needs wiring**: its `score_fn` + tokenizer in `bench/backends.py` |

## Honest caveats

- Not executed here — no torch/GPU on the dev box. Backends target the T4.
- The diffusion `CustomBackend` block-diffusion loop is a generic prototype
  (block_size 32, 0.9 token unmask, 0.97 block early-stop). smoldlm's real API
  must be confirmed and `score_fn` wired before it runs.
- dLLM generate-arg names (`num_steps`, `max_new_tokens`) verified against the
  dLLM version installed by setup_colab.sh; adjust if the API differs.
- EAGLE-3 (B2) is the known day-of-friction item on T4.
