# fast-af

Sonic of LLMs. Slow pokes not allowed.

fast-af is a small, fast language model that runs on your own machine. It's early and may break. Updates come regularly.

This repo holds the model work and the benchmark harness. The harness measures tokens per second and HumanEval pass@1 on a free Colab T4, and it records the machine, the config, and the raw outputs so the numbers can be checked.

## Run it

```bash
# Colab free T4: upload bootstrap_colab.sh, then
BENCH_REPO_DIR=/content/fast-af BENCH_RUN=none bash bootstrap_colab.sh
cd /content/fast-af
python -m bench --config bench/config.yaml --out results.yaml
python -m bench --only smollm2-135m,qwen3.5-0.8b
```

## What's inside

The harness is in `bench/`, the model roster in `bench/config.yaml`, the docs in `docs/`, and the recorded runs in `results_t4*.yaml`. Start with `docs/DATA_PIPELINE.md` and `docs/TOKENIZER.md` if you want to build the data.

The [tasks and feedback workflow](docs/TASKS_AND_FEEDBACK.md) builds verified
Python tasks and grades recorded completions in isolated CPU workers.

## Backends

| backend | models | status |
|---|---|---|
| hf | SmolLM2-135M, Qwen3.5-0.8B (+bnb int4), LFM2.5-350M | ready; speed runs force exactly 1,024 output tokens |
| dllm | LLaDA-8B native | preliminary; Fast-dLLM mode fails loudly until it is genuinely wired |
| custom | smoldlm-144m cached + uncached policies | wired; the legacy 350 tok/s row is uncached |
| B2 | Qwen3.5-0.8B + DFlash2 | unavailable; the harness refuses to label plain int4 as speculative decode |

## Honest caveats

- No GPU on the dev box. Benchmarks ran on a free Colab T4, results in `results_t4.yaml`.
- The diffusion loop is D1's decode-policy prototype. The spec's D1 knobs (0.97 block early stop, prompt cache, block overlap) are not in it yet.
- Historical `results_t4*.yaml` HumanEval rows are completion-protocol, 50-problem runs. They are not valid quality evidence for the chat protocol, and no fixed accuracy is claimed from them.
- TPS counts total model computation after host-side tokenization, including AR prefill or diffusion cache warm. Prefill is reported separately.
- DFlash2 (B2) is unwired and has no 0.8B drafter. It is marked unavailable, not hidden.

## Contribute

Want to help? Pick an issue labeled good first issue, or open one. Keep it small.
