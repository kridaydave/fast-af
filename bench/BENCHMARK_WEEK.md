# Benchmark week runbook (EdgeFast-Diff 100M)

Prepared 2026-11-06. Run on Colab free T4. Execute tiers in order; do not skip a tier without recording why.

## Preflight (once)

```bash
BENCH_RUN=full bash bootstrap_colab.sh
python -m bench --config bench/config.yaml --out results.yaml
```

Verify CUDA: `python -c "import torch; print(torch.cuda.get_device_name(0))"` should print T4.

## Tier 1 — AR baselines (day 1)

- `Qwen/Qwen3.5-0.8B` (B0 reference, fp16, vanilla decode). No official HumanEval exists; we measure it.
- `HuggingFaceTB/SmolLM2-135M` (size-matched yardstick, fp16).
- `LiquidAI/LFM2.5-350M` (hybrid, fp16). License is "other", not apache; confirm terms before publishing numbers.

B1 = Qwen3.5-0.8B int4 (bitsandbytes 4bit, compute fp16). T4 has no AWQ, so GPTQ/bnb only.

B2 = Qwen3.5-0.8B int4 + DFlash2. **No DFlash2 drafter exists for 0.8B** (only 2B and 4B drafters on HF). Per spec fallback, B2 degrades to plain int4 AR. Record this honestly in results. If you want to test DFlash2, run against `Qwen/Qwen3.5-2B` with `rasyosef/Qwen3.5-2B-DFlash2` and note the size gap.

Commands:
```bash
python -m bench --only qwen3.5-0.8b
python -m bench --only qwen3.5-0.8b-int4
python -m bench --only smollm2-135m
python -m bench --only lfm2.5-350m
```

## Tier 2 — size-matched diffusion vs AR (day 2)

- `HoangHa/smoldlm-144m` (block diffusion, block_size 32) vs `HuggingFaceTB/SmolLM2-135M` (AR).
- smoldlm is NOT HF transformers compatible. It ships its own `modeling_smoldlm.py` and `generate()`. Needs a new backend class or a standalone timing script. Do not assume `AutoModelForCausalLM` works.
- smoldlm caveats: greedy decoding degenerates (use temp >= 0.5), early checkpoint (~26B of 100B tokens), no instruction tuning. TPS numbers may not reflect final quality.
- Optional: `OpenSML/OpenSML-150M` only if CUDA-portable (MLX by default). Skip if not.

## Tier 3 — paradigm check at scale (day 3)

- `GSAI-ML/LLaDA-8B-Instruct` (default config + Fast-dLLM decode) vs `Qwen/Qwen3.5-9B` (AR defender).
- LLaDA needs Fast-dLLM decode wiring (separate from dLLM's native generate). Budget friction.
- LLaDA-8B runs via diffuse-cpp GGUF fork if transformers path fails.
- Optional: Dream 7B (double run), DiffusionGemma 26B-A4B (skip if no fit, honestly).

## Tier 4 — L1 pilot (day 4, zero cost on T4)

Two 100M twins, same data, same budget:
- (a) unlooped, all params distinct
- (b) looped core re-used across denoising steps

Metrics: generative PPL, pass@1 on MBPP (HumanEval stays sealed), TPS. Run both at 16 and 8 steps. Yardstick: SmolLM2-135M (gap to it is the meaningful number).

Kill criterion: if (b) doesn't beat (a) on quality at equal TPS at either step count, drop looping. One clean negative result.

## Known friction items (budget time)

1. smoldlm-144m custom loader — new backend class needed.
2. LLaDA-8B Fast-dLLM decode — wiring, not plug-and-play.
3. EAGLE-3 / DFlash2 — no 0.8B drafter; B2 degrades to plain int4 AR.
4. LFM2.5 license "other" — verify redistribution terms before publishing.
5. T4 has no bf16, no AWQ. fp16 + bnb int4 only.

## Deliverable

`results.yaml` with per-model `speed.{median_tps,min_tps,max_tps}` and `humaneval.pass@1` for AR models. L1 pilot gets its own results file.

## Single open decision

Tokenizer: train our own 32k BPE on a corpus sample, or reuse an existing one (SmolLM2's cosmo2-tokenizer merges + Qwen3 pre-tokenizer, as smoldlm did)? This affects pretrain data prep but not benchmark week. Decide before pretrain, not before benchmark week.