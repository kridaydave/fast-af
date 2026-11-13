# Benchmark week runbook (EdgeFast-Diff 100M)

Prepared 2026-11-06. Run on Colab free T4. Execute tiers in order; do not skip a tier without recording why.

## Preflight (once)

```bash
BENCH_REPO_DIR=/content/fast-af BENCH_RUN=none bash bootstrap_colab.sh
cd /content/fast-af
python -m pytest -q
python -m bench --config bench/config.yaml --out results.yaml
```

Verify CUDA: `python -c "import torch; print(torch.cuda.get_device_name(0))"` should print T4.

## Tier 1 — AR baselines (day 1)

- `Qwen/Qwen3.5-0.8B` (B0 reference, fp16, vanilla decode). No official HumanEval exists; we measure it.
- `HuggingFaceTB/SmolLM2-135M` (size-matched yardstick, fp16).
- `LiquidAI/LFM2.5-350M` (hybrid, fp16). License is "other", not apache; confirm terms before publishing numbers.

B1 = Qwen3.5-0.8B int4. AWQ runs on T4 (AutoAWQ CC 7.5+, verified 2026-11-11; repo archived, vLLM/llm-compressor is the maintained path), but bitsandbytes is what bootstrap installs, so bnb 4bit (compute fp16) is the default here. Note our own T4 run measured bnb int4 SLOWER than fp16 (14.7 vs 18.6, results_t4.yaml), so int4 is an experiment, not a free win.

B2 = Qwen3.5-0.8B int4 + DFlash2. **No DFlash2 drafter exists for 0.8B** (only 2B and 4B drafters on HF). The placeholder row now fails loudly until a real speculative backend is wired; plain int4 must never be reported as B2. If you want to test DFlash2, run against `Qwen/Qwen3.5-2B` with `rasyosef/Qwen3.5-2B-DFlash2` and note the size gap.

Commands:
```bash
python -m bench --only qwen3.5-0.8b
python -m bench --only qwen3.5-0.8b-int4
python -m bench --only smollm2-135m
python -m bench --only lfm2.5-350m
```

## Tier 2 — size-matched diffusion vs AR (day 2)

- `HoangHa/smoldlm-144m` (block diffusion, block_size 32) vs `HuggingFaceTB/SmolLM2-135M` (AR).
- smoldlm is NOT HF transformers compatible. Its custom backend is wired to `modeling_smoldlm.py`, including cached and uncached policy rows. Do not assume `AutoModelForCausalLM` works.
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

## Deferred to the end (decided 2026-11-11)

Tier 1 and the smoldlm speed run are done (see `results_t4.yaml`). Everything
below is pushed to the end of benchmark week. Do not start it early.

- Tier 3 (LLaDA-8B default + Fast-dLLM vs Qwen3.5-9B) — biggest friction item, run last.
- Tier 4 (L1 pilot, looped vs unlooped twins) — needs training code that does not exist yet.
- Clean re-run of Tier 1/2 with the full kernel set installed. The 2026-11-09 run
  was a lean env, and two models missed the 1k output target: smollm2-135m
  emitted 307 tokens and lfm2.5-350m emitted 12, so their TPS is not comparable
  to the rest of the table. Diagnose why they stopped early before re-running.
  smoldlm (350) and qwen3.5-0.8b (18.6) are the two numbers to trust.

## Known friction items (budget time)

1. smoldlm-144m custom loader — wired; next step is the corrected cached/uncached T4 measurement.
2. LLaDA-8B Fast-dLLM decode — wiring, not plug-and-play.
3. DFlash2 — no 0.8B drafter; B2 remains unavailable and plain int4 stays B1.
4. LFM2.5 license "other" — verify redistribution terms before publishing.
5. T4 has no bf16. AWQ does run on sm75 (verified 2026-11-11). fp16 + bnb int4 is what bootstrap installs; bnb int4 measured slower than fp16 on T4.

## Deliverable

`results.yaml` with per-model `speed.{median_tps,min_tps,max_tps}` and `humaneval.pass@1` for AR models. L1 pilot gets its own results file.

## Tokenizer decision (resolved 2026-11-06)

Train our own 32k BPE on a corpus sample. Reuse nothing. This affects pretrain data prep; benchmark week is unaffected.

## G1 historical sampler regression

`bench/token_baseline.py` captures full token IDs from an exact committed
`CustomBackend` source without switching the checkout. Use the parent of
`b741770` for the pre-patch greedy baseline and `HEAD` for the patched sampler.
Both use uncached confidence commits at temperature 0. This comparison isolates
the sampler change; cached-versus-uncached benchmark rows are separate policies.

On the same T4 session, set `G1_MODEL_SHA` to an immutable 40-character commit
from `HoangHa/smoldlm-144m` on Hugging Face, then run:

```bash
# Create g1-prompt.txt with the exact prompt to freeze (UTF-8, including newlines).
# Set this to a real immutable Hugging Face model revision:
export G1_MODEL_SHA='<40-character-model-commit>'
python bench/token_baseline.py capture --role baseline --source-ref 'b741770^' \
  --model-revision "$G1_MODEL_SHA" --prompt-file g1-prompt.txt \
  --output-length 1024 --block-size 32 --steps 8 --confidence 0.9 \
  --dtype float16 --device cuda --seed 0 --out g1-baseline.json
python bench/token_baseline.py capture --role candidate --source-ref HEAD \
  --model-revision "$G1_MODEL_SHA" --prompt-file g1-prompt.txt \
  --output-length 1024 --block-size 32 --steps 8 --confidence 0.9 \
  --dtype float16 --device cuda --seed 0 --out g1-candidate.json
python bench/token_baseline.py compare g1-baseline.json g1-candidate.json \
  --json-out g1-comparison.json
```

Capture requires Git history plus PyTorch, `huggingface_hub`, `safetensors`, and
`tokenizers`. It downloads the model definition, config, weights, and tokenizer
at the specified revision and records their SHA-256 hashes. An optional
`--tokenizer-revision` pins a different tokenizer commit; by default it uses the
model revision. The model's download helper is bypassed so it cannot silently
fetch assets from `main`. The exact committed decode loop runs unchanged, with
a recording tokenizer retaining the historical loop's IDs before text decoding.
The candidate uses committed source too: commit sampler edits before capturing.

The artifacts include source commit/hash, prompt text/IDs, all policy knobs,
seed, dtype, and runtime versions/device. Output length must be a positive whole
number of blocks. Early EOS/short output fails capture; do not treat a short
preview or re-encoded text as a baseline. Capture re-runs the historical sampler
under the recorded assets/environment; it does not recover the November 9 output
or prove that its original model revision is known. Preserve both JSON artifacts
and their comparison report as the newly frozen G1 evidence.

The direct-script comparator uses only Python's standard library. It requires
matching assets, prompts, lengths, policies, seeds, and environments, and distinct
backend source hashes. It reports every mismatching position, total matches,
agreement, source commits, and input artifact hashes. Exit codes are **0** for
agreement >=99%, **1** for a valid comparison below 99%, and **2** for invalid or
incompatible inputs. Invalid comparisons also overwrite `--json-out` with an
error, so a previous passing report cannot be mistaken for the latest result.
No G1 pass is claimed until the real model capture/comparison is run.
