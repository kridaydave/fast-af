# Speed hillclimb notes (EdgeFast-Diff 100M, T4)

Scope: speed side of D1 only. Goal: move block-diffusion decode from the measured 350 tok/s baseline toward the spec's 1-3k tok/s claim on a free T4, without inventing numbers.

## Measured baseline (this repo, T4, Colab free, 1k in / 1k out, batch 1, median of 5)

Source: `results_t4.yaml` (2026-11-09, lean env noted in file).

| Model | Kind | Median tok/s | Notes |
|---|---:|---:|---|
| `HoangHa/smoldlm-144m` | block diffusion, bs 32, 8 steps, conf 0.9 | 350.0 | Full 1024 tokens/run. Sample text degenerate (greedy/early checkpoint caveat in file). Speed number usable, quality not. |
| `Qwen/Qwen3.5-0.8B` | AR fp16 | 18.6 | 1024 tokens/run. pass@1 0.06 on 50 HumanEval (measured). |
| `Qwen/Qwen3.5-0.8B` int4 (bnb) | AR int4 | 14.7 | Slower than fp16. pass@1 0.02. |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | AR fp16 | 23.9 | Only 307 tokens emitted/run, not comparable to 1k-target runs. |
| `LiquidAI/LFM2.5-350M` | AR fp16 | 41.4 | Only 12 tokens emitted/run, not comparable. |

Lean env caveat from `results_t4.yaml`: missing flash-linear-attention, causal-conv1d, dLLM kernels. Qwen GDN may have used fallback kernels, so Qwen speed is lean-env, not final.

Prototype caveat from `README.md` / `bench/backends.py`: decode loop has block_size 32, 8 steps, 0.9 confidence unmasking, top-1 commit when a pass clears nothing, forced fill on last pass. Missing spec knobs: 0.97 block early stop, prompt cache, block overlap.

## 1) Why int4 made Qwen slower on T4 (14.7 vs 18.6), and what it means for D1 int4/QAT

**Measured:** 18.6 tok/s (fp16) -> 14.7 tok/s (int4 bnb 4-bit, compute fp16), batch 1, 1k/1k, T4. Same model/repo, same harness (`TransformersBackend`, `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` in `bench/backends.py`).

**Why this can happen (mechanism, partly inferred):**
- At batch 1 decode, work per token is small. Weight-only int4 saves memory bandwidth on weight reads but adds dequantization/packing overhead every matmul. If the kernel or GPU cannot hide that overhead, wall time rises. This is a common small-batch failure mode for weight-only quant on older GPUs; I do not have a T4+bitsandbytes benchmark source fetched in this session to pin the exact crossover (unconfirmed from primary benchmark).
- T4 is Turing (sm75), no bf16, limited tensor-core paths vs newer GPUs. bitsandbytes 4-bit kernels and their fast paths vary by architecture; a suboptimal path on sm75 would explain a regression. Unconfirmed without a bitsandbytes kernel-support matrix citation fetched here.
- Memory-bound vs compute-bound: AR decode at batch 1 is often dominated by reading weights/KV state per generated token. Shrinking weights from fp16 (~1.6 GB for 0.8B) to ~4-bit (~0.4-0.5 GB) helps bandwidth, but only if dequant cost is lower than bytes saved. The measured regression suggests overhead won, at least in this lean stack/config.
- Measurement detail: `TransformersBackend.generate` times the full `model.generate` call including prompt prefill (`bench/backends.py`), but both runs used same 1k-in/1k-out setting so the comparison is directionally valid.

**What this means for D1's int4/QAT lever:**
- D1's 2x bandwidth claim for int4 (`spec/edgefast-diff-100m.md`) is a ceiling from bytes/pass, not a guaranteed tok/s win. On T4 batch 1 it can be negative.
- For a 100M block-diffusion model, weights are already tiny (~0.06 GB at int4 per spec math). Dequant overhead has less to amortize, so int4 may hurt more than help at batch 1 unless kernels are good on sm75.
- Implication: treat int4 as an experiment, not a default. Compare fp16 vs int4/QAT vs GGUF Q4_K_M paths on the actual D1 decode loop and T4 kernels. Spec already allows fallback: "if QAT degrades the pilot badly, fall back to fp16..." (`spec/edgefast-diff-100m.md`).

## 2) Decode knobs from spec not in current prototype

Current prototype (`bench/backends.py` CustomBackend): block_size 32, fixed `steps` passes, per-slot confidence threshold 0.9, if no slot clears threshold commit top-1 confident open slot, last pass force-fills all open slots. No block-level early stop, no prompt cache, no block overlap. Timing includes full forward over `prefix + block` each pass.

| Knob (spec) | What it does | Buys tok/s? | Rough size / source | T4 check |
|---|---|---|---|---|
| Confidence unmasking 0.9 (present) | Unmask all slots with confidence >= threshold each pass instead of one-by-one | Yes, core parallel-unmask win | Paper says Fast-dLLM (arXiv 2505.22618) reports up to 27.6x throughput over vanilla dLLM baselines in their settings with KV cache + confidence-aware parallel decoding (training-free, LLaDA/Dream). This is not a T4 or 100M number. | Already in prototype; tune threshold/factor on smoldlm-144m. |
| Tunable steps / block size (partly present as fixed args) | Fewer steps -> fewer forwards per block; block size trades parallel slots vs confidence | Yes, direct (tokens/pass and passes/block) | Spec cites fixed-state paper: generative PPL 15.3 at 4 steps vs 6.4 at 32 (paper says, per spec). Fewer steps raise tok/s but risk quality cliff. | Sweep 4/8/16 steps and block 16/32/64; log tok/s and sample quality. |
| Block early stop 0.97 (missing) | If block-level confidence high, fill remaining slots and end block early | Yes, skips remaining rounds on easy blocks | No primary paper number in repo for this exact rule; spec calls adjacent work Prophet and defines block score as min confidence over remaining slots (spec's own design, not paper-measured here). Expected gain depends on fraction of "easy" blocks: unconfirmed magnitude. | Implement min-confidence block score; measure rounds saved/block and tok/s delta. |
| Prompt cache (missing) | Encode prompt once and reuse prefix state/KV or equivalent across blocks | Yes, removes repeated prompt compute inside decode window | Fast-dLLM (arXiv 2505.22618, paper says): block-wise approximate KV cache exploits high similarity across adjacent steps; reports large end-to-end gains especially with long prefill (e.g. up to 27.6x in their setup). Fast-dLLM v2 (arXiv 2509.26328, paper says): block-level + sub-block cache, up to 2.5x over standard AR decoding for their block-dLLM. Magnitudes are model/hardware/setup-specific, not T4/100M. | Big lever because current loop re-scores `prefix+block` every pass. Must match model's attention type (bidirectional block diffusion differs from AR KV). |
| Block overlap (missing) | Start next block's first pass while current block finishes | Maybe small on T4 batch 1 | Spec labels "engineering, free" with no quality cost (design claim). Real gain needs GPU/CPU concurrency and pipelining; on single-stream T4 likely limited. Unconfirmed. | Low priority; try only after cache/steps wins measured. |

**Reading:** the biggest likely wins from knobs already specced are fewer effective passes (steps tuning + block early stop) and avoiding repeated prefix work (prompt/prefix cache). All multipliers from papers are "paper says" on other models/hardware.

## 3) Fast-dLLM v2, dLLM-Cache, BPD on a 100M block-diffusion model

### Fast-dLLM v2 (arXiv 2509.26328, fetched abstract)
- Paper says: block diffusion dLLM adapted from pretrained AR models with ~1B tokens fine-tuning; hierarchical caching (block-level cache across blocks, sub-block cache inside partially decoded blocks) + parallel decoding; up to 2.5x speedup over standard AR decoding without compromising generation quality; evaluated as "across diverse benchmarks" in abstract.
- Applies to 100M block diffusion? Partly. The caching ideas (reuse stable context across blocks/sub-blocks) target blockwise bidirectional decoding, which matches D1's setting. But the paper's recipe converts pretrained AR models and reports results in that context; transfer to a tiny from-scratch 100M block-diffusion model is unconfirmed. 2.5x over AR is not the same baseline as 350 tok/s already measured for a 144M block diffusion.
- Verdict: **concept applies, multiplier does not transfer directly**. Worth borrowing cache structure ideas for D1's custom loop; do not expect paper's 2.5x on top of 350 tok/s.

### dLLM-Cache (arXiv 2506.06295, fetched abstract)
- Paper says: training-free adaptive caching for dLLMs (long-interval prompt caching + partial response updates guided by feature similarity), exploits static prompt and mostly stable tokens across adjacent denoising steps; up to 9.1x FLOPs reduction on LongBench-HotpotQA; maintains competitive output quality; tested on LLaDA 8B and Dream 7B.
- Applies to 100M block diffusion? The stability observation is general to masked diffusion decoding and should hold at small scale, maybe even more if blocks/contexts are short. But reported gains are on 7-8B full-attention diffusion LLMs with their step dynamics; 100M block-diffusion has different compute mix (smaller layers, blockwise structure). FLOPs reduction != tok/s on T4.
- Verdict: **promising for prompt/prefix reuse and selective response refresh, but 7B evidence only**. Treat as a design to prototype and measure, not a guaranteed 9x.

### BPD pivot decoding (arXiv 2610.26040, fetched abstract)
- Paper says: Balance-Aware Pivot Decoding selects central pivots within contiguous masked intervals and decodes selected positions in parallel; balanced pivots give logarithmic decoding depth in interval length; combines confidence-based prefilling with confidence-thresholded center selection; evaluated on LLaDA and Dream across 5 math/code/instruction benchmarks.
- Applies to 100M block diffusion? Selection policy is architecture-size agnostic in principle and could fit masked-interval decoding. But evidence shown is on LLaDA/Dream (large dLLMs), not 100M block diffusion. `fastest-llm-4-candidates.md` in repo claims "up to 10.35x throughput vs vanilla diffusion decoding" (arXiv 2610.26040 cited there); that number is not in the fetched abstract, so I cannot confirm magnitude from abstract alone (unconfirmed from primary text I fetched).
- Verdict: **interesting alternative/complement to confidence-only ordering, unproven for small block-diffusion on T4**. Needs a small A/B in the CustomBackend loop.

## 4) Kernel reality on Turing (sm75, free T4)

**Measured/fixed facts:**
- T4 is sm75, 16 GB, ~320 GB/s bandwidth, no bf16, no native fast bf16 tensor cores (spec/hardware notes in repo). Quantized tier in repo plan: fp16 + bnb int4 or GGUF Q4_K_M; AWQ not supported on Turing (`spec/edgefast-diff-100m.md`).
- Lean env run lacked flash-linear-attention, causal-conv1d, dLLM kernels (`results_t4.yaml` notes). Qwen3.5 speed treated as lean-env.
- Prototype `CustomBackend` is plain PyTorch/tokenizers loop, not fused diffusion kernels (`bench/backends.py`).

**What likely runs / breaks (best effort, partly unconfirmed):**
- Plain PyTorch fp16 CUDA ops: run on sm75. This is what current benchmarks used.
- flash-linear-attention (fla-org): fetched repo states it targets efficient implementations for linear attention/SSM models and is platform-agnostic in claims, but I did not find an explicit sm75/Turing support statement in fetched README content. Triton kernel support depends on GPU/Triton/PyTorch combo; on T4 this is a real install risk (unconfirmed).
- causal-conv1d: commonly used with Mamba/GDN stacks and often expects newer CUDA builds; sm75/T4 compatibility is environment-specific and was missing in measured run (unconfirmed from primary package docs fetched here).
- FlashAttention-style Ampere+ kernels: generally unavailable or degraded on Turing; expect fallback attention paths.
- bitsandbytes int4: runs but can be slower than fp16 at batch 1 on T4 (measured above).

**Practical read:** on sm75, correctness/fallback paths matter more than chasing newest kernels. A kernel that "should" be fast but falls back or fails install gives zero gain. Every kernel claim needs a T4 smoke test + speed compare.

## 5) Anything newer (2026) that beats confidence-based parallel unmasking for small models?

**What I can support from fetched sources:**
- BPD (arXiv 2610.26040, 2026, fetched): different selection rule (balanced pivots in masked intervals + confidence-thresholded center selection) rather than pure confidence ranking. Paper says it targets quality-efficiency tradeoff on LLaDA/Dream; it is a candidate to beat or complement confidence-only unmasking in some regimes. No small-model (100M) evidence in abstract, so "beats for small models" is **unconfirmed**.
- dLLM-Cache (ICML 2026 acceptance noted in fetched arXiv page, method from 2025 line) and Fast-dLLM v2 (2025) are caching/acceleration around diffusion decoding, not direct replacements for the token-selection rule.

**What I cannot confirm:** web search failed in this session, so I could not survey 2026 literature beyond sources already known/fetched in repo docs. There may be newer 2026 methods that beat confidence-based parallel unmasking for small block-diffusion models; I did not find and verify them here. Marking as **unknown/unconfirmed** beyond BPD.

## Ranked actions (speed, T4-realistic)

Ranking = expected tok/s upside on measured 350 tok/s smoldlm baseline * likelihood on free T4 * verification speed. Multipliers are hypotheses unless marked measured/paper.

| Rank | Action | Expected effect | Cost/risk | How to verify on T4 |
|---:|---|---|---|---|
| 1 | Remove repeated prefix work in CustomBackend (prompt/prefix cache or avoid rescoring unchanged prefix each pass) | High. Current loop forwards `prefix+block` every step (`backends.py`); caching stable prefix compute can cut large redundant work. Paper motivation: Fast-dLLM 2505.22618 block-wise KV reuse; dLLM-Cache 2506.06295 prompt caching (paper says). No T4 number. | Med engineering risk: block diffusion uses bidirectional context inside block; cache must be correct for this model API. | Implement prefix caching for smoldlm-144m; compare tok/s, rounds, and sample text vs current 350 tok/s at same 1k/1k, 5 runs. |
| 2 | Sweep steps, block size, confidence threshold (and top-k/factor style policy) | Med-high. Directly trades passes/block and tokens/pass. Fast-dLLM 2505.22618 shows confidence-aware parallel decoding matters (paper says, large gains in their setup). | Low code cost, quality risk at very few steps/high threshold. | Grid: steps {4,8,12,16}, block {16,32,64}, conf {0.8,0.9,0.95}; record tok/s and degeneration/quality spot checks. |
| 3 | Add block early stop (spec 0.97, min-confidence over remaining slots) | Med, workload-dependent. Saves rounds on easy blocks; 0 gain on hard blocks. Spec idea; magnitude unconfirmed. | Low-med code cost; wrong block score can hurt quality (spec warns mean can hide bad slot). | Log rounds used per block before/after; A/B tok/s on same prompts; inspect forced-fill diffs. |
| 4 | Try BPD-style pivot/interval selection in the unmask policy | Unknown-med. 2026 alternative to confidence-only order (arXiv 2610.26040, paper says). Could reduce wasted passes by exposing dependencies earlier; unproven on 100M/smoldlm. | Med research/code risk, may not fit smoldlm API cleanly. | Implement as optional selector behind flag; compare tok/s and output sanity vs confidence baseline. |
| 5 | Keep AR/int4 path honest: do not assume int4 speedup on T4 | Measured negative for Qwen (14.7 < 18.6). Avoid spending time on bnb int4 for batch-1 speed unless kernel/path changes. | Low cost (decision). | Any future int4/QAT claim must beat fp16 on same T4 setup; otherwise report fp16. |
| 6 | Fix benchmark comparability and clean env before chasing kernel wins | Needed for signal quality. smollm2/lfm2.5 runs emitted 307/12 tokens (`results_t4.yaml`); lean env missing fla/causal-conv1d/dLLM. | Time cost on Colab. | Re-run with fixed stopping/length; install/check kernels one by one and record deltas. |

## Top 5 ordered experiments for T4

### 1. Prefix-cache the diffusion loop
- **Hypothesis:** Re-scoring the full prompt every denoising pass is a major avoidable cost in `CustomBackend`; caching/reusing prefix computation will raise smoldlm-144m tok/s materially at 1k/1k.
- **Number to hit:** >= 450 tok/s median (>= ~1.3x over measured 350) with same block/steps/conf and 1024 emitted tokens, 5-run median.
- **Kill criterion:** < 380 tok/s after a correct implementation, or output becomes clearly more degenerate than baseline at same decode settings.

### 2. Decode-policy sweep (steps, block size, confidence)
- **Hypothesis:** 8 steps / block 32 / conf 0.9 is not the speed optimum; fewer steps or larger blocks can increase tokens/s if the model still commits coherent blocks.
- **Number to hit:** Find a config with >= 500 tok/s and no worse sample sanity than current 350 tok/s run (spot-check, since HumanEval not wired for smoldlm in roster).
- **Kill criterion:** No config reaches >= 400 tok/s without obvious collapse (repetition/mask artifacts/garbage worse than baseline sample).

### 3. Block early stop (min confidence over remaining slots, 0.97)
- **Hypothesis:** Many blocks become confident before the final fixed steps; early stop cuts average passes/block and raises tok/s.
- **Number to hit:** >= 10% reduction in mean denoising passes/block and >= 1.1x tok/s vs experiment 2's best fixed-step config.
- **Kill criterion:** < 3% passes saved or no measurable tok/s gain on same prompt set.

### 4. A/B unmask selector: confidence-only vs BPD-style pivot selection
- **Hypothesis:** BPD's balance-aware pivot selection (arXiv 2610.26040, paper says) can reduce decoding rounds or improve the speed/quality point compared with confidence-threshold unmasking on block diffusion.
- **Number to hit:** >= 1.1x tok/s at similar sample quality, or clearly better sample quality at similar tok/s, vs experiment 2/3 winner.
- **Kill criterion:** Worse or equal tok/s and no quality gain after a fair implementation pass (one focused iteration, not weeks of tuning).

### 5. Kernel/env sanity pass on T4 (only after loop wins)
- **Hypothesis:** Some missing kernels (fla/causal-conv1d/dLLM) change throughput, but gains are uncertain on sm75 and may not affect the smoldlm-144m custom path.
- **Number to hit:** Any single kernel/env change that gives >= 1.05x on a target model's tok/s with correctness unchanged.
- **Kill criterion:** Install/runtime failure on T4, fallback path, or < 1.02x delta. Do not block D1 loop work on this.

## Sources

- Repo (measured/implementation): `results_t4.yaml`, `bench/backends.py`, `README.md`, `bench/BENCHMARK_WEEK.md`, `spec/edgefast-diff-100m.md`, `fastest-llm-4-candidates.md` (candidate claims labeled where not re-fetched).
- Fast-dLLM (paper says): Wu et al., "Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding", arXiv:2505.22618, https://arxiv.org/abs/2505.22618
- Fast-dLLM v2 (paper says): Wu et al., "Fast-dLLM v2: Efficient Block-Diffusion LLM", arXiv:2509.26328, https://arxiv.org/abs/2509.26328
- dLLM-Cache (paper says): Liu et al., "dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching", arXiv:2506.06295, https://arxiv.org/abs/2506.06295
- BPD (paper says): Lee et al., "Balance-Aware Pivot Selection for Parallel Diffusion LLM Decoding", arXiv:2610.26040, https://arxiv.org/abs/2610.26040
- BD3-LM (paper says): Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models", arXiv:2503.09573, https://arxiv.org/abs/2503.09573

Web search failed in this session (HTTP 402), so 2026 coverage beyond fetched/cited sources is incomplete and marked unconfirmed where relevant.
