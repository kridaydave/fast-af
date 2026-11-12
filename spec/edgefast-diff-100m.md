# EdgeFast-Diff 100M — block-diffusion hybrid spec (v3)

Revised 2026-11-01: target size retargeted from 1B to 100M (user call). With zero budget, the 100M is the real, trainable model — not just a pilot. A 1B scale-up stays future work if compute appears. Supersedes v2 of this spec (1B target) and `edgefast-1.5b.md` (AR edge model, kept for reference). Written 2026-10-28 after the hybrid-attention and diffusion-decoding research passes.

## Goal

Maximum single-stream TPS. One guardrail: pass@1 on HumanEval tracked at every step, because a model that outputs garbage at 50k tok/s wins nothing.

The governing formula:

> TPS ≈ bandwidth ÷ bytes read per forward pass × tokens emitted per forward pass

The design attacks both terms: a mostly-delta-rule backbone shrinks the per-pass read (no growing KV cache), and block-diffusion decoding emits several tokens per pass (Mercury's trick, LLaDA's missed opportunity).

## Baseline

B0: Qwen3.5-0.8B (researched 2026-10-30: Qwen3.5 small series is 0.8B / 2B / 4B / 9B; 0.8B is the closest small hybrid). fp16, vanilla decode, batch 1. Everything is measured against it: same GPU, batch 1, 1k in / 1k out, median of 5 runs. B1 (bitsandbytes post-training int4 in the benchmark harness) and B2 (DFlash2, currently unavailable for the 0.8B target) build on the same Qwen model, not on a separately trained baseline. Known caveat: D1 is 100M vs B0's 0.8B — 8x smaller, so a TPS win over B0 is expected from size alone and proves little by itself. The honest size-matched comparison is against SmolLM2-135M (see benchmark week); B0 is the "latest small hybrid" reference point. The real guardrail is quality: D1's pass@1 will be far below B0's, and the gap is tracked, not hidden. No official code benchmarks exist for Qwen3.5-0.8B, so its HumanEval pass@1 is measured by us. Reference numbers in this doc are H100-scale; all actual benchmark runs happen on Colab's free T4 (see hardware section).

| Step | Config | Expected vs B0 |
|---|---|---|
| B0 | vanilla AR (Qwen3.5-0.8B fp16) | 1x (the reference) |
| B1 | B0 + bitsandbytes post-training int4 | 0.79x measured on the lean T4 stack; the old ~2x expectation was invalidated |
| B2 | B0 + int4 + DFlash2 (replaces EAGLE-3) | unavailable for 0.8B until a real drafter/backend exists |
| D1 | this spec | 10x+ hoped |

If D1 doesn't clearly beat B2, there is no reason for it to exist.

## Architecture

~100M params, 16 blocks, d_model 640. Same shape family as the old 1B design, scaled down (was 24 blocks, d_model 2048). Verify the exact param count in code before locking the config.

| Component | Choice | Evidence |
|---|---|---|
| Linear layers (12) | Gated DeltaNet (GDN) | Delta-rule layers win recall: GDN-2 > GDN > KDA > Mamba-3 > Mamba-2 at 1.3B (arXiv:2605.22791). GDN ships in Qwen3-Next, kernels exist. GDN-2 is the stretch option once it matures. |
| Attention layers (4) | Full attention, GQA 2 KV heads, head dim 128, NoPE | 3:1 ratio, uniformly interleaved (attention at blocks 4, 8, 12, 16). Recall saturates at 3:1 (arXiv:2507.06457); Kimi Linear measured 3:1 best PPL. Full attention on sparse layers is what every shipped hybrid uses; sliding window retracted (Waleffe et al. found windowed hybrids degrade past the window, arXiv:2406.07887). 5 query heads at d_model 640, so 2 KV heads keeps GQA a real saving. |
| Placement | Uniform interleave | Waleffe et al.: no ordering beats evenly dispersed. |
| FFN | SwiGLU, 8/3 expansion (1728, 64-multiple) | Standard. |
| Norm | RMSNorm, pre-norm | Standard. |
| Vocab / embeddings | 32k, tied | At 100M the table is a bigger share of the budget (~20M of ~105M); 32k tied keeps it affordable while covering code tokens. |
| Train context | 32k | Mamba state is unbounded at constant cost; attention layers cache the prompt once. |
| Precision | int4 QAT from the start | ~0.5-2pt quality cost for ~2x bandwidth ceiling. At 100M the quality cost of int4 hurts more than at 1B; if QAT degrades the pilot badly, fall back to fp16 and keep the speed win honest. |

## Decoding

Block-diffusion in the BD3-LM / fixed-state style (arXiv:2503.09573, arXiv:2609.11998):

- Block size 32 masked slots. LLaDA and the fixed-state paper both converge on 32.
- 8-16 denoising steps per block, so ~2-4 tokens revealed per forward pass. NOT LLaDA's 512-step config, which is 1 token per pass and slower than AR. The LLaDA lesson: speed lives in the decode policy, not the architecture.
- Unmasking: parallel with confidence threshold (0.9, Fast-dLLM style) rather than strict low-confidence sequential ordering. Measured TPS wins live here.
- Early stopping (user idea, 2026-10-30): a separate block-level confidence score, checked once per round. Above 0.97, fill every remaining slot and finish the block, skipping the unused rounds. The two thresholds coexist cleanly: 0.9 governs individual tokens within a round, 0.97 governs whether the block continues at all. Block score defaults to the MIN confidence across remaining masked slots, not the mean, because a 97% average can hide one garbage slot and commit it. (Adjacent published work: Prophet early-stopping; block-score definition is ours.)
- Block overlap (engineering, free): start the next block's first pass while the current block runs its last, so the GPU never idles between blocks. No quality cost.
- Deferred knobs, do not build yet: (a) adaptive rounds by difficulty — post-D1, fused with the L1 verification skill; difficulty is hard to define and can flip mid-generation, so it needs the model's own judgment rather than a static classifier (user call, 2026-10-30). (b) Adaptive block size (AdaBlock-dLLM) — noted as the trickier cousin of early stopping.
- Few-round training: the model must be fine-tuned/distilled FOR the 4-8 round regime, not trained for many rounds then starved at decode time. Consistency-models-style: teach the model to land guesses early instead of hedging across rounds. Lives in training recipe step 2.
- Prompt encoded once and cached. Attention layers cross-attend the cached clean prefix; GDN layers carry a constant-size recurrent state across blocks. Training computes the clean prefix the same way inference does, so the cache is exact, not approximated.

Back-of-envelope on T4: 100M int4 ≈ 0.06 GB per pass, T4 bandwidth ~320 GB/s → ~5,000 passes/s ceiling, times ~3 tokens per pass ≈ 15k tok/s paper math; expect a third of that after scheduling overhead and remasking waste. The fixed-state paper measured 297 tok/s single-stream for a 3B pure-Mamba block-diffusion model at 256k context, and 231 for the hybrid — at 8x smaller with the same decode trick, meaningfully higher than both is the target.

## Training recipe (revised 2026-10-30: no budget, no teacher)

User decisions on record: no paid teacher distillation (compute costs money), SFT stays, self-distillation after the base model replaces both teacher distillation and hard RLVR for v1.

1. Pretrain ~2-10B tokens (code-heavy mix, FineWeb-Edu + code + math), block-diffusion objective, block size 32, context 1024 then extend. At 100M, a few billion tokens is the realistic free-tier budget (Kaggle T4, 30h/week); the fixed-state 3B trained on only 300B tokens with acceptable quality, so 100M on a few B tokens is not crazy. This IS the pretrain — there is no bigger run behind it. POLICY (user call, 2026-10-30): the pretrain is boring by design, proven defaults only, no experimental knobs, no data experiments. All experiments live in L1 and the decode policy. The only question the run answers is "does the recipe work at all."
2. SFT on public instruction/code datasets (Magicoder / OSS-Instruct / Tulu class). These datasets are already teacher-distilled by their authors; we use the outputs without paying for the teacher. Few-round decode setting (4-8 rounds) applies here too. DATA HYGIENE (user call, 2026-10-30, "slop inheritance" worry): teacher-generated SFT data carries the teacher's tics (tool-call JSON, filler openers, emoji headers, agent-speak). Big models learn past the slop; a 100M model copies surface patterns, so it learns the slop AS the skill. Rules: small and clean over big and sloppy (LIMA lesson, target tens of millions of SFT tokens max); prefer sources seeded from real code over chat dumps; mechanically scrub tool-call traces, filler openers, and formatting slop before training; keep a general-chat slice in the mix; eval on non-code prompts during SFT.
3. Self-distill (STaR / ReST style): model generates many solutions to generated problems, a sandbox runs the tests, only passing solutions become fine-tuning data, repeat. Tests are free judges. This is the v1 replacement for RLVR.
4. RLVR (policy-gradient, tests as reward) stays a later-stage upgrade when compute exists. DiffuCoder's finding still on the table for that stage: RL post-training made the model tolerant of halved denoising steps, so RL buys competence AND speed.

Goodhart defense for the "test fixer" failure mode (model learns to pass tests without writing good code): many varied generated problems, test suites held out from training, quality filters beyond test-passing (dedup, length sanity), HumanEval never touched until final eval. Risk is managed, not eliminated; self-distillation is the mildest test-based training signal, which is part of why it beats hard RLVR for v1.

## Benchmark week (before any training)

D1 is 100M. Baselines and contenders are size-matched to the 100M band where possible; larger "latest small" models are included with the size gap acknowledged (honest direction).

1. Reproduce AR baselines at ~100M: SmolLM2-135M (yardstick) and the latest small hybrid Qwen3.5-0.8B (GDN backbone, same family as D1) + LiquidAI LFM2.5-350M. B1 is bitsandbytes post-training int4. B2 is DFlash2 and remains unavailable for the 0.8B target until a real drafter/backend exists; the harness refuses to mislabel plain int4 as B2. No Qwen model exists near 100M; size gap is acknowledged.
2. Size-matched diffusion vs AR at ~100M: HoangHa/smoldlm-144m (144M, block diffusion, block_size 32, Apache 2.0, latest) vs SmolLM2-135M. OpenSML/OpenSML-150M (150M, AR, Sep 2026) only if CUDA-portable (MLX by default). These numbers are the whole scoreboard for D1's own scale.
3. Paradigm check at scale: LLaDA-8B (default config AND Fast-dLLM decoding) vs Qwen3.5-9B as AR defender. Dream 7B double run optional. DiffusionGemma 26B-A4B optional (skip honestly if no fit). If diffusion can't win after tuning decode policy at 8B, D1 pivots to B2 + competence training.
4. Optionally run the fixed-state authors' setup if code exists (nearest proxy for D1).

## EAGLE-3 replacement (B2) — revised 2026-11-05

EAGLE-3's reference stacks target newer GPUs (H100+), so on a free T4 it was budgeted as "a day of friction" and ran as plain int4 AR until wired. Research passes since the spec was written show better, T4-friendly options. The replacement is not another autoregressive drafter — it must stay consistent with D1's block-diffusion thesis.

| Replacement | Draft style | Key trait | T4 fit | Source |
|---|---|---|---|---|
| DFlash / DFlash2 | block-diffusion drafter | emits a whole block of draft tokens in one forward pass; conditioned on target hidden states; ~2.5x faster than EAGLE-3 | strong — diffusion drafting matches D1's decode policy; small draft model | arXiv:2602.06036 (ICML 2026); llama.cpp `draft-dflash` / `draft-dflash2` (merged 2026-08-27) |
| DART | parallel logits | predicts multiple masked positions in one forward pass; N-gram-enforced tree pruning; ~30% over EAGLE-3 | strong — no autoregressive drafting rollout | EMNLP 2026 (2026.aclanthology.org/2026.emnlp-main.492) |
| P-EAGLE | parallel EAGLE | parallelizes EAGLE-3's autoregressive drafting | moderate — still EAGLE-family | arXiv:2602.01469 |
| DSpark | DFlash + Markov head | adds per-position confidence for adaptive verification | moderate | vLLM Speculators |
| SuffixDecoding / n-gram prompt lookup | draft-free | no trained drafter; reuses prompt/output n-grams | strong — zero draft cost, architecture-agnostic | vLLM; llama.cpp `ngram-simple` |

**Selection:** DFlash2 as the B2 replacement. It is block-diffusion-native (same paradigm as D1's decode loop), ships in llama.cpp `draft-dflash2` (auto-recognized from a supported checkpoint), and reports up to 2.5x EAGLE-3 speedup. Fallback chain: DART → P-EAGLE → DSpark → SuffixDecoding (draft-free, zero training). If none wire cleanly on T4, B2 is unavailable; plain int4 remains B1 and must not be relabeled.

**No EAGLE-3 drafter is trained or downloaded.** D1 is 100M; there is no pre-trained EAGLE-3 drafter for a 100M target, and training one adds a mechanism the pilot doesn't need. DFlash2's draft model is small and target-conditioned; if a Qwen3.5-0.8B DFlash2 drafter appears, use it; otherwise B2 remains unavailable.

EAGLE-3's original caveat stands: "budget a day of friction." That day is now spent evaluating DFlash2/DART instead of wiring EAGLE-3.

## Hardware reality

Budget status (updated 2026-10-30): no training budget. Lightning AI's free GPU tier is gone. Everything runs on free tiers until credits or money appear.

- Benchmarks run on Colab free T4 (16 GB, sm75, ~320 GB/s), chosen for ease of use. Identical silicon to Kaggle's T4, and the second T4 was never used (batch 1, one benchmark per GPU), so the original math is unchanged.
- The 100M pretrain is the plan now (Kaggle T4 free tier), not a rehearsal for one. A 1B run (rented H100, ~$1-1.5k at 200-300B tokens) stays future work if credits or grants appear. The speed thesis (TPS) is fully testable at 100M; only 1B-scale pass@1 quality needs the bigger run.
- Zero-budget sequencing: benchmark week on Colab → pretrain D1 (100M, same architecture as specced) free on Kaggle T4 (30h/week, checkpoints to Drive every N steps; Colab's short sessions make it bad for training) → with proof the physics works, apply for compute grants or rent for the 1B scale-up.

T4 consequences (unchanged):

- T4 has no bf16. AWQ does run on T4 (sm75): AutoAWQ requires Compute Capability 7.5 "Turing and later" (README, verified 2026-11-11), vLLM lists AWQ for Turing. AutoAWQ is archived (read-only since 2025-05); the maintained path is vLLM's llm-compressor. Full-precision tier is fp16; quantized tier is AWQ/GPTQ int4 or llama.cpp Q4_K_M. Turing also misses the Ampere+ attention kernels, so absolute TPS will sit below the bandwidth math; ratios between configs are the deliverable, and those carry over.
- 8B models fit: Q4 is ~4.5-5 GB. LLaDA-8B runs via the diffuse-cpp GGUF fork (community conversion with the diffusion decode loop). EAGLE-3's reference stacks target newer GPUs; B2 may need llama.cpp spec-decode or an older vLLM. Budget a day of friction for B2.
- The TPU is excluded from everything: the entire stack (vLLM, SGLang, llama.cpp, EAGLE, Fast-dLLM, diffuse-cpp) is CUDA, and porting decode loops to JAX/XLA buys nothing even for free. This project is PyTorch/CUDA.

## Pilot experiment L1v2: looped mid-block (revised 2026-11-11)

User decision on record: if D1 passes, the looped + verification direction is the follow-up, no further debate.

Replaces the 2026-10-30 L1 section, including the stale claim that L1 joins benchmark week at zero cost. L1v2 is blocked behind gates G0-G3 and training-pipeline readiness.

The 2026-10-30 L1 tied loop iterations to denoising steps and used equal TPS as the kill bar. Looping inside a denoising step adds work to each pass, so TPS falls by construction and equal TPS is the wrong bar. L1v2 keeps the twin-experiment structure and the "one clean negative kills only looping" discipline, but changes the treatment, the metric, and the prerequisites.

**Mechanism (treatment).** Loop a small early-middle mid-block inside each denoising step (LoopMDM-style, PAPER arXiv 2605.26106: 170M MDM, 12 layers, two early-middle layers looped 1-2x inside each denoising step, up to 3.3x fewer training FLOPs at matched performance; same broad arrangement class in 2610.04198 (adaptive/token-wise recurrence, 1.7B/8B) and 2610.30066 (selective looping of a few early layers), PAPER, verified 2026-11-11): a head of unshared layers, a mid-block of 2 layers looped S times, then a tail. During the S inner iterations the denoising-step input and context stay fixed; the step advances only after iteration S. Denoising step count T and inner loop count S are independent. Do not tie loop iterations to T. Control arm A: unlooped baseline (S=1), same architecture family and same training pipeline. Treatment arm B: 2 layers of our 16 looped inside each denoising step (our design choice, not a paper configuration), fixed S_train chosen before the run (default 4), S_decode swept {1,2,4}.

**Hypothesis.** At matched total training FLOPs and matched or reported inference FLOPs, the looped-mid-block twin beats the unlooped twin on validation NLL, with the advantage at least as large at 8 decode steps as at 16. This is a training-efficiency and quality-per-compute hypothesis, not a free-TPS hypothesis.

**Prerequisites (hard gates, all must pass before training L1v2 twins):**
1. **G0 Harness calibration (MEASURED instrument check):** run `HuggingFaceTB/SmolLM2-1.7B` base once on HumanEval-164 through our harness with a frozen, prewritten prompt template and execution script. Do not tune prompts, temperature, commit mode, thresholds, or parsing on HumanEval results. Record the score only as harness calibration. Pass: within +/- 6 points of 22.6 (SmolLM2-1.7B base, HumanEval, PAPER arXiv 2502.02737 Table 4; confirm the exact HumanEval version/protocol from the paper when running G0). HumanEval remains sealed for all D1 model decisions.
2. **G1 Sampler regression:** patched decode backend at temperature 0.0 reproduces a frozen pre-patch smoldlm baseline with >= 99% token agreement, allowing fp32-softmax near-tie flips. The legacy `results_t4.yaml` stores only a text preview and cannot prove this gate; re-capture the baseline from the pre-patch commit with full token IDs. Current result files record full `sample_token_ids` plus pass/fill/fallback accounting for future regressions.
3. **G2 Sampler and world check:** decode supports explicit `temperature` and `commit` (`topk` and confidence), fp32 softmax, and logs `tokens_emitted`, `passes`, `blocks`, `tokens_per_pass`, per-step fill counts, fallback commits, and `stop_reason`/EOS stats for code eval runs. Run the smoldlm author's own decoder as an upper bound, run a temperature/commit sweep, and run a teacher-forced likelihood check (real vs line-shuffled code). Reject any decode-policy cell if fills collapse to about 1-token fallback behavior without explanation. If the author decoder scores above 0 and our loop scores 0, fix the loop before training. If checks show no usable code signal (World A), drop MBPP pass@1 as a pilot decision endpoint and use validation NLL and gPPL only.
4. **G3 Recipe and pipeline freeze:** week-2 compute probe is measured; train/checkpoint/resume/eval works on a base D1 smoke pretrain (100-200M tokens). Decode policy (temperature, commit, steps, block, confidence), train/infer spec (objective, mask schedule, S_train unroll, and whether backward is unrolled through all S_train mid-block passes or truncated BPTT is explicitly chosen and logged before training), and data prep are frozen before twin training starts.

**Data and reproducibility freeze (required before G3 is closed):** log tokenizer id/hash/config, corpus manifest/hash, mix percentages, packing/max-length, train/val split, sample order/seed schedule, and use identical packed shards for twins. Run decontamination against HumanEval and MBPP with at least n-gram checks and one stronger method (substring/docstring match or embedding similarity). Report counts removed and residual. A known MBPP leak invalidates MBPP pass@1 for pilot decisions. Every run must log git commit/config hash, decode-policy hash, prompt template id/hash, seeds, library versions, GPU type, param counts, tokens seen, and train FLOPs.

**MBPP template prerequisite (UNCONFIRMED in repo until done):** copy the exact few-shot/task template from arXiv 2108.07732 into the repo before the first MBPP run, freeze it, and record its text/hash in every results YAML. The 500-problem test split is confirmed (PAPER arXiv 2108.07732); which appendix holds the copyable template text is UNCONFIRMED until read from the paper itself. Do not invent or change the template mid-comparison.

**Checkpoint and resume contract:** each checkpoint must include model, optimizer, scheduler, RNG states (Python, NumPy, PyTorch, CUDA), dataloader/shard cursor, global step, tokens seen, and config hash. Upload atomically and verify integrity. If a resume cannot prove equal effective tokens/steps/config for a twin comparison, mark that comparison invalid/inconclusive.

**Setup.** Two ~100M twins, same data, same optimizer family, same training plan, evaluated at T=16 and T=8 decode steps with the frozen decode policy: (A) unlooped control, S=1. (B) looped mid-block, 2 layers looped inside each denoising step, fixed S_train chosen before the run (default 4), S_decode swept {1,2,4}. Pilot budget: 500M tokens per arm (feasible pilot on free T4/Kaggle, ESTIMATE). 1B tokens per arm is allowed only if the week-2 probe and pipeline stability justify it. Do not run 3-5B tokens per arm as the first L1v2 pilot on free tiers (128-556 T4-h class, 4-19 weeks wall clock at 30 GPU-h/week, ESTIMATE). Use two seeds if affordable; a single-seed run must be labeled "single-seed, low confidence" and requires larger margins (below).

**Compute-matched definitions (implementable).** Let `L_total=16`, `L_loop=2`, `L_fixed=14`, `F_block` the FLOPs of one full forward pass over the scored block context, `F_fixed = F_block*(L_fixed/L_total)`, and `F_mid = F_block*(L_loop/L_total)`. Then inference FLOPs per block are `F_infer(T,S) = T*(F_fixed + S*F_mid) = T*F_block*(0.875 + 0.125*S)`. Report all grid cells and compare at equal T and at the closest available `F_infer` pair; state the `F_infer` ratio. A claim of matched-inference-compute win requires ratio within +/- 10%. If no +/- 10% pair exists on the grid, label the comparison "unmatched inference compute" and do not use it as PASS. An interpolated estimate may be reported only as secondary and clearly labeled ESTIMATE.

Training comparison is at matched total training FLOPs. Count one forward through fixed layers plus `S_train` mid-block passes; count backward as 2x forward and unrolled through all `S_train` mid-block passes unless truncated BPTT is explicitly chosen and logged before training. Log `train_flops_per_token_A`, `train_flops_per_token_B`, `tokens_seen_A`, `tokens_seen_B`, and require `F_train_total` equal within +/- 10% with the ratio reported. "Equal TPS" and "equal denoising steps" are reported but are not pass conditions. TPS is an outcome metric, not the kill bar.

**Metrics.** Every results row must carry: arm, seed, `N_unique`, `S_train`, `S_decode`, `T`, block_size, temperature, commit, confidence, tokens_emitted, passes, blocks, `tokens_per_pass`, median_tps, min_tps, max_tps, val_nll, genppl, genppl_ratio, genppl_ci95, mbpp_n, mbpp_pass1, mbpp_seeds, soup, rep4, parse_ok, mask_frac, entropy_H, `stop_reason`/EOS stats for code eval runs, prompt template id/hash, decode-policy hash, git commit/config hash, and per-seed raw outputs or paths. A row missing `tokens_emitted` or `passes` is not a result (MEASURED failure mode in `results_t4.yaml`). In G1/G2 debug runs and at least sampled pilot evals, log per-step fill counts and fallback commits by block.

- **Validation NLL (primary):** teacher-forced, on held-out validation shard, same tokenizer/mask policy as training, at matched `F_train_total`.
- **Generative PPL ratio (secondary):** `gPPL(model)/gPPL(ground-truth continuation)` scored by `gpt2-large` fp16 on 512 prompts x 64 in / 128 out from held-out validation data, bootstrap 95% CI. Double to 1024 samples if CI half-width > 5%. Ground-truth sanity: gPPL < 30. Treat gPPL ratio deltas < 3% as noise unless consistent across decode steps and seeds (ESTIMATE floor, tighten after measurement).
- **MBPP pass@1 (tertiary):** `google-research-datasets/mbpp` test split (500 problems, PAPER arXiv 2108.07732), n=1 per problem, 3 seeds reported as median/min/max, frozen template text/hash recorded in YAML, 256 max new tokens. Binomial noise floor: < 2.0 points absolute is not a win unless it holds across all 3 seeds and is supported by NLL/gPPL (DERIVED). Likely underpowered at 500M tokens; never the sole kill signal.
- **TPS (outcome only):** median of 5, batch 1, 1k in / 1k out, reported with tokens_emitted, passes, blocks, `tokens_per_pass`, min/max TPS. Comparisons require equal-task denominators.
- **Diagnostics:** per-inner-iteration top-1 agreement / NLL delta at decode time (to detect "decided in iteration 1", PAPER arXiv 2610.18127 risk) and degeneracy metrics (`soup`, `rep4`, `parse_ok`, `mask_frac`, `entropy_H`).

**Kill criterion (compute-matched).**
- **PASS (keep looping):** at 8 decode steps, arm B beats arm A on validation NLL at matched `F_train_total` (within +/- 10%, ratio reported). Report all grid cells and compare at equal T and at the closest available `F_infer` pair; a matched-inference-compute win claim requires `F_infer` ratio within +/- 10%, and comparisons with no +/- 10% pair are labeled "unmatched inference compute" and do not count as PASS. The NLL win must hold in both seeds, or by >= 1.0% in a single-seed run with consistent gPPL-ratio direction. Convergence diagnostics must show S>1 contributing (not iteration-1-only). At 16 steps B must be no worse than A at matched compute, and the B-vs-A gap at 8 steps should be >= the gap at 16 steps to support the few-step hypothesis.
- **FAIL (drop looping from v1):** no validation NLL win at matched `F_train_total` at 8 steps (delta < 0.5% single-seed, or split across seeds), or any claimed win exists only at higher inference FLOPs with no matched-compute win.
- **INCONCLUSIVE (no decision):** divergence, harness/pipeline bug, gate failure, or any protocol violation. Protocol violations include unequal effective tokens/steps when equality is required, mismatched decode settings, missing decontamination audit, failed/unverifiable resume state, or missing required log fields. Fix and rerun once at pilot scale per root-cause class; do not scale up. A 16-step-only win is not sufficient. Equal-TPS is never a pass condition.

**Accepted risks and non-goals.** Negative evidence exists (PAPER arXiv 2610.18127: 76-89% of tokens settled in iteration 1 in 6/8 looped **AR** LMs at 1-3.5B, not diffusion, so transfer to us is a risk not a fact; 2604.07822: overthinking at excessive recurrence, e.g. no OOD gain past ~15 iterations in their dynamic-recurrence setting only, plus three-stage grokking in small transformers trained from scratch), but positive evidence is closer to this setting (LoopMDM 170M MDM, up to 3.3x fewer training FLOPs at matched FLOPs, PAPER arXiv 2605.26106). No learned exit gate, no adaptive steps, and no verification-during-refinement in L1v2. L1v2 does not change the product TPS claim except by reporting measured TPS; any TPS drop from looping inside the step is an honest cost.

**Verification skill (deferred).** The deferred verify-during-refinement idea is not stacked onto L1v2. A cheap training-free confidence-remask diagnostic may run separately, at matched NFE, because confidence is a same-forward-pass signal (CoDR-style, training-free). PRISM (PAPER arXiv 2510.01384) is separate evidence: a fine-tuned adapter/remasking method, not training-free, reporting 170M MDM results. The learned verification/adaptive-steps mechanism remains its own experiment after L1v2 and should be dropped if remasking shows no usable signal at 100M.

## Risks

- Benchmark contamination. HumanEval is the guardrail metric and RLVR rewards "tests pass." If HumanEval problems or solutions leak into pretrain, distill, SFT, or RLVR data, pass@1 becomes meaningless. Mitigations: hold HumanEval out of all training data entirely; use other coding tests (MBPP, generated problems) for RLVR reward; run n-gram overlap checks between training corpus and HumanEval before training; touch HumanEval only at final eval.
- Code-biased pretrain. A code-heavy mix can make the base model spit code even for normal chat prompts ("hello" -> code). Mitigations: keep general text (FineWeb-Edu, books) in the pretrain mix, not just code; rely on SFT to teach chat/instruction behavior; eval on non-code prompts during training. First-time pretraining: start with a smaller pilot run to sanity-check output behavior before full budget.
- Few-step quality cliff. Generative PPL degrades steeply below ~8 steps (15.3 at 4 steps vs 6.4 at 32 in the fixed-state paper). Steps are the speed-quality dial; the spec's 8-16 range is a bet, not a fact.
- Long-context recall. The fixed-state paper's passkey scores are weak at 16k for all mixers (undertrained at 1024 context, but unproven). Our 3:1 ratio exists for this; watch it.
- Adaptive steps per block (more steps for hard code, fewer for boilerplate) has no paper making it the headline mechanism. Either a trap or our edge. AdaBlock-dLLM does adaptive block size, Prophet does early stopping; steps-per-difficulty is unclaimed.
- Diffusion RLVR immaturity could block the competence half. If it stalls, fallback is distillation-only with the competence bar lowered for v1.
- Diffusion + tool calling. Interactive RL (slime-style) needs multi-turn tool use, which diffusion (block generation) handles poorly vs AR. v1 stays single-shot code RL: model generates code, sandbox runs it, reward from execution. Defer interactive/agentic RL to v2.
- First-time pretrainer. Kriday has never pretrained before. Risk of wasted compute and bad decisions. Mitigations: pilot runs on small budgets first; read and understand each stage before running; no blind "vibe-pretraining."
- Vibe-pretraining risk. Ad-hoc pretraining (random data mix, no evals, no ablations) produces a model nobody understands. Mitigations: every run has a written hypothesis, a metric, and a kill criterion before it starts; results logged; no "just try it" runs on full budget.
- No tool-calling integrated training. v1 does not train tool use at all. If the product later needs agents, that's a separate capability, not bolted onto this model.
- Process gate. Kriday must understand every part of the puzzle before work continues. No week or stage proceeds until explained and agreed. If unclear, stop and explain in simple terms before acting.

## Pretrain data (decided 2026-11-01)

Three buckets, user-approved: general web = FineWeb-Edu (`sample-10BT` slice); code = The Stack Dedup, Python subset (mined); math = FineMath or OpenWebMath. Mix for the ~3-5B token run: ~50% code / ~35% web / ~15% math (web stays a third — the cure for code-biased pretrain). Decontamination (n-gram check vs HumanEval) runs inside the prep script on everything. SFT data (Magicoder class) is recipe step 2, separate from this. Tokenizer: train our own 32k BPE on a corpus sample (decided 2026-11-06: train our own). Workflow: ~100MB samples locally to build/test the prep script, full downloads on Colab/Kaggle where training happens.

## Deliverables

- B0/B1/B2 benchmark harness and results (week 1)
- Pretrained 100M checkpoint (GDN hybrid, block-diffusion objective)
- Custom decode loop (confidence-threshold unmasking, block-level early stop at 0.97, prompt cache, block overlap, block scheduler with tunable steps)
- TPS + pass@1 report against B0/B1/B2
