# EdgeFast-Diff 100M: block-diffusion hybrid spec (v4)

> **PROCESS APPROVED: TASKS AND FEEDBACK FIRST.** Kriday, 2026-11-17. Build verified Python tasks, prove the trainer, then compare the data recipes below. This approves the process; hardware readiness and model quality still require measurements.

Revision 2026-11-17 supersedes the earlier ban on pretraining data experiments and int4 QAT from the start. The next experiment changes task data only. Looping, quantization, longer context, adaptive decoding and agentic RL remain separate later experiments.

Revised 2026-11-01: target size retargeted from 1B to 100M (user call). With zero budget, the 100M is the real, trainable model — not just a pilot. A 1B scale-up stays future work if compute appears. Supersedes v2 of this spec (1B target) and `edgefast-1.5b.md` (AR edge model, kept for reference). Written 2026-10-28 after the hybrid-attention and diffusion-decoding research passes.

## Goal

Maximum single-stream TPS at useful quality. During development, track unseen task-family correctness and per-bucket validation loss. HumanEval stays sealed for final D1 evaluation; the one fixed G0 baseline calibration below is the exception. Report speed and quality together under the same decode policy.

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
| Attention layers (4) | Full attention, 5 query heads, 1 KV head, head dim 128, NoPE | 3:1 ratio, uniformly interleaved at blocks 4, 8, 12, 16. One KV head gives valid multi-query attention at d_model 640; the previous 5-query/2-KV configuration violated standard GQA divisibility. Recount parameters and verify the actual implementation before freezing. |
| Placement | Uniform interleave | Waleffe et al.: no ordering beats evenly dispersed. |
| FFN | SwiGLU, 8/3 expansion (1728, 64-multiple) | Standard. |
| Norm | RMSNorm, pre-norm | Standard. |
| Vocab / embeddings | 32k, tied | At 100M the table is a bigger share of the budget (~20M of ~105M); 32k tied keeps it affordable while covering code tokens. |
| Train context | 1024 initially | Matches the frozen smoke corpus. Longer context, up to the previous 32k target, requires a separate measured extension stage. |
| Precision | Floating-point baseline first | On T4, validate FP16 mixed precision with gradient scaling and FP32 computation where recurrence or loss stability requires it. Int4 QAT is deferred until the baseline works. |

## Decoding

Block-diffusion in the BD3-LM / fixed-state style (arXiv:2503.09573, arXiv:2609.11998). Freeze one working policy for the task-data comparison. Early stopping, block overlap and adaptive decoding are disabled for that comparison; the mechanisms below are later candidates.

- Block size 32 masked slots. LLaDA and the fixed-state paper both converge on 32.
- 8-16 denoising steps per block, so ~2-4 tokens revealed per forward pass. NOT LLaDA's 512-step config, which is 1 token per pass and slower than AR. The LLaDA lesson: speed lives in the decode policy, not the architecture.
- Unmasking: parallel with confidence threshold (0.9, Fast-dLLM style) rather than strict low-confidence sequential ordering. Measured TPS wins live here.
- Early stopping (user idea, 2026-10-30): a separate block-level confidence score, checked once per round. Above 0.97, fill every remaining slot and finish the block, skipping the unused rounds. The two thresholds coexist cleanly: 0.9 governs individual tokens within a round, 0.97 governs whether the block continues at all. Block score defaults to the MIN confidence across remaining masked slots, not the mean, because a 97% average can hide one garbage slot and commit it. (Adjacent published work: Prophet early-stopping; block-score definition is ours.)
- Block overlap is deferred. Validate prefix dependencies, output quality and measured scheduling benefit before enabling it.
- Deferred knobs, do not build yet: (a) adaptive rounds by difficulty — post-D1, fused with the L1 verification skill; difficulty is hard to define and can flip mid-generation, so it needs the model's own judgment rather than a static classifier (user call, 2026-10-30). (b) Adaptive block size (AdaBlock-dLLM) — noted as the trickier cousin of early stopping.
- Few-round training is a separate speed-focused post-training experiment after useful correctness is established. Ordinary instruction tuning or test-filtered self-training does not establish good quality at 4-8 rounds.
- Prompt encoded once and cached. Attention layers cross-attend the cached clean prefix; GDN layers carry a constant-size recurrent state across blocks. Training computes the clean prefix the same way inference does, so the cache is exact, not approximated.

Historical quantized bandwidth estimate, not a prediction for the first floating-point run: 100M int4 is roughly 0.06 GB per pass, giving about 5,000 passes/s at 320 GB/s before overhead. Actual TPS, useful tokens per pass and quality require measurement on the chosen backend. No training budget or pass condition is derived from this ceiling.

## Training recipe: tasks and feedback first (approved 2026-11-17)

No paid teacher generation. Use deterministic task generators, human-reviewed references and suitable existing public examples with recorded provenance. Public instruction tuning stays; test-filtered self-training follows only when it produces useful verified solutions. Full repository agents and policy-gradient RL remain deferred.

### 1. Build and audit the task environment

Start with short, single-function Python tasks covering strings, lists, dictionaries, counting and small repairs. Each task records its description, starter code, reference solution, generator version, seed, skill family, provenance and verifier version. Keep complete task-and-solution examples within the initial context limit.

- Require the reference solution to pass normal, boundary and independently authored hidden tests. Require plausible incorrect implementations to fail. For repairs, verify both fail-to-pass cases and preserved pass-to-pass behavior.
- Example: deduplicate a list while preserving order. Include empty input and repeated values; reject solutions that sort or lose order.
- Run candidates in isolated workers with enforced time and memory limits. Keep grading code and hidden tests outside candidate control. The existing in-process HumanEval executor is not the training sandbox.
- Split by generator template, underlying solution structure and perturbation family before producing variants. Renamed variables or changed constants do not create independent evaluation tasks. Maintain separate training, development and sealed evaluation families, plus independently sourced transfer tasks.
- Apply the existing HumanEval/MBPP decontamination to references, prompts and accepted model outputs. Neither suite supplies training rewards. Review ambiguous requirements, broken verifiers and repeated failure patterns before adding more tasks.

### 2. Prove the trainer on the frozen corpus

Use the existing training-v1 bundle, 99,983,360 training tokens, for a plumbing run. Keep it unchanged. Use an unlooped floating-point model, block size 32 and context 1024. Follow one published block-diffusion objective; freeze its corruption distribution, loss weighting and normalization, optimizer, learning-rate schedule, effective batch size, token budget and decode policy before the run.

Implement causal clean history, bidirectional visibility within the active noisy block, and isolation from future blocks and other packed documents. Apply equivalent boundaries to recurrent state; an all-one padding mask is insufficient. Verify cached generation against the reference path. Test interruption/resume against an uninterrupted run, restoring the checkpoint contract below plus gradient-scaler state. Passing this run establishes plumbing, not competitive model quality.

### 3. Run one controlled task-data pilot

Hypothesis: early exposure to verified task-and-solution examples improves unseen Python-task correctness at matched training compute without unacceptable general-text regression.

| Arm | Raw code | Verified Python task/solution examples | Web | Math |
|---|---:|---:|---:|---:|
| A: control | 50% | 0% | 35% | 15% |
| B: task data | 40% | 10% | 35% | 15% |

Fractions are by tokens. The 10% share is an experimental starting point. Task examples remain a separately indexed subtype inside the 50% code bucket. Store prompt/solution boundaries and task-family metadata alongside shards; do not imply the current three-bucket builder already implements this treatment.

Start both arms from the same fresh initialization, using the same tokenizer, architecture, corruption/loss, optimizer schedule, effective batch, total tokens and decode policy. Keep shared source selection/order matched where possible and log the deliberate substitutions. Freeze unique pools, repetition schedules, seeds and hashes. Deduplicating the unique corpus does not prohibit documented multi-epoch training; report unique tokens and total exposure separately.

Use a measured hardware probe to freeze an affordable budget before starting either arm, capped initially at 500M tokens per arm. This cap is a planning limit, not a measured feasibility claim. Require total training FLOPs within +/- 10% and report the ratio, GPU-hours and all evaluation/generation costs. Use two seeds if affordable; label a single-seed result low confidence. Do not combine this pilot with L1v2, QAT, context extension or decode-policy tuning.

Before the pair starts, freeze the evaluation tasks, scoring, minimum useful gain, acceptable per-bucket regression limits and uncertainty method using development-only measurements. Promote B only if unseen-family correctness improves beyond that criterion, independent transfer tasks support the direction, and general-text/code/math validation stays within the frozen limits. Otherwise retain A or report inconclusive and diagnose. HumanEval remains sealed. This pilot uses its own task-data gate; it does not replace the L1v2 looping criteria.

### 4. Instruction tuning and feedback

Complete the selected full pretraining run within its measured, frozen budget, then instruction-tune on clean public code/instruction examples and verified references. Keep the initial SFT budget to tens of millions of tokens and freeze its exact size. Record data reuse, scrub irrelevant tool traces and formatting filler, keep a general-language slice, and evaluate non-code behavior. Preserve complete prompts and solutions during packing. Freeze whether prompts are corrupted and which response tokens contribute loss.

Then measure solve rates by skill family and difficulty on training tasks. Start test-filtered self-training only when a predeclared minimum yield makes generation worthwhile. Generate with a frozen checkpoint, independently verify and deduplicate successes, record checkpoint IDs, then fine-tune on the accepted batch. Keep development and sealed evaluation families out of sampling and feedback. If yield is poor, improve references or task suitability rather than spending more on repeated failures.

Adjust difficulty using observed solve rates while retaining skill diversity. Early supervised exposure to solved hard examples and selecting solvable rollout tasks are separate decisions; do not assume a universal easy-to-hard curriculum. Track novel verified successes per GPU-hour including rejected attempts, acceptance rate, timeout/invalid-code rate, repetition and per-family correctness. Test execution has CPU and wall-clock costs even without a paid judge.

### Evidence and limits

- [DeepSeek V4.1, section 5.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf): task/environment/verifier construction, difficulty calibration and quality re-auditing.
- [Qwen3-Coder-Next](https://arxiv.org/html/2603.00729v1): executable tasks, bug/fix verification and complete-context packing.
- [Falcon-H1-Tiny](https://tiiuae-tiny-h1-blogpost.hf.space/?__theme=system): early target-data exposure and controlled reuse at 90M. Its reported 800B-token schedule is far larger than ours and uses AR training; benefit at our diffusion scale/budget is unproven.
- [Curriculum study](https://arxiv.org/html/2603.24202v1): difficulty and diversity affect RL learning at 8B/32B. Transfer to 100M and test-filtered self-training remains an experiment.

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
- The 100M model remains the product target. Kaggle T4 training is a candidate hardware plan pending the compatibility and compute probe below. A 1B scale-up remains future work.
- Current sequence: build/audit tasks and complete benchmark calibration; prove the exact training backend and resume behavior; run the controlled task-data pilot; freeze the full-run recipe and affordable token budget. Task generation and verification can proceed on CPU while hardware readiness is unresolved.

**Training hardware gate, added 2026-11-17:** current [Triton requirements](https://github.com/triton-lang/triton) list NVIDIA compute capability 8.0+, while T4 is 7.5. Do not assume the modern GDN stack runs on T4. Pin the exact PyTorch, CUDA, kernel and optimizer versions; verify real forward/backward/optimizer steps, finite gradients, reference agreement and cache/state behavior on the intended GPU. Measure steady-state training tokens/s, peak memory, evaluation time and checkpoint overhead before setting budgets. If no correct, sufficiently fast T4 backend exists, report the block and revise hardware or architecture explicitly. CPU fallbacks do not establish GPU feasibility.

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

**Checkpoint and resume contract:** each checkpoint must include model, optimizer, scheduler, gradient-scaler state when used, RNG states (Python, NumPy, PyTorch, CUDA), dataloader/shard cursor, global step, tokens seen, and config hash. Upload atomically and verify integrity. Verify interrupted/resumed training against an uninterrupted reference, including sample order and corruption RNG, under frozen numeric tolerances. If a resume cannot prove equal effective tokens/steps/config for a twin comparison, mark that comparison invalid/inconclusive.

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

- Benchmark contamination. Keep HumanEval and MBPP out of all training examples, generated references, self-training outputs and rewards. Use separate verified task families for training feedback and development, with sealed families for evaluation. Apply the existing overlap checks at every data-producing stage; zero detected overlap does not prove semantic independence. HumanEval is final-only for D1, apart from the fixed G0 baseline calibration.
- Code-biased pretrain. A code-heavy mix can make the base model spit code even for normal chat prompts ("hello" -> code). Mitigations: keep general text (FineWeb-Edu, books) in the pretrain mix, not just code; rely on SFT to teach chat/instruction behavior; eval on non-code prompts during training. First-time pretraining: start with a smaller pilot run to sanity-check output behavior before full budget.
- Few-step quality cliff. Generative PPL degrades steeply below ~8 steps (15.3 at 4 steps vs 6.4 at 32 in the fixed-state paper). Steps are the speed-quality dial; the spec's 8-16 range is a bet, not a fact.
- Long-context recall. The fixed-state paper's passkey scores are weak at 16k for all mixers (undertrained at 1024 context, but unproven). Our 3:1 ratio exists for this; watch it.
- Adaptive steps per block (more steps for hard code, fewer for boilerplate) has no paper making it the headline mechanism. Either a trap or our edge. AdaBlock-dLLM does adaptive block size, Prophet does early stopping; steps-per-difficulty is unclaimed.
- Feedback yield. A weak model may produce too few correct, novel solutions to justify self-training. Measure accepted solutions per GPU-hour and retain supervised references if yield is inadequate. Test-filtered self-training has no established speed benefit by itself.
- Diffusion + tool calling. v1 trains single-function code generation and repairs with external execution-based verification. It does not introduce interactive tool-use RL. Repository agents and policy-gradient RL remain deferred.
- First-time pretrainer. Kriday has never pretrained before. Risk of wasted compute and bad decisions. Mitigations: pilot runs on small budgets first; read and understand each stage before running; no blind "vibe-pretraining."
- Vibe-pretraining risk. Ad-hoc pretraining (random data mix, no evals, no ablations) produces a model nobody understands. Mitigations: every run has a written hypothesis, a metric, and a kill criterion before it starts; results logged; no "just try it" runs on full budget.
- No tool-calling integrated training. v1 does not train tool use at all. If the product later needs agents, that's a separate capability, not bolted onto this model.
- Process gate. Kriday approved this staged process on 2026-11-17. Stage progression depends on the recorded measurements and prerequisites; process approval does not mark a gate passed. Explain material changes to the agreed scope, architecture or budget before adopting them.

## Pretrain data (updated 2026-11-17)

Keep the frozen training-v1 corpus and custom 32k byte-level BPE tokenizer for the plumbing run. Pinned sources are The Stack Dedup Python, FineWeb-Edu sample-10BT and FineMath finemath-4plus; preparation and audits are documented in [TRAINING_DATA.md](../docs/TRAINING_DATA.md).

The approved pilot compares the current 50/35/15 token mix with the 40/10/35/15 raw-code/task-code/web/math treatment above. Keep task lineage and splits explicit. All newly generated examples need provenance, deduplication and benchmark exclusion. Existing artifacts remain unchanged; build new versioned bundles for the pilot.

The full pretraining budget is not frozen. Earlier 2-10B and 3-5B ranges were estimates. Choose the budget and mixture after the hardware probe and task-data result, then freeze them before scaling. Public instruction tuning remains a later stage even when verified task/solution examples appear during pretraining.

## Deliverables

- B0/B1/B2 benchmark harness and results (week 1)
- Pretrained 100M checkpoint (GDN hybrid, block-diffusion objective)
- Audited Python task generator, independent verifier, versioned family splits and isolated execution workers
- Matched-compute task-data pilot report, including transfer correctness, per-bucket loss, unique data/reuse and total cost
- Custom decode loop (confidence-threshold unmasking, block-level early stop at 0.97, prompt cache, block overlap, block scheduler with tunable steps)
- TPS + pass@1 report against B0/B1/B2
