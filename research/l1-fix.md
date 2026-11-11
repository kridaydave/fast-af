# L1 pilot fix: looped core, rewritten to be measurable

Author: design sub-agent (iceman-free). Date: 2026-11-11.
Scope: rewrite of the "Pilot experiment L1: looped core" section in `spec/edgefast-diff-100m.md` for EdgeFast-Diff 100M (D1), zero budget, free T4/Kaggle only. No edits to existing files. No invented numbers.

Evidence base (primary/verified, read this session): `spec/edgefast-diff-100m.md` (lines 110-127, L1 as specified), `research/hillclimb-quality-ernie.md`, `research/quality-fixplan-ernie.md`, `bench/backends.py` (MEASURED: `CustomBackend.generate` does `probs.max`, temperature 0, fp16 softmax, uncached; no pass counter), `results_t4.yaml` (MEASURED: smoldlm-144m speed-only, `humaneval: false`, degenerate sample text, 349.979 tok/s median; AR pass@1 0.0-0.10 are greedy AR runs), `bench/config.yaml` (MEASURED: smoldlm row `humaneval: false`, block 32, 8 steps, confidence 0.9), `bench/BENCHMARK_WEEK.md` (MEASURED: Tier 4 L1 deferred; smollm2-135m emitted 307 tokens, lfm2.5-350m 12). Papers cited are carried from the verified ledgers in `hillclimb-quality-ernie.md` §6 and `quality-fixplan-ernie.md` §7 (fetched/opened 2026-11-11 in that pass): arXiv 2605.26106 (LoopMDM, NeurIPS 2026, 170M MDM), 2610.04198 (ALoDLM), 2610.30066 (Looped UDLM), 2609.40305 (Looped-DiT, vision), 2610.18127 (monitoring looped LMs), 2604.07822 (Loop/Think/Generalize), 2609.11998 (fixed-state, Table 5), 2609.33947 (sampler sharpening, 34-170M), 2508.19529 (blockwise-aligned SFT, LLaDA-8B LoRA), 2412.02674 (GV gap), 2510.01384 (PRISM 170M), CoDR (EMNLP 2026 Findings), 2610.17590 (compute gap by block size), 2502.02737 (SmolLM2), 2108.07732 (MBPP). Where a paper's scope does not reach 100M/free-T4 I say so. UNCONFIRMED items are labeled.

---

## 1. Verdict on the cut: shrink and re-scope L1, do not run it as specified

**Verdict: L1 survives, but only as a different experiment. Run a shrunk, compute-matched, loop-inside-step pilot (L1v2) after the instrument and pretrain recipe are frozen. The spec's L1 as written is cut.**

### Fixplan arguments, addressed head-on

| Fixplan argument (`quality-fixplan-ernie.md` §4 item 4, echoed in `hillclimb-quality-ernie.md` §2 A4 and §5 cut 1-2) | Response | Consequence |
|---|---|---|
| "Loop iterations ARE the denoising steps" + "TPS untouched by construction" is a configuration nobody in the literature uses; equal-TPS framing makes a null result unreadable. | **Accepted.** Every positive looped-diffusion result loops *inside* each denoising step, not across steps: LoopMDM (PAPER, arXiv 2605.26106, 170M MDM) loops a 2-layer early-middle mid-block S times inside each step (input fixed during the loop, changes between steps) and reports up to 3.3x fewer training FLOPs at equal NLL at matched total FLOPs; ALoDLM (PAPER, arXiv 2610.04198, 1.7B/8B) refines in latent space inside each step with K<=4 passes and a learned exit gate; Looped UDLM (PAPER, arXiv 2610.30066) loops early layers within each step with 2.8x/3.3x fewer FLOPs at matched NLL; Looped-DiT (PAPER, arXiv 2609.40305, vision, 260M-1.7B, analogy only) explicitly decouples loop depth from denoising step count. "TPS untouched" is true only for the spec's version (which the evidence argues against). Looping inside a step adds work per forward pass, so TPS must fall by construction. | The treatment arm and the success metric both change (see §2, §3). "Equal TPS" is removed as the bar. |
| L1 as specified stacks a new mechanism on an unfrozen, untrusted recipe (decoder broken, no gPPL/MBPP harness for diffusion, pretrain not probed). A null would be unreadable. | **Accepted.** MEASURED: `CustomBackend` is greedy argmax on fp16 softmax with no temperature/commit control (`bench/backends.py:165-210`), smoldlm sample is degenerate (`results_t4.yaml:113-114`), smoldlm has no pass@1 row (`bench/config.yaml:68`, `results_t4.yaml:103-121`), and no generative-PPL eval exists in repo. The fixplan's instrument-first sequence (Parts 1-2) is a hard prerequisite. | L1v2 cannot start until gates G0-G3 pass (§6). |
| Pretrain costs ~64 T4-h (2-3 weeks Kaggle wall clock) and the recipe must freeze first; looping adds risk/cost before the base recipe is proven. | **Partially accepted.** The 64 T4-h figure is an ESTIMATE (`quality-fixplan-ernie.md` §6: 3B tokens at 13k tok/s, A3; plausible 5k-13k tok/s, to be replaced by the week-2 probe). Freezing the base recipe first is mandatory. But the fixplan's own argument for looping is that 3.3x training-FLOP efficiency (PAPER, LoopMDM 170M, matched total FLOPs) is directly 3.3x more tokens on zero budget, i.e. a *training-efficiency* lever, not a luxury. That only helps if the pretrain pipeline exists and is reusable. | L1v2 runs as a **500M-token de-risk pilot** on the proven pipeline, not as a full 3B-token twin pair. Full-scale looping is attempted only if the 500M-token pilot clears its kill criterion and the pipeline can be re-run cheaply. |
| "76-89% of tokens decided in loop iteration 1; Ouro-Thinking best at iteration 1" means extra iterations may buy little. | **Accepted as a real risk, not a veto.** PAPER arXiv 2610.18127 (8 looped AR LMs, 1-3.5B, not MDM): 76-89% tokens settle on final top-1 in iteration 1 in 6/8 models; Ouro-Thinking degrades after iteration 1. PAPER arXiv 2604.07822 (small transformers, from scratch): three-stage grokking, overthinking past ~15 iterations. Counterpoint: positive evidence (LoopMDM 2605.26106) is exactly 170M MDM, closer to our setting than the AR recurrent-depth negatives. | L1v2 must include a per-iteration convergence diagnostic (top-1 agreement / NLL delta by inner iteration S) so a "mostly iteration 1" result is visible, and the kill criterion is compute-matched NLL, not a story about depth. |

**What must change for L1 to run:** (a) sampler/eval instrument fixed and calibrated, (b) a working pretrain + eval pipeline proven on a base run, (c) treatment becomes loop-inside-step, (d) metric becomes matched training/inference FLOPs and matched forward-pass budget, (e) pilot budget shrunk to 500M tokens/arm (or less if probe says so), (f) explicit seed and noise-floor reporting so a small gap is not misread.

---

## 2. Treatment arm: loop inside the step (LoopMDM style), one arm, one fallback

### Decision

**Primary arm: loop-inside-step, early-middle mid-block (LoopMDM style). Do not run the spec's "loop iterations = denoising steps" arm. Do not run both arms.**

Rationale (from evidence, not preference):
- LoopMDM (PAPER, arXiv 2605.26106, 170M MDM): head of unshared layers, mid-block of 2 layers looped S times, tail with LM head; noisy input fixed during the loop and changes only between diffusion steps. Matches same-size MDM with up to 3.3x fewer training FLOPs at equal NLL and beats deeper non-looped MDMs at comparable per-step compute. Scope: 170M, 3 corpora, matched total FLOPs. This is the closest positive evidence to D1's size and objective.
- ALoDLM/Looped UDLM (PAPER, 2610.04198, 2610.30066): same structural choice, loop inside step, at larger scale. Looping across steps (tying step count to loop count) is not the arrangement these papers converge on.
- Mechanism: if loop == step, a 4-step decode gets 4 core applications total and fights a large jump; if loop is inside the step, 4 steps × S applications each refine on a near-correct context. This is the reason the field separated the two (hillclimb §2 A4, consistent with the cited papers).
- Running both arms doubles the pilot cost on a project where one 3B pretrain is already ~64 T4-h ESTIMATE. The spec's arm tests a configuration nobody uses; spending free-tier wall clock on it is not justified.
- Negative evidence (2610.18127, 2604.07822) is real but from recurrent-depth AR, not MDM; it argues for a compute-matched kill criterion and convergence diagnostics, not for reviving the spec's arm.

### Concrete treatment definition (L1v2)

| Item | Control (A, unlooped) | Treatment (B, looped-mid-block) |
|---|---|---|
| Total params (nominal) | N = 100M target (same shape family as spec: 16 blocks, d_model 640, GDN/attn mix per spec) | N nominally equal: 14 unshared blocks + 1 mid-block of 2 transformer/GDN blocks looped, so unique params = N - 2*P_block but *architectural slots* comparable. Exact param counts must be printed and logged before training; "same data, same budget" means same token budget and matched FLOPs, not identical param count by construction. |
| Loop placement | none | early-middle 2 layers, looped **inside each denoising step** (LoopMDM arrangement). Input to the mid-block is the same for all S iterations within a step; only the mid-block hidden state evolves; the denoising step advances only after S iterations. |
| Inner iterations S | 1 (implicit) | S_train in {4, 6} (pick one before run; default 4 to limit cost). S_decode must be configurable and swept at eval: {1, 2, 4} at 8 steps and {1, 2, 4} at 16 steps. |
| Denoising steps T | 16 and 8 (eval) | same T set: 16 and 8 (eval) |
| Decode policy | fixed to the E1 winner (§5) | identical to control |
| Data / tokens / optimizer / seed protocol | identical | identical token budget and optimizer recipe; **two seeds minimum** for the arm that is cheaper to re-run, or one shared seed pair reported with variance caveat (see §4) |

**Fallback (only if the looped-mid-block implementation cannot be made correct on the pipeline before the pilot window):** run a smaller *training-efficiency* proxy instead of abandoning the question: train control to 500M tokens and treatment to 500M/S_effective-equivalent FLOPs if measurable, else declare L1v2 "blocked on implementation" with no quality claim. Do **not** fall back to the spec's loop-across-steps arm. Do **not** claim "looping doesn't work" from a blocked or mis-implemented run.

**Explicit non-goal:** adaptive/learned exit gates (ALoDLM style) are out of scope for the pilot; fixed S only.

---

## 3. Kill criterion: compute-matched, not equal-TPS

The spec's criterion ("if (b) doesn't beat (a) on quality at equal TPS at either step count, looping is dropped") is invalid for loop-inside-step because each forward pass does more work.

### 3.1 Definitions (implementable)

Let:
- `N_unique` = number of unique parameter tensors actually stored (bytes read per pass scale with this in the bandwidth view; in practice for a weight-tied/looped mid-block the unique weights are read once per pass but applied S times — see inference FLOPs below). Log `N_unique` and `N_effective_pass = N_unique` reads + compute multiplier.
- `B` = block size (32).
- `T` = denoising steps per block.
- `S` = inner mid-block iterations per denoising step (S=1 for control).
- `L_total` = total transformer blocks (16). `L_loop` = blocks in the looped mid-block (2). `L_fixed` = `L_total - L_loop` (14).
- `P` = prompt+prefix length in tokens for the scored sequence.
- `F_block` = forward FLOPs of one full non-looped pass over `(P + B)` tokens at 100M scale. Use the standard dense approximation: `F_pass_dense ≈ 2 * N_unique * (P+B)` per forward (MEASURED convention in transformer accounting; attention sequence term at seq ~1024 is small at 100M and must be included consistently if included at all — be consistent across arms).
- `F_step(A)` = FLOPs of one denoising step for arm A. Control: `F_step = F_block`. Looped: `F_step = F_fixed + S * F_mid`, where `F_fixed = F_block * (L_fixed / L_total)` and `F_mid = F_block * (L_loop / L_total)`. (DERIVED from layer count share; acceptable if both arms use the same accounting. Log actual wall-clock and `torch.profiler`/CUDA event FLOP proxies if available, but the formula is the decision rule.)
- `F_infer(T,S)` = total decode FLOPs per block = `T * F_step(S) = T * (F_fixed + S * F_mid)`. This is the **inference compute** matched in decode comparisons.
- `F_train` = total training FLOPs = (tokens seen) × (6 * `N_unique_train_equiv` accounting) with the same layer-share adjustment for the looped mid-block executed S_train times in the forward (and backward through the unrolled loop unless truncated-BPTT is explicitly chosen and logged). Use: `F_train_step ≈ 3 * F_step_forward(S_train)` per token/block position under standard 2N forward + 4N backward approximation, applied consistently.

**Compute-matched (inference, decode comparison):** arm B at `(T_b, S_b)` is compute-matched to arm A at `(T_a, S_a=1)` when `F_infer(T_b, S_b)` is within ±10% of `F_infer(T_a, S_a)`. Concretely with `L_loop=2, L_total=16`: `F_step(S) / F_block = (14/16) + S*(2/16) = 0.875 + 0.125*S`. So B at (T=8, S=4) has per-block inference FLOPs `8 * (0.875 + 0.5) * F_block = 11.0 * F_block`; A at (T=8, S=1) has `8 * F_block`; A at (T=11, S=1) would be `11 * F_block` (not an integer step in the standard grid, so use the nearest feasible grid point and report the ratio). **Therefore "equal decode steps" is not "equal compute"; comparisons must be reported both at equal T and at equal `F_infer` (interpolated or nearest-grid).**

**Compute-matched (training):** the primary pilot comparison is at **matched total training FLOPs** (LoopMDM's metric). Run each arm for the same `F_train` budget (e.g. both consume 500M tokens *at their respective per-step FLOP cost*, meaning the looped arm sees fewer optimizer steps or fewer tokens if S_train>1 — choose one and state it; preferred: **equal wall-clock-equivalent training FLOPs = equal token count × per-token-train-FLOP ratio**, so the looped arm may train on fewer tokens to hold `F_train` equal; log both tokens and `F_train`). If equal-token is run instead of equal-`F_train`, it must be labeled "equal-token, higher-compute treatment" and cannot be used to claim efficiency.

**Compute-matched (TPS/fairness reporting):** report TPS and `tokens_per_pass` for every (arm, T, S, τ, commit) cell. TPS is an outcome metric, not the kill bar. A win that costs 2x inference FLOPs is a different claim than a win at equal inference FLOPs.

### 3.2 Exact pass/fail rule at 16 and 8 steps

Pilot unit: 500M tokens/arm baseline budget (adjustable down by week-2 probe, see §7), blockwise-aligned objective if Q1 gate is cleared else classical (stated in run log), same data/shuffle/optimizer family, 2 seeds if feasible (seed A, seed B); if only 1 seed is affordable, require larger margins (below) and label result "single-seed, low confidence".

Primary endpoint: **validation NLL** (teacher-forced, on held-out val shard, same tokenizer/mask policy as training) at matched total training FLOPs. Secondary: **gPPL ratio** at 8 and 16 decode steps (per §4). Tertiary: **MBPP pass@1** (per §4, likely underpowered at 500M tokens; never the sole kill signal).

**At 8 decode steps (the product regime):**
- **PASS (keep looping):** at matched `F_train` and at matched `F_infer` (nearest grid, ratio reported), arm B beats arm A on val NLL in both seeds, OR beats by >= 1.0% in the single seed run AND shows a consistent direction on gPPL ratio (B lower/better). Additionally report the per-iteration convergence diagnostic (`hillclimb-quality-ernie.md` A4 risk, arXiv 2610.18127): fraction of positions whose top-1 token is unchanged from iteration 1 to S at decode time. A "win" driven entirely by iteration 1 (S>1 indistinguishable from S=1 on diagnostics) is treated as "no evidence of useful looping" even if NLL moved (flag, do not auto-pass).
- **FAIL (drop looping from v1 plan):** arm B does not beat arm A on val NLL at matched `F_train` and matched `F_infer` (within noise: absolute delta < 0.5% NLL in single-seed run, or split across seeds), OR B wins only at higher `F_infer` than A with no matched-compute win.
- **INCONCLUSIVE:** training diverges, pipeline bug, or metric/harness gate failed. Fix and re-run once at same scale; do not scale up.

**At 16 decode steps:**
- Same NLL rule. Secondary check: the gap B-vs-A at 8 steps should be >= gap at 16 steps (looping should help the few-step regime more; this is the mechanism claim). If B wins at 16 but is equal/worse at 8, record as "depth helps at easy decode, not at product regime" and do not claim L1's core hypothesis.

**Overall L1v2 decision:** PASS requires a pass at 8 steps (matched compute). A 16-step-only win is not sufficient. Equal-TPS is reported but never a pass condition.

---

## 4. Metrics: generative PPL, MBPP pass@1, TPS — how, on free T4, and noise floors

All metrics are meaningless until sampler and harness gates pass (§5). Every results row must carry: `arm, seed, N_unique, S_train, S_decode, T, block_size, temperature, commit, confidence, tokens_emitted, passes, blocks, tokens_per_pass, median_tps, min_tps, max_tps, val_nll, genppl, genppl_ratio, genppl_ci95, mbpp_n, mbpp_pass1, mbpp_seeds, soup, rep4, parse_ok, mask_frac, entropy_H`. A row missing `tokens_emitted` or `passes` is not a result (MEASURED failure mode: `results_t4.yaml` rows with 307/12 tokens were not comparable).

### 4.1 Generative PPL (gPPL)

- **What:** `gPPL = exp(mean token NLL under gpt2-large, fp16, on the generated continuation only, prompt excluded)`. Also report `gPPL_ratio = gPPL(model_continuation) / gPPL(ground_truth_continuation)` on the same prompts with the same judge (removes judge difficulty; absolute gPPL is not comparable across corpora/judges). Protocol follows the field standard used in the verified ledgers (PAPER: arXiv 2609.33947 and 2609.11998 score masked-diffusion samples with GPT-2 PPL; fixed-state used 64->128-token continuations).
- **Script:** `eval/genppl.py` (new file, to be built; does not exist yet, MEASURED). Must be deterministic given seed, log model/judge ids, prompt ids, and per-sample NLLs.
- **Prompts:** 512 prompts of 64 tokens from a held-out validation shard of our own pretrain mix, never trained on, never from MBPP/HumanEval. 128 generated tokens per prompt (512 x 128 = 65,536 scored tokens). For a 500M-token pilot checkpoint that is weak, also allow a smaller 256-prompt debug run labeled `debug`, but decisions use 512.
- **Judge/sanity:** `gpt2-large` (774M) fits T4 fp16 (~1.5 GB). Sanity floor: ground-truth continuation gPPL < 30 on the chosen prompt slice; if not, the prompt slice/judge setup is wrong and no model number is valid.
- **Uncertainty / noise floor:** bootstrap 95% CI over the 512 samples. If CI half-width > 5% of estimate, double to 1024 samples. **Noise floor for decisions:** treat gPPL ratio differences < 3% as noise unless consistent across both decode steps and both seeds. (3% is a conservative floor relative to the 5% CI-stability rule; ESTIMATE, must be tightened once week-1 variance is measured.) Sharpening-scale effects reported in literature are 2-5x (PAPER 2609.33947, 34-170M, different setting) so 3% is a safe "no claim" band for L1-scale deltas.
- **T4 cost:** ~3.5 min/config, ~0.06 T4-h (ESTIMATE per fixplan §2.1: 512x128 gen at ~350 tok/s + GPT-2-large forward over 65k tokens). Cheap enough to run per checkpoint.
- **Command (template, once built):**
  ```bash
  python -m eval.genppl --model <ckpt_or_repo> --backend custom \
    --prompts data/val_prompts_512x64.jsonl --n 512 --in-len 64 --out-len 128 \
    --steps 8 --temperature <tau> --commit <topk|confidence> --seed 0 \
    --judge gpt2-large --out results/genppl_<run>.yaml
  ```

### 4.2 MBPP pass@1

- **What:** pass@1, n=1 sample per problem (greedy-equivalent at fixed seed/policy; for sampled decode report seed and keep n=1 per problem per seed, aggregate across 3 seeds).
- **Dataset:** `google-research-datasets/mbpp`, test split = 500 problems (PAPER, Austin et al. 2021, arXiv 2108.07732 §2: 10 few-shot prompt / 500 test / 374 fine-tune). MEASURED in fixplan from dataset card. Few-shot block: 10 rows from `prompt` split only, never test rows.
- **Script:** `eval/mbpp.py` (new, to be built). Execution: completion + `test_list` asserts via subprocess with timeout (same discipline as `bench/humaneval.py`, but never assume a passing environment is a proof of correctness beyond the provided tests). **Prompt template: UNCONFIRMED in this repo session** (fixplan notes "must be copied from arXiv 2108.07732 Appendix A before first run"); freeze whatever template is chosen as a constant recorded in results YAML and do not change it mid-comparison.
- **Seeds/runs:** 3 seeds per config for the pilot endpoint; report median and min/max. For 500M-token weak checkpoints, 1 seed is acceptable only if labeled and paired with gPPL.
- **Noise floor:** binomial 95% interval at n=500 is about ±1.9 points at p=5%, ±2.6 at p=10%, ±4.4 at p=50% (DERIVED: 1.96*sqrt(p(1-p)/n)). At the expected 0-3% regime for 100M/3-5B tokens, a 1-2 point move is within noise across seeds; **do not call a < 2.0 point absolute MBPP gain a win unless it holds across all 3 seeds and is supported by gPPL/NLL**. At n=164 (HumanEval) intervals are wider (±3.3 at p=5%); HumanEval is sealed/final only per spec.
- **T4 cost:** 500 problems x 256 tokens / 350 tok/s = ~366 s = ~0.10 T4-h per config (ESTIMATE using MEASURED smoldlm speed; slower if model/TPS differs). 3 seeds = ~0.3 T4-h.
- **Command (template):**
  ```bash
  python -m eval.mbpp --model <ckpt_or_repo> --backend custom \
    --split test --n 500 --max-new-tokens 256 --steps 8 \
    --temperature <tau> --commit <topk|confidence> --seeds 0,1,2 \
    --out results/mbpp_<run>.yaml
  ```

### 4.3 TPS (reported, not the kill bar)

- **What:** `TPS = generated_tokens / decode_seconds`, median of 5 timed runs after 1 warmup, batch 1, 1k in / 1k out, same GPU (existing contract, `bench/config.yaml` common; MEASURED harness behavior in `bench/run.py`/`bench/backends.py`). Also report `tokens_emitted`, `passes`, `blocks`, `tokens_per_pass = tokens_emitted/passes`, min/max TPS.
- **Why not a kill criterion:** loop-inside-step increases work per pass, so TPS drops by construction; "equal TPS" would penalize the treatment for doing the thing being tested. TPS is reported to quantify the speed cost of any quality win.
- **Noise floor:** use median of 5 plus min/max (existing practice). Treat <5% TPS differences as measurement noise on free Colab T4 (ESTIMATE; session variance/thermal throttling not measured). Always report `tokens_emitted == 1024` (or explain); short generations inflate TPS (MEASURED issue: 307 and 12 token rows in `results_t4.yaml`).
- **Command:**
  ```bash
  python -m bench --config bench/config.yaml --only <model_id> --out results/speed_<run>.yaml
  ```

### 4.4 Degeneracy/validity metrics (gate metrics, reported per cell)

Computed on generated text (from `eval/metrics.py`, new): `soup` (fraction of bracket/paren/equals/semicolon tokens), `rep4` (1 - unique_4grams/total_4grams), `parse_ok` (fraction where `ast.parse` succeeds on extracted Python/code block), `mask_frac` (must be 0), `entropy_H` (mean predictive entropy at commit time). These do not replace gPPL/MBPP but they catch "World B" harness failures early (MEASURED motivation: bracket soup sample in `results_t4.yaml:113-114`).

---

## 5. Greedy-decoder problem and sampler fix required BEFORE L1 runs

**L1 inherits the current broken measurement/decode path. No L1 training or L1 quality comparison may run until the sampler and eval harness are fixed and pass gates G0-G2.**

### Evidence (MEASURED / PAPER)

- MEASURED: `bench/backends.py:165-210` does `probs = torch.softmax(logits[0, -self.block_size:], dim=-1)` on fp16 logits and `conf, tok = probs.max(dim=-1)` (greedy argmax, τ=0), commits slots above confidence 0.9, last step fills rest. No temperature, no Gumbel/categorical sampling, no `passes` count.
- MEASURED: `results_t4.yaml` smoldlm-144m sample is bracket soup; smoldlm row has no HumanEval (`bench/config.yaml:68`, `humaneval: false`).
- PAPER (smoldlm model card / modeling_smoldlm.py behavior, as cited in verified ledgers): "Always use temperature >= 0.5; greedy decoding causes mode collapse in diffusion LMs" (scope: this checkpoint). UNCONFIRMED here beyond the ledger citation; treat as a strong hint, not a proof until V3 confirms.
- PAPER arXiv 2609.33947 (34-170M MDLM, verified in hillclimb/fixplan ledgers): τ=1 is generically suboptimal under parallel unmasking; modest sharpening improved gPPL 3.5-4.9x at 16-1024 steps on unchanged 170M checkpoints, largest at few steps. Caveat from same paper: tested 8B models already use greedy and positive temperatures do not help (Section A.8), so this is a small-model/few-step effect. 100M is in-scope for the effect, but the magnitude is not guaranteed.
- Defect D3 (DERIVED/predicted in fixplan §1.1): the 0.9 confidence threshold is calibrated to greedy max-probs. Lowering τ changes confidences and can reduce fills per pass, making the last step dump many tokens in parallel from an under-denoised context (effective denoising budget collapse). This must be measured, not assumed.

### Required sampler fix (minimum, before any L1 measurement)

Implement in `bench/backends.py` (or an equivalent new backend used by eval) without changing timing contract semantics except added reporting:
1. **`temperature` (τ) knob:** default 0.0 preserves old behavior for regression; quality runs must set τ explicitly. Support Gumbel-style categorical sampling in fp32/fp64 noise precision as per smoldlm's own sampler spirit (fixplan §1.2 shows a reference patch; exact implementation must be verified against model behavior).
2. **fp32 softmax:** compute probabilities in fp32 over the vocab (49,152 for smoldlm; our model vocab 32k per spec) to avoid fp16 tail errors (MEASURED defect D2).
3. **`commit` mode:** support `topk` schedule (`n = max(1, n_masked_open // remaining_steps)` by confidence) in addition to confidence-threshold commit, because confidence commit confounds τ with denoising budget (D3). Run τ sweeps under `topk` for isolation; report both.
4. **Instrumentation:** add `passes`, `blocks`, `tokens_emitted`, `tokens_per_pass` to `Generation`/results and log per-step fills in debug runs to detect D3.
5. **Do not wire KV cache into this fix:** cache changes semantics and would confound measurement (fixplan §1.2, hillclimb-speed notes referenced there). Quality measurement runs uncached for comparability; cache is a later prerequisite for distillation throughput, not for L1 validity.

### Mandatory gates before L1v2

| Gate | Check | Pass | If fail |
|---|---|---|---|
| G0 Harness calibration | Run SmolLM2-1.7B base through our HumanEval/MBPP exec harness (all 164 HumanEval problems, greedy, n=1) and compare to published 22.6 pass@1 (PAPER, arXiv 2502.02737 Table 3). T4 cost ~0.6 T4-h ESTIMATE. | Within ±6 points of 22.6 (sampling + prompt-format allowance). | Fix harness before any quality claim. |
| G1 Regression | Patched backend at τ=0 on the same 1k/1k prompt reproduces `results_t4.yaml` smoldlm sample behavior (≥99% token agreement, allowing fp32-softmax near-tie flips). | ≥99% agreement or explained. | Understand fp32/argmax delta before proceeding. |
| G2 Sampler sanity (V3/V4/V5 style) | (a) Run smoldlm author's own decoder on MBPP-500 (τ=0.7, steps=10, cached/topk per model default) as upper bound; (b) run our loop at τ=0.9 and τ=0.7 with `topk`; (c) teacher-forced likelihood on 200 MBPP reference solutions vs line-shuffled bodies. | If author decoder > 0 and our loop == 0, bug is in our loop. If τ sweep gives ≥2x better gPPL ratio vs τ=0 at 8 steps (or clear degeneracy-metric improvement with stable `tokens_emitted`), decoder is a lever. If likelihood shows no code signal, we are in World A. | Stop and debug loop (mask/EOS/prompt/attention) if author>0 but ours==0. If World A, drop pass@1 endpoint for pilot and use gPPL/NLL only. |
| G3 Recipe/pipeline freeze | Week-2 compute probe (200 train steps, batch/seq per plan) on the D1 base architecture; data prep + decontamination plan frozen (HumanEval and MBPP held out; n-gram at minimum, stronger passes recorded); chosen τ/commit/steps/block written explicitly. | Probe gives feasible wall-clock for 500M-token pilot (see §7). Val NLL can be logged. | Do not start L1v2 twins; adjust scale first. |

**HumanEval remains sealed for final eval** (spec). MBPP is the pilot coding endpoint. Calibration use of HumanEval-164 on SmolLM2-1.7B is allowed only as a harness-validation run on a model we did not train (fixplan §2.3), and must not contaminate D1 data.

---

## 6. Run order and cost (rough T4-hours, zero budget)

All costs are T4-hours unless stated; wall clock assumes Kaggle ~30 GPU-h/week and Colab sessions <=12 h with disconnects (FACT per fixplan §6, A4). ESTIMATEs are labeled.

| Order | Item | Depends on | T4-h (rough) | Wall clock (rough) | Output / decision |
|---:|---|---|---:|---:|---|
| 0 | Build eval/metrics: `eval/genppl.py`, `eval/mbpp.py`, `eval/metrics.py`; add pass/degeneracy reporting | none | 0 (coding) + 0.3 T4-h smoke | days | gPPL sanity (GT < 30), MBPP loads 500 |
| 1 | Sampler patch + regression (G1) | 0 | 0.2-0.3 | same day | τ=0 parity, fp32 softmax, `topk`/τ knobs, pass counters |
| 2 | Harness calibration (G0) + author-decoder upper bound + likelihood (G2a,c) | 1 | 0.8-1.0 | 1 session | harness trusted or fixed; World A/B signal |
| 3 | τ/commit grid at 8/10/16/32 on smoldlm (G2b), gPPL + MBPP-500 subset/full as feasible + degeneracy metrics | 2 | 2.5-3.5 | 1-2 sessions | frozen decode policy (τ*, commit, steps) |
| 4 | Week-2 compute probe (200 train steps, D1 base config) + data decontamination freeze (HumanEval+MBPP) | 3 | 0.3-0.5 T4-h + CPU time | 2-4 days | tok/s, feasible pilot token count, G3 |
| 5 | Base pretrain smoke: D1 control at 100-200M tokens (or smallest run that yields stable val NLL + gPPL curve) | 4, G3 | 2-5 | ~1 week | working pipeline, checkpoints, eval loop, resume path |
| 6 | **L1v2 pilot twins** (A control, B looped-mid-block), 500M tokens each at matched `F_train` plan, 2 seeds if affordable else 1 | 5 | 20-30 (ESTIMATE: 2 x ~10-15 T4-h for 500M-token arms; scale with probe) | 2-3 weeks | val NLL @ matched FLOPs, gPPL @ 8/16, MBPP, convergence diagnostics |
| 7 | Decision: pass/fail per §3.2. If pass and pipeline cheap enough, optional 1B-token confirmation; if fail, stop looping | 6 | 0-20 | 0-2 weeks | L1 closed with clean result |

**Rough whole-pilot cost (G0-G3 + L1v2): ~24-40 T4-h, ~4-7 weeks wall clock** on free tiers (ESTIMATE, dominated by step 6 and free-tier friction). The 3B-token full pretrain (~64 T4-h ESTIMATE) is **not** part of L1v2; L1v2 is a gate before committing the full run to a looped recipe.

### What must be true before L1v2 starts (hard gates)

1. G0-G3 all passed and logged.
2. Decode policy frozen with explicit τ/commit/steps/block/confidence and D3 checked (`tokens_per_pass` sane).
3. Pretrain pipeline can train, checkpoint to Drive, resume, and evaluate val NLL + gPPL on schedule.
4. Decontamination run against HumanEval and MBPP completed and residual reported (MBPP leak invalidates L1's pilot metric).
5. Param counts, `F_step`, `F_train`, and `F_infer` formulas for A and B computed and written into the run config before step 1 of training.
6. HumanEval not used for pilot decisions; MBPP prompt template frozen and recorded.

---

## 7. Size/budget check: two 100M twins on 3-5B tokens each on a free T4

### Honest answer

**Two 100M twins on 3-5B tokens each is not honestly feasible as a default plan on a free T4/Kaggle-only budget.** It is *possible in arithmetic* under optimistic throughput, but not a responsible commitment.

Calculation (transparent):
- `quality-fixplan-ernie.md` §6: 3B tokens at 13,000 tok/s (A3, ESTIMATE) = 64 T4-h for one run; 5B at 13k tok/s = ~107 T4-h. Plausible range 5,000-13,000 tok/s => 64-167 T4-h for 3B, 107-278 T4-h for 5B. Two arms => **128-334 T4-h for 3B/arm, 214-556 T4-h for 5B/arm**, before eval, restarts, or failed runs.
- Free-tier wall clock: Kaggle 30 GPU-h/week (FACT in fixplan) => 128 T4-h is ~4.3 weeks minimum for 3B+3B at optimistic throughput and zero waste; 334 T4-h is ~11 weeks. 5B+5B is ~7-19 weeks. Colab adds session deaths and restart overhead.
- Context: the 135M yardstick SmolLM2-135M was trained on 2T tokens (PAPER, arXiv 2502.02737); 3-5B is already 400-600x less, so a weak/ambiguous result is likely and would consume months.
- LoopMDM's looped treatment also changes per-step compute; if run at equal tokens with S_train>1, training FLOPs (and time) rise for arm B unless tokens are reduced to match `F_train`.

**Conclusion:** "two 100M twins on 3-5B tokens each" is a stretch goal, not a pilot. Committing to it before probe data risks burning the project's free-tier runway on an unreadable result.

### What is feasible

| Plan | Tokens/arm | Arms | Est. T4-h (at 13k tok/s; halve confidence until probed) | Est. wall clock (30 h/wk) | What it can answer |
|---|---:|---:|---:|---:|---|
| **Feasible pilot (recommended)** | 500M | 2 (A,B) | ~21 T4-h total (+ ~3-5 for eval/probe) | ~1.5-2.5 weeks | Directional val NLL / gPPL at matched `F_train`; convergence diagnostic; pipeline proof. MBPP likely ~0-2% (low power). |
| **Stretch pilot** | 1B | 2 | ~43 T4-h total | ~2.5-4 weeks | Better NLL signal; still MBPP-underpowered. Run only if 500M pipeline is stable and probe >= ~10k tok/s. |
| **Single full run, no L1 twin at 3B** | 3B | 1 | ~64 T4-h | ~2-3 weeks | Base D1 quality; compare to SmolLM2-135M gap. No looping claim. |
| **Not feasible as pilot** | 3-5B | 2 | 128-556 T4-h | 4-19 weeks | Theoretically informative, practically a schedule risk; only consider after a passing 500M/1B pilot and a measured >=13k tok/s probe. |

**Rule:** the week-2 probe (G3) sets the real number. If measured throughput < 6,000 tok/s, cut to 250-500M tokens/arm or revisit block/seq settings; do not start a 3B twin run on hope (fixplan §5 D2 logic applied to L1).

---

## 8. Replacement L1 section (ready to paste into spec)

Replace the entire `## Pilot experiment L1: looped core (decided 2026-10-30)` section (through the end of its "Verification skill..." paragraph) with the following.

```md
## Pilot experiment L1v2: looped mid-block (revised 2026-11-11)

Replaces the 2026-10-30 L1. The original spec tied loop iterations to denoising steps and claimed "TPS untouched by construction" with an equal-TPS kill criterion. That arrangement is not the one supported by looped-diffusion evidence, and equal TPS is the wrong bar when looping adds work inside a denoising step. L1v2 keeps the twin-experiment structure and the "one clean negative kills only looping" discipline, but changes the treatment, the metric, and the prerequisites.

**Mechanism (treatment).** Loop a small early-middle mid-block *inside* each denoising step (LoopMDM-style, arXiv 2605.26106, 170M MDM): a head of unshared layers, a mid-block of 2 layers looped S times, then a tail. During the S inner iterations the denoising-step input/context is fixed; the step advances only after iteration S. Denoising step count T and inner loop count S are independent (do not tie loop iterations to T). Control arm: unlooped baseline, same architecture family, same training pipeline.

**Hypothesis.** At matched total training FLOPs and matched inference FLOPs, the looped-mid-block twin beats the unlooped twin on validation NLL, with the advantage at least as large at 8 decode steps as at 16 (looping should help the few-step/product regime). This is a training-efficiency / quality-per-compute hypothesis, not a "free TPS" hypothesis.

**Prerequisites (hard gates, all must pass before training L1v2 twins):**
1. Sampler/eval instrument fixed: `bench` decode supports explicit `temperature` and `commit` (`topk` and confidence), fp32 softmax, and logs `tokens_emitted`, `passes`, `blocks`, `tokens_per_pass`. τ=0 regression reproduces prior behavior; degeneracy metrics and gPPL harness (`eval/genppl.py`) and MBPP harness (`eval/mbpp.py`, 500 test problems, frozen template recorded in YAML) are in place. Smoldlm author-decoder check and a τ sweep establish that decode policy is not the dominant artifact (or the result is labeled World A and pass@1 is dropped as an endpoint).
2. Harness calibration: HumanEval-164 on SmolLM2-1.7B base through our harness is within ±6 points of 22.6 (arXiv 2502.02737 Table 3) as an instrument check on a model we did not train. HumanEval stays sealed for D1 final eval; MBPP is the pilot coding endpoint.
3. Recipe frozen: week-2 compute probe measured, data decontaminated against HumanEval and MBPP (residual reported), decode policy (τ, commit, steps, block, confidence) written explicitly, and a base D1 smoke pretrain (100-200M tokens) proves train/checkpoint/resume/eval works.

**Setup.** Two ~100M twins, same data, same optimizer family, same token budget plan, evaluated at T=16 and T=8 decode steps with the frozen decode policy: (A) unlooped control (S=1). (B) looped mid-block, 2 layers looped inside each denoising step, fixed S_train chosen before run (default 4), S_decode swept {1,2,4}. Pilot budget: 500M tokens per arm (preferred feasible pilot on free T4/Kaggle); 1B per arm only if the week-2 probe and pipeline stability justify it. Do not run 3-5B tokens per arm as the first L1 pilot on free tiers (128-556 T4-h class, 4-19 weeks wall clock at 30 GPU-h/week, ESTIMATE; see `research/l1-fix.md` §7). Report param counts, per-step FLOP accounting, and tokens seen for both arms.

**Compute-matched definitions (implementable).** Let `L_total=16`, `L_loop=2`, `L_fixed=14`, `F_block` the FLOPs of one full forward pass over the scored block context, `F_fixed = F_block*(L_fixed/L_total)`, `F_mid = F_block*(L_loop/L_total)`. Then inference FLOPs per block are `F_infer(T,S) = T*(F_fixed + S*F_mid) = T*F_block*(0.875 + 0.125*S)`. Arms are inference-compute-matched when `F_infer` is within ±10% (nearest grid point reported with ratio). Training comparison is at matched total training FLOPs (LoopMDM metric): hold `F_train` equal by adjusting tokens/steps for the looped arm's higher per-token cost, and log both tokens and `F_train`. "Equal TPS" and "equal denoising steps" are reported but are not the pass condition.

**Metrics.**
- Validation NLL (teacher-forced, held-out val shard), at matched `F_train`. Primary endpoint.
- Generative PPL ratio: `gPPL(model)/gPPL(ground-truth continuation)` scored by `gpt2-large` fp16 on 512 prompts x 64 in / 128 out from held-out val, bootstrap 95% CI; double to 1024 samples if CI half-width > 5%. Treat gPPL ratio deltas < 3% as noise unless consistent across steps and seeds (ESTIMATE floor, tighten after measurement). Ground-truth sanity: gPPL < 30.
- MBPP pass@1, `google-research-datasets/mbpp` test split (500 problems), n=1 per problem, 3 seeds reported as median/min/max, frozen prompt template (copied from arXiv 2108.07732 Appendix A, UNCONFIRMED in repo until copied), 256 max new tokens. Noise floor: < 2.0 points absolute is not a win unless it holds across all 3 seeds and is supported by NLL/gPPL. HumanEval stays sealed for final eval only.
- TPS: median of 5, batch 1, 1k in / 1k out, reported with `tokens_emitted`, `passes`, `blocks`, `tokens_per_pass`, min/max TPS. Outcome metric only.
- Diagnostics: per-inner-iteration top-1 agreement / NLL delta (to detect "decided in iteration 1", arXiv 2610.18127 risk) and degeneracy metrics (`soup`, `rep4`, `parse_ok`, `mask_frac`, `entropy_H`).

**Kill criterion (compute-matched).**
- **PASS (keep looping):** at 8 decode steps, arm B beats arm A on val NLL at matched `F_train` and matched `F_infer` (within ±10%, ratio reported) in both seeds, or by >= 1.0% in a single-seed run with consistent gPPL-ratio direction. Convergence diagnostics must show S>1 contributing (not iteration-1-only). At 16 steps B must be no worse than A at matched compute; B-vs-A gap at 8 steps should be >= gap at 16 steps to support the few-step hypothesis.
- **FAIL (drop looping from v1):** no val NLL win at matched `F_train` and matched `F_infer` at 8 steps (delta < 0.5% single-seed, or split across seeds), or win only at higher inference FLOPs than A.
- **INCONCLUSIVE:** divergence, harness/pipeline bug, or gate failure; fix and rerun once at pilot scale, do not scale up. A 16-step-only win is not sufficient. Equal-TPS is never a pass condition.

**Accepted risks and non-goals.** Negative evidence exists (arXiv 2610.18127: 76-89% tokens settled in iteration 1 in 6/8 looped AR LMs; 2604.07822: overthinking/grokking risk), but positive evidence is closer to our setting (LoopMDM 170M MDM, 3.3x fewer training FLOPs at matched FLOPs). No learned exit gate, no adaptive steps, no verification-during-refinement in L1v2. L1v2 does not change the product TPS claim except by reporting measured TPS; any TPS drop from looping inside the step is an honest cost.

**Verification skill (deferred).** The spec's deferred verify-during-refinement idea is not stacked onto L1v2. A cheap training-free confidence-remask diagnostic may be run alongside decode-policy work (separately, matched NFE) because confidence is a same-forward-pass signal (PRISM reports at 170M, arXiv 2510.01384; CoDR training-free), but the learned verification/adaptive-steps mechanism remains its own experiment after L1v2, and should be dropped if remasking shows no usable signal at 100M.
```

---

## 9. One-paragraph summary of what changed and why

L1 is not killed, but the spec's version is: "loop iterations = denoising steps" and "equal TPS, TPS untouched" are replaced by a LoopMDM-style loop-inside-each-denoising-step treatment (evidence at 170M MDM, arXiv 2605.26106, with the same arrangement in 2610.04198/2610.30066), because every positive looped-diffusion result loops inside the step and doing so makes each pass cost more FLOPs, so the kill bar becomes matched training FLOPs and matched inference FLOPs (`F_infer(T,S)=T*F_block*(0.875+0.125*S)` for a 2-of-16-layer mid-block) instead of equal TPS. The pilot is shrunk to 500M tokens/arm behind hard gates (sampler fix for greedy/fp16/D3, gPPL+MBPP harness, harness calibration, compute probe, decontamination) because our current diffusion decode is greedy/degenerate and unmeasured (MEASURED in `bench/backends.py`, `results_t4.yaml`) and two 3-5B-token twins are not honestly feasible on free T4/Kaggle (ESTIMATE 128-556 T4-h, 4-19 weeks). Quality is judged by val NLL at matched `F_train` (primary), gPPL ratio and MBPP-500 (with explicit noise floors and seeds), and TPS is reported as an outcome, while convergence diagnostics guard against the real "decided in iteration 1" risk (arXiv 2610.18127). HumanEval stays sealed; MBPP is the pilot endpoint.
