# L1 pilot edges (adversarial pass, independent)

Date: 2026-11-11
Author: iceman (review sub-agent)
Scope: edge-finding on L1 "looped core" as specified in `spec/edgefast-diff-100m.md` section "Pilot experiment L1: looped core (decided 2026-10-30)" and related repo state. I did **not** read any parallel revised-L1 draft.

Source labels used here:
- **MEASURED**: observed in repo files I read (`spec/edgefast-diff-100m.md`, `research/hillclimb-quality-ernie.md`, `research/quality-fixplan-ernie.md`, `bench/backends.py`, `results_t4.yaml`, `bench/BENCHMARK_WEEK.md`).
- **CONFIRMED (repo research)**: claim appears with a primary-source citation in repo research notes; I did not fetch the paper myself in this session.
- **UNCONFIRMED**: plausible but not backed by a source I can point to in this repo/session. No invented numbers.

## What L1 claims today (MEASURED)

From `spec/edgefast-diff-100m.md`:
- Two 100M twins, same data, same budget: (a) unlooped, all params distinct; (b) looped core re-used across denoising steps (fewer params, same inference cost).
- Metrics: generative PPL, pass@1 on MBPP (HumanEval sealed), TPS. Run both at 16 and 8 steps.
- Hypothesis: at equal TPS, (b) beats (a) on quality, gap widens as steps shrink.
- Kill: if (b) doesn't beat (a) on quality at equal TPS at either step count, drop looping.
- Framing: "Extra effective depth at zero extra passes, so TPS is untouched by construction."
- Joins benchmark week at zero cost on T4.

`research/quality-fixplan-ernie.md` already states L1 as specified should not run (treatment/mechanism and equal-TPS framing called out). `research/hillclimb-quality-ernie.md` raises related measurement and mechanism concerns. This document lists the edges the revised spec must close, independent of that draft.

## 1. Measurement edges

### 1.1 Generative PPL is not implemented and not calibrated
- **MEASURED**: `bench/` measures speed and HumanEval; no generative-PPL code/metric path is described in the files read. `results_t4.yaml` has no generative PPL.
- **Edge**: L1 lists generative PPL as a primary metric but there is no harness, no prompt set, no judge model, no continuation protocol, no normalization, and no ground-truth sanity floor in-repo.
- **Why it matters**: without a fixed protocol, "generative PPL" can be moved by prompt slice, generation length, judge, tokenization, or whether prompt tokens are included. Two twins can look different for harness reasons.
- **How revised spec must handle it**: define the full gPPL recipe before L1 (prompt source held out from training, continuation length, sample count, judge model, ratio/normalization, bootstrap/CI rule, and a ground-truth sanity check). Treat missing gPPL infra as a blocker, not a detail.
- **Severity**: Critical.

### 1.2 MBPP pass@1 sample size and noise floor are undefined
- **MEASURED**: L1 names "pass@1 on MBPP" but gives no n, no seeds/runs, no confidence interval rule, no pass/fail definition details (prompt template, stop conditions, test execution). `bench/BENCHMARK_WEEK.md` says HumanEval stays sealed but does not define MBPP protocol.
- **CONFIRMED (repo research)**: `research/quality-fixplan-ernie.md` states MBPP test split is 500 problems and notes binomial noise is large at small n/low pass rates (it gives HumanEval-50 95% interval examples as ±6.0 pts at p=0.05 and ±8.3 pts at p=0.10, derived/binomial; not a primary citation I fetched). It also recommends seeds and reporting median/min/max.
- **UNCONFIRMED**: I do not have a primary pass@1 variance paper/source in this session. A standard binomial approximation for a single run is `SE = sqrt(p*(1-p)/n)` and a 95% interval is roughly `±1.96*SE` when assumptions hold; at very low p or small n this is approximate and run-to-run decode randomness adds variance on top.
- **Illustrative (DERIVED, not measured)**: if true pass rate is 5% on n=50, 95% interval ~ ±6.0 pts; on n=200 ~ ±3.0 pts; on n=500 ~ ±1.9 pts. A 2-point gap is unreadable at n=50 and still fragile at n=200 once seed/decode noise is included.
- **Why it matters**: L1 compares two weak 100M twins where differences may be 1-3 points. Without predeclared n, seeds, and a minimum detectable gap, the result can be noise dressed as signal.
- **How revised spec must handle it**: predeclare MBPP n (prefer full 500 for decisions), number of seeds, decode settings per seed, aggregation, and the minimum gap that counts as a win. Require paired evaluation on identical problems/seeds where possible. State "inconclusive if CI/gap overlap" up front.
- **Severity**: Critical.

### 1.3 Current quality baseline is contaminated by known decode/harness issues
- **MEASURED**: `results_t4.yaml` shows `smoldlm-144m` sample as bracket soup and no HumanEval entry (`humaneval: false` in config context per notes; row has speed only). `smollm2-135m` emitted 307 tokens/run and `lfm2.5-350m` emitted 12 tokens/run in timed runs, yet TPS is reported as tokens/decode_seconds. `bench/BENCHMARK_WEEK.md` defers diagnosing early stops. `bench/backends.py` `CustomBackend` uses greedy `probs.max(dim=-1)` and fp16 softmax over vocab.
- **CONFIRMED (repo research)**: `hillclimb-quality-ernie.md` cites smoldlm model-card guidance "Use temperature >= 0.5" and arXiv 2609.33947 on sampler effects at small scale. I did not fetch these sources myself.
- **Edge**: "quality" numbers and samples in the repo today cannot anchor L1. The size-matched diffusion-vs-AR reference is not yet valid.
- **Why it matters**: if the yardstick and diffusion sample path are broken/misaligned, L1 can "beat" a bad baseline or "lose" because of harness issues.
- **How revised spec must handle it**: require a pre-L1 measurement gate: fix early-stop/token-count accounting, run a non-greedy diffusion decode path per model guidance, log tokens emitted and passes, and validate at least one external/published anchor where possible before trusting L1 deltas.
- **Severity**: Critical.

### 1.4 TPS is not a stable quality denominator when generation behavior differs
- **MEASURED**: AR models in results stopped early (307, 12 tokens) but timing/TPS are reported over decode seconds; diffusion emits fixed blocks/passes. No EOS/stop parity defined across twins.
- **Edge**: if one model emits fewer useful tokens, stops differently, or degenerates into loops, tokens/s and "quality at equal TPS" compare different work.
- **Why it matters**: L1's kill criterion is "quality at equal TPS". A model can win TPS by doing less useful work or lose TPS by generating longer/worse continuations.
- **How revised spec must handle it**: define denominator(s): tokens generated, tokens accepted/useful, wall time per completed problem, and forward passes. Report stop reasons and tokens emitted. Compare at matched compute and matched decode budget, not TPS alone.
- **Severity**: High.

### 1.5 "Generative PPL" can reward fluency while missing code correctness
- **MEASURED**: L1 goal includes pass@1 guardrail in spec intro, but L1 metrics listed are gPPL, MBPP pass@1, TPS.
- **Edge**: at 100M, a model may lower PPL by producing fluent but wrong or repetitive code-like text; pass@1 may stay near zero/noisy.
- **Why it matters**: a twin could "win" gPPL and lose executable correctness, or vice versa. L1 needs a decision rule across metrics.
- **How revised spec must handle it**: predeclare primary decision metric for the kill (and why), secondary diagnostics, and what happens when metrics disagree.
- **Severity**: High.

## 2. Confounds that leak into twin (a) vs twin (b)

### 2.1 Greedy/custom decode policy is a confound, not a neutral backdrop
- **MEASURED**: `CustomBackend` is greedy argmax with confidence threshold 0.9 and last-pass fill; `bench/BENCHMARK_WEEK.md` notes smoldlm greedy degenerates.
- **CONFIRMED (repo research)**: `hillclimb-quality-ernie.md`/`quality-fixplan-ernie.md` argue sampler/temperature and commit policy can dominate quality at few steps.
- **Leak?**: if L1 is run through a decode policy tuned/calibrated for one behavior, the looped/unlooped twins may interact differently with confidence dynamics. This does **not** automatically cancel.
- **How revised spec must handle it**: either (a) fix and freeze a justified decode policy before L1 and report sensitivity, or (b) include decode-policy controls. Do not claim "architecture effect" from a known-bad sampler.
- **Severity**: High.

### 2.2 Uncached loop and pass accounting
- **MEASURED**: `CustomBackend.generate` rebuilds `seq = torch.tensor([out_ids + block], ...)` every step/fill and runs full forward passes without KV/prefix cache. No pass counter in results.
- **Edge**: uncached recompute cost scales with sequence length and block updates; any structural change that alters logits/confidence fill patterns changes number of effective "progress" events and timing profile differently than a cached design.
- **Leak?**: may not cancel if (a) and (b) differ in confidence trajectories or per-pass compute.
- **How revised spec must handle it**: log total forward passes, tokens committed per pass, and wall time; keep cache mode explicit (uncached/cached). If comparing to spec claims about prompt cache/overlap, state that L1 prototype does not implement them.
- **Severity**: Medium.

### 2.3 fp16 softmax over 49k-class vocab
- **MEASURED**: `CustomBackend` does `torch.softmax(logits[0, -self.block_size:], dim=-1)` on fp16 logits (`load_smoldlm` casts model to half when dtype float16). Vocab in model context is ~49k (MEASURED in repo research notes; not re-verified from config file here except implied by smoldlm references).
- **Edge**: reduced precision in softmax/confidence can perturb argmax near ties and confidence thresholds. With confidence-gated filling, tiny prob shifts can change which/how many slots fill.
- **Leak?**: unlikely to be identical across twins because different representations produce different near-tie distributions. Probably small but not proven zero.
- **How revised spec must handle it**: compute probabilities/confidence in fp32 for measurement runs, or run an fp16-vs-fp32 ablation showing no decision change.
- **Severity**: Medium.

### 2.4 0.9 confidence threshold calibrated to greedy, reused blindly
- **MEASURED**: threshold 0.9 is hardcoded in config patterns; last step forces fill-all-open; if nothing clears threshold, code commits the single most confident slot.
- **CONFIRMED (repo research)**: `quality-fixplan-ernie.md` identifies threshold behavior as interacting with temperature/commit policy (defect D3 argument).
- **Leak?**: yes, if looped and unlooped twins have different confidence calibration. A threshold can starve one model (many 1-token fallback steps) and not the other, changing effective denoising schedule and TPS.
- **How revised spec must handle it**: log fill counts per step/block and fallback frequency. Either retune threshold per model on a dev slice (risky/leaky if not careful) or use a fixed schedule/commit rule that is not confidence-distribution-dependent for the core comparison.
- **Severity**: High.

### 2.5 Early checkpoint / undertrained regime
- **MEASURED**: `bench/BENCHMARK_WEEK.md` notes smoldlm is an early checkpoint (~26B of 100B tokens) and no instruction tuning; `results_t4.yaml` shows degenerate sample.
- **Edge**: in an undertrained regime, quality metrics are near floor and variance is high. Small architectural effects can be masked or exaggerated.
- **Leak?**: affects absolute interpretation; can interact with twin comparison if twins learn at different speeds.
- **How revised spec must handle it**: for L1-trained twins, track learning curves (checkpoints over time), not a single late/early snapshot. State power limits of the pilot.
- **Severity**: High.

### 2.6 No instruction tuning / prompt-format mismatch
- **MEASURED**: spec says 100M pretrain/SFT path later; L1 is described as twins trained same way but benchmark context includes base/early models and MBPP/HumanEval style prompting.
- **Edge**: MBPP pass@1 is sensitive to prompt template and instruction-following behavior. Base models may need different few-shot formatting than instructed models.
- **Leak?**: cancels only if both twins are equally (mis)aligned to the eval format. Looping could change instruction adherence or calibration differently.
- **How revised spec must handle it**: freeze and document prompt templates for all evals; run format sanity checks; report both raw completion behavior and scored pass@1.
- **Severity**: Medium.

### 2.7 "Same data, same budget" is underspecified
- **MEASURED**: spec states same data, same budget but does not define token budget unit, optimizer steps, batch size, sequence lengths, number of seen tokens, or compute budget (FLOPs/GPU-hours/passes).
- **Edge**: if (b) has fewer parameters but same inference cost claim, training cost per token/step differs. "Same budget" can mean same wall clock, same tokens, same optimizer updates, or same FLOPs; each gives a different fairness claim.
- **Leak?**: directly breaks interpretation.
- **How revised spec must handle it**: define budget in explicit units (tokens seen, train FLOPs, and/or GPU-hours) and justify which one is the fairness criterion. Report all three.
- **Severity**: Critical.

## 3. Compute-matched kill criterion: break it

L1 kill criterion (MEASURED): "If (b) doesn't beat (a) on quality at equal TPS at either step count, looping is dropped..."

### 3.1 Why "equal TPS" is gameable / wrong-shaped
- **MEASURED**: spec also says "Extra effective depth at zero extra passes, so TPS is untouched by construction." But if loop iterations are literally denoising steps (spec text: "a looped model whose loop iterations ARE the denoising steps"), then mapping loops to steps changes what a "step" means; if instead looping happens inside each denoising step (alternative interpretation discussed in repo research), each step does more work and TPS drops.
- **CONFIRMED (repo research)**: `hillclimb-quality-ernie.md` argues the inside-step interpretation is what papers do and that equal-TPS is the wrong bar for that version. I did not verify those papers myself.
- **Edge**: the kill criterion depends on an implementation choice that is ambiguous in the current spec.

### 3.2 Write the denominator both ways (formulas)

Let:
- `T` = wall-clock decode seconds for a fixed task/output budget
- `N_tok` = generated tokens counted (must be defined consistently)
- `P` = total forward passes
- `F_pass` = FLOPs per forward pass (depends on active params and sequence/block work)
- `F_total = P * F_pass` (approx; ignores overhead)
- `theta` = stored parameter count
- `theta_active` = parameters actually used per pass
- `tok/pass = N_tok / P`

**Way A (throughput-matched, current L1 wording):**
- Compare quality at equal `TPS = N_tok / T` (or equal `T` for same `N_tok`).
- Gameable when: one model emits easier/shorter/useless tokens; one benefits from different cache/overhead behavior; "equal TPS" is achieved by changing steps/block policy rather than architecture; pass count differs but is hidden.

**Way B (compute-matched):**
- Compare quality at equal `F_total` (training and/or inference), or at equal `P * theta_active` as a proxy.
- Practical proxies: equal forward passes `P` and equal active params per pass; and separately equal training FLOPs/tokens seen.
- Gameable when: `F_pass` differs because sequence length/context handling differs; looped design reuses params but does extra compute per pass (or vice versa); parameter count differs but "effective depth" is claimed without counting FLOPs.

**Way C (user/value-matched, useful but different question):**
- Compare quality at equal wall-clock on T4 and equal output contract (same prompt, same max tokens, same stop rules).
- Gameable when: measurement includes/excludes prefill inconsistently or token accounting differs.

**What is fair here?** For "extra effective depth at zero extra passes" you need equal `P`, equal `N_tok` contract, and equal or explicitly reported `F_pass`/active params. For "loop inside step gives more compute" you need equal `F_total` (or at least equal `P * F_pass`) and must accept TPS may drop. A single "equal TPS" kill criterion cannot be fair to both mechanisms.

- **How revised spec must handle it**: pick one mechanism definition for (b), then predeclare the matching rule: (1) inference compute match, (2) training budget match, (3) latency/TPS reporting. Include formulas and required logs (`P`, `theta`, `theta_active`, tokens, time, config).
- **Severity**: Critical.

### 3.3 Param-count asymmetry inside the twin definition
- **MEASURED**: spec says (b) has "fewer params, same inference cost".
- **Edge**: fewer stored params but same passes means less model capacity stored; same inference cost means compute per token/pass is claimed equal. That is a different tradeoff than "same params, more depth". The experiment as written mixes parameter efficiency and loop benefit.
- **How revised spec must handle it**: define the control precisely (parameter-matched, FLOP-matched, or latency-matched) and state what claim each control supports. "Fewer params, same inference cost" is a product claim; it needs a separate control from "looping helps quality".
- **Severity**: High.

## 4. Training edges (two 100M twins, 3-5B tokens, free T4)

### 4.1 Token budget may be infeasible or undefined on free T4 wall clock
- **MEASURED**: spec gives 3-5B tokens as a run size and says free on Kaggle T4 (30h/week). `research/quality-fixplan-ernie.md` estimates ~64 T4-h for 3B tokens (ESTIMATE, assumption stated there) and ~2-3 weeks wall clock.
- **Edge**: if throughput is lower than assumed, "3-5B" becomes months or gets silently cut. A cut invalidates "same budget" comparisons across runs started at different times.
- **How revised spec must handle it**: require a measured throughput probe before committing L1, then set token budget from measured tok/s and available sessions. Log actual tokens seen per twin.
- **Severity**: High.

### 4.2 Data prep and mix are not ready
- **MEASURED**: spec says "train our own 32k BPE on a corpus sample" and lists FineWeb-Edu sample-10BT, The Stack Dedup Python, FineMath/OpenWebMath with 50/35/15 mix and in-prep decontamination. `bench/BENCHMARK_WEEK.md` says tokenizer decision affects pretrain data prep; benchmark week unaffected.
- **Edge**: tokenizer training corpus, BPE settings, special tokens (mask/EOS/pad), sequence packing, dedup, and filtering all change effective data. Any difference between twin data pipelines is a confound.
- **How revised spec must handle it**: freeze and hash: tokenizer, corpus manifest, mix, packing, max length, decontamination rules, and train/val split. Train both twins from identical packed shards and same sampling order/seed schedule.
- **Severity**: Critical.

### 4.3 MBPP/HumanEval decontamination is necessary but not sufficient
- **MEASURED**: spec says hold HumanEval out and n-gram overlap checks; MBPP is L1 metric.
- **CONFIRMED (repo research)**: `hillclimb-quality-ernie.md` cites contamination/paraphrase concerns (arXiv 2407.07565, 2602.12413) and recommends MBPP decontam + stronger checks. I did not fetch these papers.
- **Edge**: leaking MBPP (or near-duplicates/paraphrases) into pretrain/SFT data makes L1 pass@1 meaningless. n-gram-only checks can miss paraphrases (UNCONFIRMED by me beyond repo notes).
- **How revised spec must handle it**: decontaminate against both HumanEval and MBPP with multiple methods, keep an audit log, and report residual/known limits. Do not claim "decontaminated" without method details.
- **Severity**: High.

### 4.4 Checkpointing to Drive on free tiers is a reliability risk
- **MEASURED**: spec proposes Kaggle T4, checkpoints to Drive every N steps; Colab sessions short. No resume procedure specified.
- **Edge**: interrupted runs, partial uploads, optimizer-state mismatch, or shard-position loss can silently change training exposure. A resumed run is not identical unless RNG/dataloader state is restored exactly.
- **How revised spec must handle it**: specify checkpoint contents (model, optimizer, scheduler, RNG, dataloader/shard cursor, step, config hash), atomic upload/verify, and a tested resume path. Abort comparison if twins have unequal effective tokens/steps after resume.
- **Severity**: High.

### 4.5 Reproducibility on T4/sm75
- **MEASURED**: T4 is sm75 (no bf16, per spec). Determinism flags/settings not specified.
- **UNCONFIRMED**: exact bitwise reproducibility of PyTorch training on CUDA/T4 across runs/kernels is not guaranteed without careful settings and may cost speed; I cannot cite a specific guarantee here.
- **Edge**: run-to-run noise can exceed the L1 quality delta. Without seed control, a "win" may be luck.
- **How revised spec must handle it**: fix and log seeds (Python/NumPy/PyTorch/CUDA), dataloader order, config hash, library versions, and GPU type. Run at least one seed repeat (or paired checkpoints) to estimate variance before calling a winner.
- **Severity**: High.

### 4.6 Mid-run divergence or instability
- **MEASURED**: spec gives no divergence criteria, LR schedule details, grad clipping, loss-spike handling, or early-stop rules.
- **Edge**: one twin can diverge, NaN, or plateau due to hyperparams that are not equally suitable for looped vs unlooped parameterization. Treating a failed run as "looping lost" is wrong unless instability is part of the claim.
- **How revised spec must handle it**: predeclare stability monitors, max-loss/NaN policy, and rerun rules. If hyperparams are tuned, tuning budget/protocol must be symmetric and documented. Compare best-stable or matched-schedule checkpoints by a predeclared rule.
- **Severity**: High.

### 4.7 Block-diffusion training/inference mismatch
- **MEASURED**: spec says training computes clean prefix same way inference does (prompt cache exact, not approximated) and uses block-diffusion objective, block size 32. L1 introduces loop-tying across denoising steps/passes but does not spell out the training-time analogue.
- **CONFIRMED (repo research)**: `hillclimb-quality-ernie.md` references blockwise/objective-alignment concerns (arXiv 2508.19529 cited there). I did not fetch it.
- **Edge**: if training masks/steps/objective do not match how the looped core is used at inference (and at 8 vs 16 steps), the pilot measures mismatch, not the loop idea.
- **How revised spec must handle it**: specify the exact training objective, masking schedule, number of refinement/denoise steps seen in training, and how weights are tied/unrolled during training for twin (b). Make train/infer parity explicit.
- **Severity**: High.

## 5. Decision edges

### 5.1 What result leaves us unable to act?
- **MEASURED**: kill criterion is binary on "quality at equal TPS"; spec does not define tie/inconclusive handling.
- **Unactable cases**:
  - Metrics disagree (gPPL favors A, MBPP favors B, TPS differs).
  - Differences are inside predeclared noise/CI.
  - One twin is more stable/faster to train but slightly worse at the measured checkpoint.
  - Equal TPS cannot be achieved without changing decode settings (confound).
  - Harness/protocol bug discovered after runs.

### 5.2 Tie/ambiguous handling must be designed in advance
- **How revised spec must handle it**:
  - Define a **win**: primary metric improves by >= minimum detectable effect at predeclared n/seeds/CI, with no guardrail regression beyond threshold.
  - Define a **loss**: primary metric worse by >= MDE or guardrail regression.
  - Define **inconclusive**: confidence intervals overlap / MDE not reached / protocol violation / unequal budgets. Inconclusive is not a kill and not a pass.
  - Predeclare follow-up for inconclusive: increase n/seeds, fix harness, or run a narrower follow-up. Cap reruns/cost.
  - Predeclare precedence when metrics conflict (e.g. executable correctness over gPPL, but only if powered).
  - Require stopping the decision if logs show unequal tokens, failed resume, or mismatched decode settings.
- **Severity**: Critical.

### 5.3 "At either step count" can overstate weak evidence
- **MEASURED**: kill says drop looping if (b) doesn't beat (a) at either 16 or 8 steps.
- **Edge**: a tiny win at one step count and a loss/tie at another is not the same as "looping works". Conversely, requiring a win at both may be too strict if the hypothesis is step-dependent.
- **How revised spec must handle it**: state the decision rule per step count and overall (e.g. must win at 8 and not regress at 16, or aggregate by predeclared rule).
- **Severity**: Medium.

## 6. Spec edges already falsified or weakened by repo research

These are not "maybe later" concerns; current L1 text assumes them.

1. **"Loop iterations ARE the denoising steps" as the mechanism.** 
   - **MEASURED/CONFIRMED (repo research)**: `hillclimb-quality-ernie.md` cites multiple 2026 papers looping inside the denoising step (LoopMDM 2605.26106 at 170M, ALoDLM 2610.04198, Looped UDLM 2610.30066, Looped-DiT 2609.40305 vision) and argues step=loop conflation is the opposite arrangement. I did not verify papers myself. Current spec wording still states step=loop identity.
   - **Spec impact**: mechanism definition is likely wrong or at least not the evidenced one.

2. **"TPS is untouched by construction" under looping.**
   - **MEASURED**: spec asserts zero extra passes and unchanged TPS. **CONFIRMED (repo research)**: if looping is inside each step, per-step compute rises and TPS falls by construction (`hillclimb-quality-ernie.md`). 
   - **Spec impact**: kill criterion and fairness claim are internally tied to a disputed mechanism.

3. **"Joins benchmark week at zero cost."**
   - **MEASURED**: `bench/BENCHMARK_WEEK.md` says L1 needs training code that does not exist yet and defers it to the end. Spec says "runs free on a T4, so it joins benchmark week at zero cost."
   - **Spec impact**: schedule/cost claim is stale relative to benchmark-week plan.

4. **"Same inference cost" with "fewer params" is not a complete compute description.**
   - **MEASURED**: spec states fewer params, same inference cost. Without FLOPs/pass, active params, and pass count definitions, this is not checkable.
   - **Spec impact**: compute-matching is underspecified.

5. **Quality measurement readiness assumed.**
   - **MEASURED**: L1 requires generative PPL and valid MBPP pass@1; harness currently lacks gPPL and has known decode/token-accounting issues (`results_t4.yaml`, `bench/BENCHMARK_WEEK.md`, `quality-fixplan-ernie.md`).
   - **Spec impact**: L1 cannot be executed as specified with trustworthy numbers today.

6. **Yardstick comparison not yet valid.**
   - **MEASURED**: SmolLM2-135M emitted 307 tokens in timed runs; LFM2.5-350M 12 tokens. `bench/BENCHMARK_WEEK.md` marks these non-comparable until early-stop cause is diagnosed.
   - **Spec impact**: "gap to SmolLM2-135M" cannot be used as L1 context until fixed.

7. **Decode defaults in repo conflict with diffusion quality assumptions.**
   - **MEASURED**: `CustomBackend` greedy, fp16 softmax, confidence 0.9, uncached. Research notes flag these as problematic for smoldlm-like models (CONFIRMED in repo research, primary sources not re-fetched by me).
   - **Spec impact**: running L1 through current backend defaults risks measuring decode artifacts.

## 7. Minimum gates before any L1 run (summary)

1. Freeze mechanism: define exactly what "looped core" means in training and inference (across steps vs inside step), active params, and pass count.
2. Freeze fairness: state training budget (tokens/FLOPs/hours) and inference comparison (FLOPs, passes, wall time, TPS) with formulas.
3. Freeze data: tokenizer, corpus manifest/hash, mix, packing, decontam report, train/val split.
4. Freeze eval: gPPL harness, MBPP harness (n, template, seeds, execution), logging schema (tokens, passes, stop reason).
5. Pass measurement sanity: token accounting fixed, decode policy justified, at least one calibration/anchor check.
6. Freeze decision rule: primary metric, MDE/win-loss-inconclusive thresholds, tie handling, and rerun cap.

## Top 10 sharpest edges

| # | Edge | Why it matters | How the revised spec must handle it | Severity |
|---|---|---|---|---|
| 1 | "Equal TPS" kill criterion is tied to an ambiguous/wrong-shaped loop mechanism | Spec says "loop iterations ARE the denoising steps" and "TPS untouched by construction"; if looping is inside each step (the evidenced reading in repo research), compute per step rises and TPS falls. A kill rule on equal TPS can reject the right idea or accept the wrong implementation. | Define (b) mechanically (inside-step vs across-step, tied layers, unroll in training/infer). Replace "equal TPS" as the sole fairness bar with explicit formulas: log `P` passes, `N_tok`, `T`, `theta_active`, and estimated/inferred FLOPs; compare at matched inference compute and report TPS separately. | Critical |
| 2 | Generative PPL is required but no protocol/harness exists | L1 cannot be decided on a metric that has no prompt set, judge, length, normalization, CI, or sanity floor. Different choices can flip the winner. | Make gPPL implementation + frozen recipe a precondition: held-out prompts, continuation length, sample count, judge, gPPL ratio, bootstrap CI, ground-truth floor, and full logging. Block L1 until this exists and is smoke-tested. | Critical |
| 3 | MBPP pass@1 is underpowered/noise-prone without predeclared n and seeds | At 100M near-floor pass rates, 1-3 point gaps are often indistinguishable from sampling/run noise (binomial intuition: n=50 is very wide; n=500 is materially tighter). Undeclared n lets the experimenter stop when it "looks good". | Predeclare full MBPP-500 (or justify n), >=3 seeds, decode settings, aggregation, and minimum detectable effect. Use paired same-problem evaluation where possible. Define "inconclusive" when MDE not met. | Critical |
| 4 | "Same data, same budget" is undefined | Same wall clock, same tokens seen, same optimizer steps, and same train FLOPs are different budgets, especially when (b) has fewer params but different per-step work. Unequal exposure makes the quality delta meaningless. | Hash and freeze tokenizer/corpus/mix/packing/decontam/split. Define budget as actual tokens seen and train FLOPs (and log GPU-hours). Require equality checks before comparison; abort or label invalid if they differ after resume/failure. | Critical |
| 5 | Current decode/harness is a known confound (greedy, fp16 softmax, confidence threshold, token accounting) | `CustomBackend` is greedy argmax with fp16 softmax and a 0.9 threshold; smoldlm sample is degenerate in results; AR baselines had early-stop token-count issues. L1 could measure sampler/threshold interactions, not looping. | Add a measurement gate: fix token/pass accounting, justify or fix sampler/precision/commit rule, log fills per pass and fallback rate, and rerun/validate baselines. Freeze decode policy for both twins and report sensitivity. | High |
| 6 | No decision rule for ties, metric conflicts, or unactable outcomes | L1 can easily produce "gPPL up, MBPP flat, TPS down" or differences inside noise. Without a prewritten rule, the pilot ends in debate instead of a decision. | Specify primary metric, guardrails, win/loss/inconclusive thresholds, metric precedence, and rerun cap. Treat protocol violations (unequal budget, bug, unmatched decode) as automatic inconclusive/no-decision. | High |
| 7 | Checkpoint/resume and determinism on free T4 can silently break comparability | Kaggle/Colab sessions interrupt; partial Drive checkpoints or lost dataloader/RNG state change effective training. sm75/T4 nondeterminism and single-run variance can exceed the effect size. | Require resumable checkpoints (weights, optimizer, scheduler, RNG, shard cursor, config hash), integrity checks, and logged effective tokens/steps. Fix/log seeds and versions; include at least one variance/repeat check before declaring a winner. | High |
| 8 | Train/infer mismatch for the looped design is unspecified | L1 states a looped core reused across denoising steps but not how this is unrolled/trained, how many steps are trained, or how 8 vs 16-step decode relates to training. Mismatch can erase or fake the benefit. | Write the exact training objective and unrolling/weight-tying scheme for (b), plus the decode schedule. Ensure 8/16-step eval conditions are represented or explicitly treated as extrapolation. | High |
| 9 | Spec is stale vs repo reality on readiness and schedule | Benchmark week defers L1 because training code doesn't exist; gPPL/eval infra is missing; yardstick runs (307/12 tokens) are not comparable. "Zero cost, joins benchmark week" overstates readiness. | Revise L1 status to blocked behind infra/training-code/measurement-gate milestones with explicit exit criteria. Remove or qualify schedule claims until prerequisites are green. | High |
| 10 | MBPP/HumanEval contamination and prompt-format effects can invalidate pass@1 | MBPP is the pilot metric; spec currently emphasizes HumanEval holdout/n-gram checks. Leaks/paraphrases or prompt mismatch can create fake gains, especially dangerous for self-generated/fine-tune stages later. | Decontam against MBPP and HumanEval with documented multi-method checks and audit logs. Freeze MBPP prompt template/execution rules and sanity-check non-code/base-model behavior. Report contamination limits alongside pass@1. | High |

## Bottom line

L1 is a good question asked with an under-specified mechanism, an unready measurement stack, and a kill criterion that can be gamed or made unreadable. The revised spec should treat mechanism definition, compute matching, eval harness readiness, and decision rules as part of the experiment, not cleanup after the fact. If those are not frozen first, any "win" or "loss" from L1 is not actionable.