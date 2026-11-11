# Hillclimb: the QUALITY half of D1

Audit date: 2026-11-11. Author: ernie (research sub-agent, quality angle).
Question: at equal TPS, what most raises quality for a ~100M block-diffusion model trained on a 3-5B token zero-budget run?

## 0. How to read this

**Hard rule I applied:** every speed or quality number below is either
(a) "paper says" with a URL/arXiv id, or (b) "we measured" with the repo file,
or (c) labelled **UNCONFIRMED**. Nothing here is from memory of "what experts say".

Three labels are used throughout:

| Label | Meaning |
|---|---|
| **MEASURED** | from this repo's own runs (`results_t4.yaml` etc.) |
| **PAPER** | quoted from a primary source I fetched this session; scope is stated |
| **UNCONFIRMED** | plausible, no primary source fetched; do not build on it |

**What I fetched this session (primary sources, all opened, not just search snippets):**
arXiv 2609.11998 (fixed-state, full HTML incl. Table 5), 2602.12262 (T3D, abstract + results HTML),
2608.02942 (OPTD, abstract + full HTML), 2603.22075 (controlled AR-vs-MDLM at 123/163M, full HTML),
2610.00673 (Closing the Loop, abstract), 2605.26106 (LoopMDM, abstract), 2610.30066 (Looped UDLM),
2604.07822 (Loop/Think/Generalize), 2610.18127 (Monitoring latent reasoning in looped LMs),
2609.33947 (sharpening, full HTML), 2610.17590 (16x compute gap), 2607.02805 (BDLM Mamba-H),
2511.03276 (super data learners), 2509.20624 (FS-DFM), 2509.26328 (Fast-dLLM v2),
2510.01384 (PRISM), 2412.02674 (Mind the Gap), 2410.21035 (SDTT), 2410.18514 (SMDM scaling),
2404.17140 (small LMs need strong verifiers), 2503.03862 (Not-Just-Scaling-Laws),
2504.11393 (DataDecide), plus the NeurIPS 2026 poster page and GitHub for T3D.
Search was available this session (unlike the 2026-11-09 pass, which hit HTTP 402).

**Scale rule for this note.** 100M is not "small 1B". Most dLLM literature starts at 1.7B
(SDAR-1.7B), 7-8B (LLaDA, Dream), or 3B (fixed-state). I flag every paper whose evidence
does not reach our band, because that is the central honesty problem of this task.

---

## 1. The headline finding (read this before the rankings)

**The weak side of D1 is probably not the architecture or the step count. It is that
nobody has checked whether our decode loop is even a *sane generator*, and the one
paper that did check found the field's default sampler is wrong.**

Three measured facts from this repo, then one paper.

MEASURED (`results_t4.yaml`, smoldlm-144m row): the sample text is
`"\n    def _sift_down(self, i):\n        if =(([[]]]]\n]]]]]]\n]]]]]][[[]]]]\n..."` —
pure bracket soup after ~6 tokens. TPS 349.979. `bench/backends.py` `CustomBackend.generate`
draws tokens with `probs.max(dim=-1)`, i.e. **greedy argmax, temperature 0**. The model card
for smoldlm-144m says (PAPER, HF model card): *"Use `temperature >= 0.5` — greedy decoding
causes mode collapse in diffusion LMs (known artifact)."* We are running the one configuration
the model author explicitly warns against, and then reading its output as a quality signal.

MEASURED (`results_t4.yaml`): `smollm2-135m` emitted 307 tokens of 1024; `lfm2.5-350m`
emitted 12 of 1024. Both TPS numbers are therefore **not comparable** to the rest of the
table. `bench/BENCHMARK_WEEK.md` already flags this and defers the fix. The quality
comparison we care about most (smoldlm vs SmolLM2-135M at 100-144M) currently has no valid
right-hand side.

PAPER (arXiv 2609.33947, "Simple Diffusion Language Models Are More Effective Few-Step
Generators Than Reported", Purdue; full HTML fetched). Unchanged public MDLM-Small checkpoint,
unconditional OWT, 128 samples/cell, GPT-2 generative PPL:

| Steps | standard sampler (τ=1, fp64) | sharpened (τ=.9, fp32) |
|---|---|---|
| 16 | 357.6 | 102.2 |
| 64 | 154.9 | 41.3 |
| 256 | 115.5 | 28.1 |
| 1024 | 107.1 | 21.8 |

Their words: *"Modest sampler sharpening, without any model retraining, enables a couple
years old masked DLM to rival supposedly far improved successors."* And the mechanism is the
one that matters for us: *"parallel unmasking destroys dependencies among simultaneously
generated tokens"* — tokens revealed in the same round are drawn independently, so their joint
dependence is lost even if every marginal prediction is exact. Theorem 2 of that paper:
temperature 1 is a stationary point only on a measure-zero set of targets, i.e. **for an exact
denoiser, τ=1 is generically not the right temperature once you reveal tokens in parallel.**
Their enumeration found sharpening (λ>1) improved 279 of 280 random small targets.

Two more of their findings bear directly on our plan:

1. **Their sharpening gains are largest at few steps and vanish at many steps** — precisely
   our operating regime (8-16 steps).
2. **Caveat, and it is a real one for us:** *"The tested larger 8B models already use greedy
   decoding, and positive temperatures do not help (Section A.8)."* So this is a small-model
   effect. 100M is squarely in the regime where it should apply. That is the good news and
   the reason it is experiment #1.

**Consequence for D1.** Before spending a single GPU-week on L1 looping, we must establish
that our decode loop produces text a human would call text. Right now pass@1 0.0-0.10 across
all measured models is consistent with two very different worlds:

- **World A:** the models are simply too small/undertrained to write correct code. Then the
  only lever is pretraining, and quality is capped and we should say so.
- **World B:** the harness is mis-decoding, and some of the 0.0-0.10 is our measurement, not
  the model.

**We do not currently know which world we are in, and every downstream decision (L1,
distillation, data mix) is being made as if it were World A.** Distinguishing A from B costs
hours, not weeks. It is the highest-value action in this document.

---

## 2. Ranked actions

Ranking rule: expected quality gain per unit of (T4-hours + engineering risk), at fixed TPS.
"Multiplier" is stated as a range with its source's scope attached. I refuse to give a single
number where the literature does not support one.

---

### A1. Fix the decoder, then re-measure smoldlm's actual quality — before anything else

**Expected multiplier: unknown but potentially the largest single item in this document;
plausibly 3.5-4.9x on generative-PPL-type metrics (PAPER, 2609.33947: computed from their
Table 1, 357.6→102.2 = 3.50x at 16 steps, 154.9→41.3 = 3.75x at 64, 115.5→28.1 = 4.11x at
256, 107.1→21.8 = 4.91x at 1024 — all on a 170M MDLM). Could be ~1.0x on HumanEval if the model
truly cannot code. That uncertainty is the point: it is cheap to resolve.**

Cost: ~1-2 hours of T4 time. Zero training.

What to do:
1. Add temperature + categorical-precision as explicit knobs to `CustomBackend`
   (`bench/backends.py:165`), defaulting to τ=0.9, fp32 Gumbel noise (the paper's MDLM♯),
   with a collapse screen (their rule: pick the sharpest τ that keeps mean unigram entropy
   above a floor; they used H≥5 nats).
2. Re-run smoldlm-144m at τ ∈ {0.7, 0.8, 0.9, 1.0} × steps ∈ {8, 16, 32}, 1k/1k, median of 5.
   Report tokens emitted and pass@1 alongside TPS. **Do not** report TPS without tokens emitted
   again — that is what produced the 307/12-token rows.
3. Fix the SmolLM2-135M and LFM2.5 early-stop (already deferred in
   `bench/BENCHMARK_WEEK.md`) so the size-matched comparison has a valid baseline.

Verify: greedy vs τ=0.9 on the *same checkpoint* is a controlled A/B — the model is fixed, only
the sampler changes. If τ=0.9 emits 1024 coherent tokens where τ=0 emitted bracket soup, we are
in World B and everything downstream gets cheaper. Kill criterion: if τ=0.9 and τ=0.0 produce
equally degenerate text on a model the author says needs τ≥0.5, then the harness has a bug
(wrong mask id, wrong EOS, wrong chat template) and no amount of training research matters
until it is found.

**Risk of skipping this:** every subsequent experiment is measured with a broken instrument.
This is the "measure twice" step and it is not optional.

---

### A2. Run the few-step distillation ladder on smoldlm-144m — the 8-step regime is the product

**Expected multiplier: the literature's few-step results are large but all above our scale.
PAPER (T3D, arXiv 2602.12262, SDAR-1.7B/4B and LLaDA-8B): at the most aggressive budget
(TokPS=8, LLaDA-8B) average accuracy 17.11 → 47.13, i.e. **+175% relative**, and 68.71 → 74.75
at TokPS=4. PAPER (PlaidQ, 0.7B continuous, arXiv 2609.04531): at 4 steps pass@10 on HumanEval
7.90 → 15.85. Those are 0.7B-8B. At 100-144M I expect materially less; I will not put a number
on it. Call it "large if it transfers, near zero if capacity-bound" and measure.**

Cost: this is the expensive one. Distillation needs teacher rollouts (T3D: *"trajectory
collection requires full-step teacher rollouts, incurring an offline cost that scales with
dataset size and decoding budget"*). On a free T4 with a 144M model this is feasible because
the model is tiny — that is the one place our size is an advantage rather than a handicap.

The ladder, cheapest first:

| Rung | Method | Source | Scope of evidence | Fits T4? |
|---|---|---|---|---|
| 1 | Sharpened sampler (A1) | 2609.33947 | 34M-170M MDLM | yes, free |
| 2 | SDTT / iterated self-distillation | 2410.21035 | **169M / 424M / 863M** | yes — closest scale match in the whole literature |
| 3 | T3D trajectory self-distillation + DDO | 2602.12262 (NeurIPS 2026) | 1.7B, 4B, LLaDA-8B | heavy but possible on 144M |
| 4 | OPTD on-policy transition distillation | 2608.02942 | LLaDA-8B only (LoRA on TAD-S checkpoint) | no — 8B-only setup, 512 LoRA updates at rank 128 |
| 5 | CDLM consistency distillation | 2605.00161 | 110M models, OWT/LM1B | yes, and 110M is our scale |
| 6 | FS-DFM shortcut/self-consistency | 2509.20624 | **0.169B / 1.3B / 1.7B** | pretrain cost too high for us, fine-tune path may fit |

**The gap the scout flagged: confirmed, and here is its exact shape.** I checked every
few-step dLLM paper I could reach. The evaluated model scales are:
T3D = SDAR-1.7B, SDAR-4B, LLaDA-8B. OPTD = LLaDA-8B only (explicitly a LoRA adapter on a
released 8B checkpoint). CDLM = Dream-7B / LLaDA-8B. CD4LM = LLaDA. SDTT = 169M/424M/863M.
FS-DFM = 169M/1.3B/1.7B. PlaidQ = 0.7B (continuous, not masked).

So the honest statement is **not** "none demonstrated below 1B" — SDTT (169M) and FS-DFM
(169M) did, and CDLM reports 110M. The accurate statement is:

> **No trajectory-distillation or on-policy-distillation paper (T3D, OPTD, CD4LM, TOPD) has
> been run below 1.7B. The only sub-200M few-step evidence is SDTT (2024, 169M, MDLM, OWT/LM1B)
> and FS-DFM (169M, discrete flow matching), and neither tested a block-diffusion model of our
> shape on a code benchmark.**

That is a real, fillable gap and it is exactly D1's size. This is the strongest argument in
this document for the project existing at 100M: **we can produce the missing measurement.**

Kill criterion: if the 8-step student does not beat the 32-step teacher on MBPP pass@1 at
equal-or-better TPS, distillation is not buying quality at our scale and we stop after rung 2.

---

### A3. Train at the step budget you will deploy at (the "few-round training" bet, upgraded)

**Expected multiplier: unknown; this is the cheapest structural change and the one with the
best mechanism. PAPER support that the effect is real at our scale: arXiv 2603.22075 trained
AR and MDLM at 123M/163M on identical 50M-token TinyStories data and found AR overfits by step
14k while MDLM was still improving monotonically at step 20k. Also 2511.03276: DLMs need
">100x more FLOPs than AR to reach optimal performance" — i.e. at fixed compute the DLM is
*undertrained*, which is a different diagnosis from "diffusion is worse".**

Cost: one pretrain run. This is a change to the recipe, not an extra run, so the marginal cost
is zero if we do it before the main pretrain.

The spec already says this (section "Few-round training", line 53): *"the model must be
fine-tuned/distilled FOR the 4-8 round regime, not trained for many rounds then starved at
decode time."* The research adds one thing: **the training objective should match the decode
policy exactly, not just the step count.**

PAPER (arXiv 2508.19529, Blockwise SFT): standard SFT for diffusion randomly masks tokens across
the whole response, but inference decodes block-by-block from a clean prefix with the future
fully hidden. Three mismatches: noisy prefixes, suffix leakage, diluted supervision. Their fix
(partition into blocks, one active block per step, prefix clean and frozen, suffix fully hidden,
loss only on the active block) beat classical SFT by **+5.2 points on GSM8K and +1.6 on MATH at
matched FLOPs**, and their ablation shows performance *peaks when the training block size equals
the inference block size*. Scale caveat, verified from their full text: **all their experiments
use `GSAI-ML/LLaDA-8B-Instruct` with LoRA** (not a from-scratch small model, and not multi-scale),
so the mechanism is scale-free but the magnitude is unmeasured below 8B.

Also PAPER (arXiv 2609.11998, fixed-state, Table 5 — the source of the spec's 15.3/6.4):
the cliff is confirmed and it is *steep and monotone* across all three backbones:

| S (steps/block) | Attn Gen-PPL | Hybrid | Mamba |
|---|---|---|---|
| 4 | 15.26 | 17.14 | 18.50 |
| 8 | 12.10 | 12.85 | 13.05 |
| 16 | 9.22 | 9.06 | 9.50 |
| 32 | 6.44 | 6.48 | 6.52 |

Two things this table says that the spec does not:
- **8 steps is not half of 32 steps' quality; it is roughly 2x the PPL.** The spec's 8-16 step
  range is a genuine bet, and 8 is the risky end.
- **Architecture barely moves the cliff** (15.26 / 17.14 / 18.50 at 4 steps; 6.44 / 6.48 /
  6.52 at 32). Whatever flattens this curve is a *training/distillation* effect, not a mixer
  choice. That is evidence for spending our effort on A2/A3 rather than on the GDN-vs-attention
  question.

Kill criterion: if a model trained with blockwise-aligned objective at 8 steps does not beat
the same model trained with classical masking at 8 steps, the alignment claim does not hold at
100M and we keep classical masking.

---

### A4. Loop *inside* the denoising step, not across denoising steps (this is the L1 correction)

**This is the most important correction in this document. L1 as specified is probably the wrong
version of a right idea.**

The spec (line 116) says: *"a looped model whose loop iterations ARE the denoising steps."*
Read literally, that means: one forward pass per step, with the pass internally iterating a
shared core, and the *denoising step count* is the loop count. That conflates two different
things, and the literature has already separated them.

What the papers actually do — and all of them loop **inside** a step:

- **LoopMDM** (arXiv 2605.26106, NeurIPS 2026). *"selectively looping the early-middle
  transformer layers"* — a head of unshared layers, a **mid-block of 2 layers looped S times**,
  a tail with the LM head. Their Figure 2 schematic is explicit: the noisy input stays fixed
  during the loop and changes only *between* diffusion steps. Result: matches same-size MDM with
  **up to 3.3x fewer training FLOPs**, and beats deeper non-looped MDMs at comparable per-step
  compute. **Their model is 170M parameters.** That is our exact size.
- **ALoDLM** (arXiv 2610.04198). *"At each denoising step, ALoDLM iteratively refines
  representations in latent space"* — a shared Recurrent Core (Prelude / Core / Coda), up to
  K=4 passes, with a learned exit gate. Trained at 1.7B and 8B.
- **Looped UDLM** (arXiv 2610.30066). *"selective looping, which recursively applies a few early
  Transformer layers within each denoising step."* 2.8x / 3.3x fewer training FLOPs at matched
  NLL.
- **Looped-DiT** (vision, arXiv 2609.40305). *"looped computation within each denoising step
  before the sampler advances, allowing loop depth to be varied independently of the number of
  denoising steps."* Finds **"increasing loop depth yields larger gains than adding more
  denoising steps under a fixed inference budget"** and that a 260M looped model beats a 6.5x
  larger one. Vision, not text — but the architecture-level claim is the one we care about.
- **arXiv 2510.14961** goes the other direction: treats a recurrent-depth LM as a diffusion LM
  and gets up to 5x speedup on an existing 3.5B model without tuning.

**So: "loop iterations ARE the denoising steps" has been tried in spirit and the field converged
on the opposite arrangement.** The reason is mechanical. If the loop *is* the step, then a
4-step decode gives you 4 core applications total, and the extra depth has to fight the
factorization error of a huge jump. If the loop is *inside* the step, a 4-step decode gets
4 × S core applications, each on a nearly-correct context, which is where the compute actually
helps.

**The strongest counter-evidence to looping at all, and it is serious:**

1. **arXiv 2610.18127** ("Monitoring latent reasoning in looped language models", 8 looped LMs,
   4 families). Their findings: *"The answer is decided early. In six out of the eight models,
   76-89% of the generated tokens already settle on their final top-1 token in the first
   iteration."* And: *"More iterations are not always better. Base Ouro predicts the prompt best
   at iteration 3-4, and gets worse thereafter. The reasoning-tuned Ouro-Thinking models are
   best at iteration 1 and get worse with every iteration. Only Huginn improves steadily."*
   If a loop's extra iterations mostly re-confirm an already-made decision, then looping buys
   much less than the parameter-matched comparison suggests — and it can actively hurt.
2. **arXiv 2604.07822** ("Loop, Think, & Generalize"): recurrent-depth transformers show a
   **"three-stage grokking process"** before systematic generalization appears, and they
   identify **"overthinking"** — *"excessive recurrence degrades predictions and limits
   generalization to very deep compositions"*, with degradation past ~15 iterations. A 3-5B token
   budget may simply not reach the grokking point.
3. **The compute-matched control is missing across the field.** The looped-transformer survey
   (Parallax, 2026-10-14) states it plainly: *"the empirical case for looping is still made
   mostly against parameter-matched rather than compute-matched baselines"* and names *"the
   shortage of compute-matched controls"* as a top open problem. LoopMDM is the exception — it
   does run matched-total-FLOPs comparisons, which is why I trust it more than Ouro.
4. **The one result that looks like L1's hypothesis is a depth-growth result, not a looping
   result.** arXiv 2602.16490 ("From Growing to Looping") finds looped and depth-grown models
   share depth-wise signatures and that *inference-time looping applied to a depth-grown model*
   improves some reasoning primitives up to 2x "despite the model never being trained to loop."
   That is a much weaker and more specific claim than "looping beats an unlooped twin at equal
   TPS."

**What this means for L1 concretely.** The spec's twin experiment — (a) unlooped, all params
distinct vs (b) looped core reused across denoising steps, at 16 and 8 steps, MBPP + generative
PPL + TPS — is still the right *shape* of experiment (two twins, one variable, kill criterion),
but the treatment arm should be:

> **(b) loop the early-middle layers inside each denoising step** (LoopMDM's arrangement,
> 2 layers, S_max 6-12), **not** "loop iterations = denoising steps".

And the cost accounting changes: looping inside a step makes each pass more expensive, so
**TPS drops by construction.** The spec's claim that looping is "TPS untouched by construction"
is only true for the spec's version, and the spec's version is the one the evidence argues
against. Under LoopMDM's arrangement, the honest framing is the one LoopMDM uses: **matched
total training FLOPs and matched inference FLOPs, quality at equal compute** — not equal TPS.

**Expected multiplier (PAPER, 170M MDMs, matched total FLOPs): up to 3.3x fewer training FLOPs
for equal NLL, and +8.5 points on GSM8K at matched training compute (LoopMDM).** At our token
budget that is the single most attractive number in this document, because we are
compute-starved, not quality-starved — 3.3x fewer FLOPs is directly 3.3x more tokens.

**Risk, stated plainly:** the two negative results above (2610.18127, 2604.07822) are on
recurrent-depth *AR* models at 1.4B-3.5B, not on MDMs at 170M. The positive result (2605.26106)
is exactly MDM at 170M. **The positive evidence is closer to our setting than the negative
evidence.** That is why I still rank looping, but as a *training-efficiency* play with a
compute-matched kill criterion, not as a free-TPS quality play.

Kill criterion: at matched total training FLOPs and matched inference FLOPs, if looped-mid-block
does not beat the unlooped twin on validation NLL, looping is dropped. Note the change from the
spec's criterion — "equal TPS" is the wrong bar for this version.

---

### A5. Data mix: keep it, but do not expect it to beat architecture, and decontaminate harder

**Expected multiplier: real but second-order at 100M. PAPER (REGMIX, ICLR 2025): mixture choice
moves single-task performance by up to 14.6%. PAPER (arXiv 2503.03862, meta-analysis of 92 open
models): code fraction has a *non-monotone* effect — "models with over 20-25% code show gains on
HumanEval but penalties on language benchmarks. A moderate 15-25% code proportion appears to
balance these competing demands."**

**Does mixture matter more than architecture at this size? Evidence says no, and here is the
cleanest test.** arXiv 2603.22075 held data, steps, batch, sequence length, d_model, layers,
heads, FFN, optimizer and hardware *identical* and varied only the paradigm: 123.6M AR vs
162.7M MDLM on 50M TinyStories tokens. Best val loss 1.589 vs 3.412 (not cross-comparable —
different objectives), and the generation difference was a diversity/fluency trade-off, not a
capability gap. The paper's own conclusion is that the paradigms are *"complementary rather than
competing."* **Paradigm choice at 123M did not produce a capability cliff. Mixture effects
(REGMIX: up to 14.6% on a single task) are of the same order.** Neither is the dominant term.

What *is* likely dominant at 100M on 3-5B tokens: **token count and data quality.** PAPER
(SmolLM2, arXiv 2502.02737): SmolLM2-135M was trained on **2T tokens** and SmolLM2-360M on 4T.
Their own table shows HumanEval for the 1.7B model going 10.97 → 9.15 → 17.68 → 22.60 across
training stages, i.e. **HumanEval moves by ~12 points over the course of a single run** as data
quality changes. Our 3-5B tokens is **400-600x less data** than the 135M yardstick. PAPER
(arXiv 2511.03276, "super data learners") is the one genuinely encouraging result here: DLMs show
a **"Crossover"** in the data-constrained regime, and *"a 1B-parameter DLM achieves >56% accuracy
on HellaSwag and >33% on MMLU using only 1B tokens, without any special tricks, just by repeating
standard pre-training data"*, with a claimed **>3x data potential vs AR**. If that holds, a
3-5B token run is *less* hopeless for a DLM than for an AR model. It is also the claim I would
most want independently replicated, because it is the load-bearing optimism of the whole project.

**On the 50/35/15 mix itself: I found no evidence that this specific ratio is right, and one
reason to shift it.** The 50% code figure is well above the 15-25% band that 2503.03862 and
REGMIX both identify as the balance point, and above the 5-15% typical of frontier mixes.
SmolLM2's 135M used Stack-Edu + FineMath + InfiMM-WebMath from the start (they found the small
models *"benefited from a single-stage training approach with consistently high-quality data"*),
not a 50% code dump. Counter-argument in our favour: our target is HumanEval, and 2503.03862
confirms code fraction helps HumanEval up to 20-25% and then starts costing language benchmarks.
**Recommendation: 30-35% code, 45-50% web, 15-20% math, and spend the saved effort on filtering
rather than ratio-tuning.** REGMIX's finding that web corpora have *"the strongest positive
correlation with downstream performance"* (not Wikipedia, which is the intuitive choice) argues
for keeping web high.

**Contamination: the rules stay, and they need strengthening.** The spec says hold HumanEval out
and run n-gram overlap checks. Evidence that this is insufficient:

- PAPER (arXiv 2407.07565 / "On Leakage of Code Generation Evaluation Datasets"): **12.2% of
  HumanEval solutions are present in the Pile and 18.9% in the Stack; 20.8% of MBPP in the
  Stack.** And a controlled test: *"models perform significantly better when similar solutions
  are seen during training"* — StarCoderBase-15.5B scores 72% on the top-10%-similarity MBPP
  questions vs 22% on the bottom 10%. That is a **50-point spread from contamination alone.**
- PAPER (arXiv 2602.12413, "Soft Contamination", Nov 2026): semantic duplicates of **100% of
  MBPP's test set** and 76% of CodeForces are present in OLMo 3's post-training corpus.
  Critically: *"No single method removes more than 6.5% of them"* and *"all methods combined
  remove 10.87% (CodeForces) and 0.57% (MBPP)."* Every standard n-gram decontaminator is
  near-useless against paraphrase.
- Field note (dev.to, Oct 2026, 360M model from scratch): a dataset labelled "decontaminated"
  still contained **52 of HumanEval's 164 problems**; retraining on the filtered version *lost*
  4.3 points, and the losses were spread across both matched and unmatched problems — i.e. the
  gain was style transfer from HumanEval-shaped practice, not memorization. Their conclusion:
  *"Do not trust 'decontaminated' as a label."*

**Concrete rules for D1 (stricter than the spec):**
1. Decontaminate against HumanEval **and** MBPP (we currently only promise HumanEval). MBPP is
   our L1 metric, so an MBPP leak invalidates the pilot.
2. Use docstring-prefix matching **plus** solution n-grams **plus** an embedding-similarity pass.
   n-grams alone provably miss paraphrases.
3. **Run a seed control** (train twice, same data, different seed) so a 4-point effect can be
   separated from run-to-run variance. The field note above found MBPP run-to-run variance ~4
   points, the same size as the contamination effect.
4. Report pass@1 with a denominator and a seed, always. A single number from one run is not a
   measurement.

Kill criterion: n/a — this is hygiene, not an experiment. But: if contamination removal costs
more than ~2 points of HumanEval, we are removing real signal (style transfer) and should report
both numbers rather than the cleaner one.

---

### A6. Verification-during-refinement: queue it, but scope it down to a measurement first

**Verdict: not a trap, but the spec's version is too big and its premise is shaky at 100M.**

The spec's premise (line 126): *"Judging is easier than doing; a model can verify code far
better than it writes it."* **At 100M that premise is exactly backwards.** PAPER (arXiv 2412.02674,
"Mind the Gap", ICLR 2025): the generation-verification gap *"scales monotonically with the model
pre-training flops"*, and *"For small (in terms of pre-training flops) models such as Qwen-1.5 0.5B,
Qwen-2 0.5B and Llama-2 7B, gap(f) is non-positive for nearly all verification methods, even
though the models have non-trivial generation accuracy."* On 4x4 Sudoku, *"only the largest models
show non-trivial gaps"* (Qwen-2 72B: gap 16.99, relative 20.81%; 0.5B/1.5B/7B all ≈ 0 or
negative). The authors' reading: *"self-improvement requires a minimal level of instruction
following and reasoning capabilities, which is not present in these small models."*

Corroborating: PAPER (arXiv 2404.17140, "Small Language Models Need Strong Verifiers"):
*"the self-correction performance is largely bottlenecked by the verifier rather than the
refiner... a weak self-verifier can only bring minor improvements, if not misguiding the
refiner."* And ICLR 2026 work on tool-integrated verification for sLMs finds sLMs *"struggle
with verification tasks requiring memorization"* and need external tools to verify at all.

**So: at 100M, a self-verifier is likely to be as weak as the generator, and verification-based
self-improvement (spec recipe step 3, STaR/ReST) has a poor prior.** This matters because
step 3 is a load-bearing part of the recipe.

**But the diffusion-specific version is different, and the evidence is better.** In a dLLM, the
"verifier" does not have to be a separate model or a separate forward pass — the model's own
confidence at a masked position *is* a per-token quality signal, computed in the same forward
pass. And here there is direct evidence at our scale:

- PAPER (arXiv 2510.01384, PRISM, v4 May 2026): trains a lightweight adapter so the MDM's own
  forward pass emits per-token quality scores, then remasks low-quality tokens. Evaluated on
  **unconditional text at 170M** and on code with LLaDA-8B, with <500 GPU-hours of fine-tuning
  for the 8B case. **170M is our scale, and it is the one verification paper that reports at
  170M.**
- PAPER (CoDR, EMNLP 2026 Findings): training-free. Measures "confidence drift" — a token's
  confidence at commit time vs under later, denser context — and remasks where support dropped.
  *"improves average accuracy across all evaluated model–sampler configurations"*: +0.85 to
  +1.44 points on LLaDA-8B, +0.79 to +1.11 on Dream-7B, at 123-135 NFE. Training-free is
  exactly our budget.
- PAPER (T2M, arXiv 2604.18738): training-free, resets a suspect token to [MASK] rather than
  overwriting it, *"improves accuracy on tasks that require exact token-level output"*, +5.92 on
  CMATH, repairing 41.3% of "correct reasoning, garbled final answer" cases. That failure mode —
  right reasoning, corrupted last mile — is a very plausible description of what a 100M code model
  does.
- PAPER (PURE, ACL 2026 Findings): two-phase draft-then-refine, +12.9 on GSM8K for LLaDA-8B
  with a small refinement budget.
- PAPER (ProSeCo, NVIDIA, Feb 2026): trains unmasking *and* correction jointly, reports up to
  ~2-3x faster sampling and up to ~1.3x benchmark improvement from inference-time compute scaling.

**Ranked recommendation:** do **not** build the full "verify-during-refinement + adaptive steps"
mechanism after L1. Instead, in order:

1. **Free first:** add a confidence-based remask pass to `CustomBackend` (CoDR/T2M style,
   training-free, ~30 lines). This is a decode-policy change with no training, so it belongs with
   A1, not after L1. It is also the honest test of whether verification signal exists at all.
2. **Then:** PRISM-style adapter at 170M, since that is the only published 170M verification
   result and it is cheap.
3. **Only if 1-2 show signal:** the full learned verify-during-refinement + adaptive-steps
   mechanism.

Kill criterion for the whole line: if confidence-based remasking does not beat plain sharpened
sampling on MBPP pass@1, the model has no usable self-verification signal at 100M, the spec's
"judging is easier than doing" premise is false at our scale, and spec recipe step 3 (STaR/ReST
self-distillation) should be dropped from v1 rather than attempted.

---

## 3. Answers to the five questions, compressed

**1. The few-step quality cliff — which attacks work at small scale?**
The cliff is real and steep (2609.11998 Table 5: 15.26→6.44 for Attn, 8→32 steps), and
architecture barely changes it, so it is a training problem. Ranked by evidence-at-our-scale:
sharpened sampling (34M-170M, 2609.33947) → SDTT (169M, 2410.21035) → CDLM (110M, 2605.00161)
→ T3D (1.7B+, 2602.12262) → OPTD (8B only, 2608.02942). **The gap is confirmed and narrower than
the scout thought:** sub-200M few-step evidence exists (SDTT, FS-DFM, CDLM) but **no
trajectory/on-policy distillation paper has run below 1.7B, and none tested a 100-150M
block-diffusion model on code.** That is the fillable gap and it is D1's exact size.

**2. L1 looped core — still the right bet?**
The idea is right; the specified mechanism is wrong. Every paper loops *inside* a denoising step
(LoopMDM 2605.26106 at 170M; ALoDLM 2610.04198; Looped UDLM 2610.30066; Looped-DiT 2609.40305),
none ties loop count to step count. Change the treatment arm to early-middle-layer looping, and
change the metric from "equal TPS" to "equal total FLOPs" — looping inside a step costs TPS by
construction. Strongest counter-evidence: 2610.18127 (76-89% of tokens decided in iteration 1;
Ouro-Thinking best at iteration 1 and degrading after) and 2604.07822 (overthinking past ~15
iterations; grokking takes three stages). Both are AR recurrent-depth, not MDM, so they do not
dominate — but they are why the kill criterion must be compute-matched.

**3. Data mix — does it matter more than architecture at this size?**
No. At 123M with everything else held identical, paradigm choice produced a diversity/fluency
trade-off, not a capability cliff (2603.22075). Mixture effects are up to 14.6% on a single task
(REGMIX); architecture effects at this size are of the same order. Token count and data quality
are the bigger terms — the 135M yardstick used 2T tokens, we have 3-5B. Recommend shifting the
mix from 50% code toward 30-35% code (above 20-25% costs language benchmarks, 2503.03862), and
decontaminate against MBPP as well as HumanEval, with docstring-prefix + n-gram + embedding
passes and a seed control.

**4. Verification-during-refinement — queue or trap?**
Not a trap, but the spec's version is oversized and its premise is inverted at 100M. The
generation-verification gap scales *with* pretraining FLOPs and is non-positive for ≤7B models
(2412.02674), so a 100M self-verifier will likely be as weak as its generator. The diffusion
version is different and better supported at our scale: confidence is free (same forward pass),
and PRISM reports at 170M (2510.01384) while CoDR/T2M are training-free. Do the free
confidence-remask pass with A1; defer the learned mechanism.

**5. Anything newer (2026) that changes the plan?**
Yes, four things:
- **2609.33947** (Sep 2026): the few-step gap is largely a *sampler* artifact; τ=1 is generically
  wrong for parallel unmasking; sharpening at 34-170M rivals distilled successors. Directly
  changes what we do first.
- **2605.26106 / 2610.30066 / 2610.04198** (2026): looped MDM is established at 170M-8B, and
  always inside the step. Changes L1's design.
- **2610.17590** (Nov 2026): the "16x compute gap" for DLMs is overstated — **7.8x at block size
  32, 4.6x at block size 8, 3.0x at block size 4**, using exact likelihood + block diffusion +
  encoder-decoder. **This is directly actionable: it says our block size is a quality lever, and
  that block 32 is the expensive end.** If we can afford block 8, the pretraining cost gap drops
  from 7.8x to 4.6x. The spec fixed block size at 32 on the grounds that "LLaDA and the
  fixed-state paper both converge on 32" — that convergence is about *throughput*, and this paper
  says the *quality* cost of 32 is real and measurable.
- **2602.12413** (Nov 2026): semantic (paraphrase) contamination is pervasive and n-gram filters
  remove ≤6.5% of it. Strengthens the decontamination rules.

---

## 4. Top-5 ordered experiment list for a free T4

Each has a hypothesis, a metric, and a kill criterion. Ordered by (value ÷ cost). All run on a
free T4 or Kaggle T4. None requires a paid teacher.

---

### E1. Decoder sanity: sharpened sampling vs greedy, on the checkpoint we already have

- **Hypothesis.** The bracket-soup output in `results_t4.yaml` is a *sampler* artifact, not a
  model-capability limit. At τ=0.9 with fp32 Gumbel noise, the same smoldlm-144m checkpoint will
  emit coherent text and a non-degenerate MBPP pass@1. (PAPER basis: 2609.33947, 34M-170M MDLM;
  smoldlm's own model card says τ≥0.5.)
- **Method.** Add τ and Gumbel-precision knobs to `CustomBackend`. Sweep τ ∈ {0.7, 0.8, 0.9, 1.0}
  × steps ∈ {8, 16, 32}, 1k in / 1k out, batch 1, median of 5. Report TPS **and** tokens emitted
  **and** MBPP pass@1. Fix the SmolLM2-135M / LFM2.5 early-stop so the size-matched baseline is
  valid. Add a collapse screen: reject any τ whose mean unigram entropy falls below a floor.
- **Metric.** Primary: MBPP pass@1 at 8 steps, τ=0.9 vs τ=0 (greedy). Secondary: generative PPL
  (GPT-2 large) at matched steps; tokens emitted (must be ~1024 for all cells).
- **Kill criterion.** If τ=0.9 and greedy produce indistinguishable MBPP pass@1 **and** both are
  ~0, then the harness (mask id, EOS id, tokenizer, template) is the suspect, not the model —
  stop all quality research and debug the decode loop. If τ=0.9 does not beat greedy by at least
  2x on generative PPL at 8 steps, sharpening does not transfer to this model and A2/A3 carry the
  whole quality plan.
- **Cost.** ~2 T4-hours. No training.

---

### E2. Block-size sweep on the decode policy: is block 32 costing us quality we can buy back?

- **Hypothesis.** Block size 32 is a throughput choice, and it carries a real quality cost. The
  fixed-state paper's 8-step Gen-PPL of 12.10 and the compute-gap paper's 7.8x-vs-4.6x penalty at
  block 32 both point the same way. Smaller blocks should improve quality at equal step count,
  at a TPS cost we can measure rather than guess.
- **Method.** Using the E1 winner as the fixed decoder, sweep block size ∈ {8, 16, 32} × steps
  ∈ {4, 8, 16}, holding total tokens constant. Record TPS, tokens/pass, and MBPP pass@1. This is
  a pure decode-policy experiment — **no retraining** — because a block-diffusion model trained
  at block 32 can still be decoded at block 8 (the block-causal mask is a decode-time choice; note
  this is an approximation of the trained configuration, not exact, so treat the numbers as a
  lower bound on the benefit).
- **Metric.** MBPP pass@1 at fixed 8 steps as a function of block size; and the quality-per-TPS
  Pareto frontier over the (block, steps) grid.
- **Kill criterion.** If block 8 does not beat block 32 by ≥1.5 points of MBPP pass@1 at equal
  steps, block size is not a quality lever here and the spec's 32 stands. If block 8 costs more
  than 2x TPS for <1.5 points, the trade is not worth it at our target.
- **Cost.** ~4 T4-hours. No training.

---

### E3. Few-round training: does training-time objective alignment beat classical masking at 8 steps?

- **Hypothesis.** A model trained with a blockwise-aligned objective (clean frozen prefix, fully
  hidden suffix, loss only on the active block — arXiv 2508.19529) will beat an identically-sized
  and identically-trained model with classical full-sequence masking, when both are decoded at
  8 steps. And the gain should peak when training block size = decode block size.
- **Method.** Two 100M twins, same data, same token budget, same everything except the masking
  construction. Train at 8-step-equivalent block size. Evaluate at 8 and 32 steps.
- **Metric.** MBPP pass@1 at 8 steps (primary); generative PPL at 8 and 32 steps (to confirm the
  cliff is flatter, not just shifted); validation NLL (to confirm no quality was traded away at
  32 steps).
- **Kill criterion.** If the aligned twin does not beat classical masking by ≥2 points of MBPP
  pass@1 at 8 steps, the alignment claim does not hold at 100M. Keep classical masking and move
  the effort to E4.
- **Cost.** This is the first run that needs the pretrain pipeline. Two 100M runs on 3-5B tokens
  each is the expensive item in this document — budget accordingly, or run at 1B tokens each for
  the pilot and accept a weaker signal.
- **Note.** PAPER basis is LLaDA-8B-Instruct + LoRA (2508.19529 reports GSM8K 62.1 base), so the
  mechanism is scale-free but the magnitude is unmeasured below 8B.

---

### E4. Looped mid-block: does looping inside the denoise step beat an unlooped twin at matched FLOPs?

- **Hypothesis.** Looping the early-middle 2 layers *inside* each denoising step (LoopMDM's
  arrangement, arXiv 2605.26106, at 170M) will beat an unlooped parameter-matched twin on
  validation NLL at matched total training FLOPs, and will beat it by more at 8 decode steps than
  at 32.
- **Method.** Two 100M twins. (a) unlooped, 16 layers distinct. (b) 12 layers + 2-layer looped
  mid-block with S_max 6-12, same parameter count. **Match total training FLOPs and match
  inference FLOPs — not TPS.** Loop inside the step, not across steps. Evaluate at 8 and 16 decode
  steps.
- **Metric.** Validation NLL at matched FLOPs (primary — this is LoopMDM's own metric and the one
  their 3.3x claim rests on); MBPP pass@1 at 8 steps; GSM8K if the run is healthy enough.
- **Kill criterion.** If the looped twin does not beat the unlooped twin on validation NLL at
  matched total FLOPs, looping is dropped. **Explicitly: do not accept "equal TPS, equal quality"
  as a pass condition for this version — TPS is expected to fall, and the claim is about
  quality-per-FLOP.**
- **Cost.** Same order as E3; run only after E3 has produced a working pretrain pipeline.
- **Why this is not the spec's L1.** The spec ties loop iterations to denoising steps and claims
  TPS is untouched. The evidence (2605.26106, 2610.04198, 2610.30066, 2609.40305) all loops
  inside the step, which costs TPS. Keeping the spec's version would test a configuration nobody
  uses, and its "TPS untouched" framing would make a null result unreadable.

---

### E5. Confidence-remask pass: free verification signal, or none at 100M?

- **Hypothesis.** A dLLM's own per-position confidence is a usable quality signal at 100M even
  though a separate verifier would not be (the generation-verification gap is non-positive at
  ≤7B, arXiv 2412.02674). Remasking tokens whose confidence *drops* under later, denser context
  (CoDR, EMNLP 2026 Findings) will improve MBPP pass@1 at matched forward passes.
- **Method.** Training-free. Instrument the decode loop to record commit-time confidence per
  position, re-probe after each block, and remask positions whose support dropped beyond a
  threshold. Compare against E1's sharpened sampler at matched forward-pass count.
- **Metric.** MBPP pass@1 at matched NFE (primary); tokens repaired per block; and the fraction
  of repairs that survive to the final output.
- **Kill criterion.** If the remask pass does not beat sharpened sampling at matched NFE by ≥1.5
  points, the model has no usable self-verification signal at 100M. **Then drop spec recipe step
  3 (STaR/ReST self-distillation) from v1** — its premise is exactly this signal — and do not
  build the deferred verify-during-refinement knob at all.
- **Cost.** ~6 T4-hours, no training. Should be run alongside E1 since both are decode-only.

---

## 5. What I would cut from the current plan

Stated plainly, because a research note that only adds is not useful.

1. **The "loop iterations = denoising steps" framing of L1.** Superseded by LoopMDM/ALoDLM. Keep
   the twin-experiment structure; change the treatment and the metric.
2. **"TPS untouched by construction" as a claim for looping.** Only true for the superseded
   version. Looping inside a step costs TPS, and pretending otherwise will make the result
   unreadable.
3. **The assumption that pass@1 0.0-0.10 measures model quality.** Until E1 runs, it measures our
   decoder. This is the single most important correction.
4. **50% code in the pretrain mix.** Above the 15-25% balance point that two independent sources
   identify, and 3-5x the typical frontier fraction. Recommend 30-35%.
5. **Spec recipe step 3 (STaR/ReST self-distillation) as a v1 component.** Its premise —
   verification is easier than generation — is false at ≤7B by the primary source, and E5 is the
   cheap test of whether the diffusion-specific version survives at 100M. Do not build it before
   E5.
6. **"8-16 steps" as a settled range.** 2609.11998's table says 8 steps is ~2x the PPL of 32, and
   2610.17590 says the training cost of a block depends sharply on block size. The step count is a
   bet, not a fact, and E2 is how we price it.

---

## 6. Source ledger

Every claim above, with its scope. "Scope" = the model scale and hardware the evidence actually
covers. Where scope ≠ our setting (100M, T4), I say so.

| Source | Claim used | Scope of the evidence | Fits 100M/T4? |
|---|---|---|---|
| arXiv 2609.11998 (fixed-state) | Gen-PPL 15.26/12.10/9.22/6.44 for Attn at S=4/8/16/32; arch barely moves the cliff | 3B, 300B tokens, H100 | cliff shape yes; absolute values no |
| arXiv 2609.33947 | τ=.9+fp32 beats τ=1+fp64 by 3.5-4.9x gPPL at 16-1024 steps; τ=1 generically suboptimal under parallel unmasking; gains vanish for 8B models already on greedy | **34M and 170M MDLM**, OWT | **yes — closest match** |
| arXiv 2602.12262 (T3D, NeurIPS 2026) | 17.11→47.13 avg at TokPS=8 (LLaDA-8B); +175% relative | SDAR-1.7B/4B, LLaDA-8B | mechanism yes; magnitude no |
| arXiv 2608.02942 (OPTD) | On-policy transition distillation, 47.21 avg / 7.87 TPF; 0.67 pts over TAD-S at matched decoder | LLaDA-8B only (LoRA on TAD-S) | no — 8B-only setup |
| arXiv 2605.00161 (CDLM) | Multi-path discrete consistency, single-stage, teacher-free; 64-128x speedup vs MDLM | **110M models**, LM1B/OWT | **yes, scale matches** |
| arXiv 2410.21035 (SDTT) | Iterated self-distillation, 16-32x fewer steps | **169M / 424M / 863M** MDLM | **yes, scale matches** |
| arXiv 2509.20624 (FS-DFM) | 8 NFEs matches 1024-step DFM; 128x fewer steps | **0.169B** / 1.3B / 1.7B, 8xA100 | scale yes; pretrain cost no |
| arXiv 2508.19529 (Blockwise SFT) | +5.2 GSM8K / +1.6 MATH at matched FLOPs; peaks when train block = decode block | **LLaDA-8B-Instruct + LoRA only** (verified in full text) | mechanism yes; magnitude no |
| arXiv 2605.26106 (LoopMDM) | Up to 3.3x fewer training FLOPs at equal NLL; +8.5 GSM8K; loops early-middle layers inside each step | **170M MDM**, 3 corpora, matched total FLOPs | **yes — best L1 evidence** |
| arXiv 2610.04198 (ALoDLM) | Loop inside each denoise step, K=4, learned exit gate | 1.7B, 8B | mechanism yes |
| arXiv 2610.30066 (Looped UDLM) | 2.8x/3.3x fewer FLOPs at matched NLL; loops early layers within each step | UDLM, OWT/FineWeb-Edu | mechanism yes |
| arXiv 2609.40305 (Looped-DiT) | Loop depth beats extra denoise steps at fixed budget; 260M beats 6.5x larger | **vision**, 260M-1.7B | analogy only — not text |
| arXiv 2610.18127 (monitoring looped LMs) | 76-89% of tokens decided in iteration 1; Ouro-Thinking best at iter 1 then degrades | 8 looped LMs, 1-3.5B AR | counter-evidence, AR not MDM |
| arXiv 2604.07822 (Loop/Think/Generalize) | Grokking takes 3 stages; overthinking past ~15 iterations | small transformers, from scratch | counter-evidence |
| arXiv 2510.14961 | Recurrent-depth LM ≈ diffusion LM; up to 5x speedup on 3.5B, no tuning | 3.5B recurrent-depth | mechanism yes |
| arXiv 2610.00673 (Closing the Loop) | Looped beats parameter-matched dense on 12/12 benchmarks at 310B tokens; 7.7T→310B | 1.4B | scale gap; recipe useful |
| arXiv 2610.17590 | Compute gap 16x→7.8x (block 32), 4.6x (block 8), 3.0x (block 4); exact likelihood + block diffusion + enc-dec | ~1,700 models, 10 variants | **yes — block size is our lever** |
| arXiv 2603.22075 | AR vs MDLM at 123/163M, identical data/compute/hardware: diversity-fluency trade-off, not a capability cliff; MDLM still improving at 20k steps | **123.6M / 162.7M**, 50M tokens, H100 | **yes — closest paradigm test** |
| arXiv 2511.03276 (super data learners) | Crossover in data-constrained regime; >3x data potential; 1B DLM gets 56% HellaSwag on 1B tokens | 1B-8B, up to 1.5T tokens | encouraging but unreplicated |
| arXiv 2412.02674 (Mind the Gap) | GV-gap scales with pretraining FLOPs; non-positive for ≤7B; only 72B self-improves on Sudoku | Qwen-1.5/2 0.5B-72B, Llama-2/3.1 | **counter-evidence, directly on point** |
| arXiv 2404.17140 | Self-correction bottlenecked by the verifier, not the refiner | LLaMA-2-13B, Gemma-7B | counter-evidence |
| arXiv 2510.01384 (PRISM) | Adapter learns per-token quality; remask low-quality | **170M text** + LLaDA-8B, <500 GPU-h | **yes, scale matches** |
| CoDR (EMNLP 2026 Findings) | Confidence-drift remasking, training-free, +0.85-1.44 pts | LLaDA-8B, Dream-7B | mechanism yes, training-free |
| arXiv 2604.18738 (T2M) | Reset-to-mask beats overwrite; +5.92 CMATH | 8 benchmarks, LLaDA2.1 | mechanism yes |
| arXiv 2602.12413 (soft contamination) | Semantic duplicates of 100% of MBPP; n-gram filters remove ≤6.5% | OLMo 3 post-training corpus | hygiene rule |
| arXiv 2403.04811 | 12.2% HumanEval / 18.9% Stack / 20.8% MBPP contamination; 72% vs 22% top/bottom similarity | Pile, Stack, StarCoderBase | hygiene rule |
| arXiv 2503.03862 (Not-Just-Scaling-Laws) | Code >20-25% gains HumanEval but costs language; 15-25% is the balance | 92 open models | mix guidance |
| REGMIX (ICLR 2025) | Mixture moves single-task performance up to 14.6%; web > Wikipedia | 1M proxies → 1B/7B | mix guidance |
| arXiv 2504.11393 (DataDecide) | 150M ranking predicts 1B at ~80%; MBPP/HumanEval predictable with likelihood proxies | 4M-1B, 25 corpora | **supports small-scale proxy runs** |
| arXiv 2502.02737 (SmolLM2) | SmolLM2-135M trained on 2T tokens; HumanEval moves 10.97→22.60 across stages | 135M/360M/1.7B | **the yardstick's real budget** |
| arXiv 2607.02805 (BDLM Mamba-H) | 87M/350M hybrid block diffusion with partial bidirectionality; exact caching | **87M and 350M** | **yes — nearest architectural precedent** |
| arXiv 2609.04531 (PlaidQ) | 0.7B continuous diffusion; 4-step pass@10 7.90→15.85 | 0.7B, continuous not masked | nearby |
| arXiv 2606.19475 (DLM experimental analysis) | Denoising steps, context, block size, unmasking all move quality; block 8B/7B tested | 8 DLMs, 8 benchmarks | decode-policy guidance |

**Sources I could not verify and did not use:** the "LoopSpec 6.83x" figure and the preprints.org
looped-LM survey (I read the Parallax survey version instead, which is the same literature and
does state the compute-matched-control gap); the Qwen3-Next blog (empty fetch, per the prior
audit); the DFlash2 speedup numbers (carried from the spec, not re-verified this session).
"Trajectory self-distillation (NeurIPS 2026)" as a phrase resolves to T3D / arXiv 2602.12262 —
confirmed via the NeurIPS 2026 poster page and the arXiv record.

---

## 7. One-paragraph answer to the assigned question

At equal TPS, the thing that most raises quality for a 100M block-diffusion model on 3-5B tokens
is **not** architecture, not the loop, and not the data ratio. It is **the decoder and the
training objective, in that order.** The decoder first, because our only quality measurement is
currently taken through a sampler the model's own author says not to use, and the one paper that
measured this at our exact scale (34-170M, arXiv 2609.33947) found that fixing the sampler alone
recovers 2-3x on generative PPL and beats models that were explicitly distilled to be fast. The
training objective second, because the few-step cliff is real and steep (15.26→6.44 from 8 to 32
steps) and architecture does not move it, which makes it a training problem — and the cheapest
training fix is to train at the block size and step budget you will decode at (2508.19529).
Looping belongs in the plan, but in its published form: loop the early-middle layers *inside*
each denoising step, at 170M, where LoopMDM reports 3.3x fewer training FLOPs for equal NLL —
which, on a zero budget, is 3.3x more tokens, and that is worth more than any quality trick.
Data mix is real but second-order (up to 14.6% on a single task, same order as paradigm choice),
and our 3-5B tokens is 400-600x below the 135M yardstick's 2T, which is the number that should
actually worry us. Verification-during-refinement should be scoped down to a free
confidence-remask pass, because the general "verification is easier than generation" premise is
false below ~7B (2412.02674) and only the diffusion-specific, same-forward-pass version has
evidence at 170M (2510.01384). And the single most valuable thing this project can produce is the
measurement nobody has made: **few-step trajectory distillation on a 100-150M block-diffusion
model, evaluated on code.** Every published result in that family starts at 1.7B.