# L1 fix: making the pilot answer its own question

Author: ernie (design sub-agent). Date: 2026-11-11.
Deliverable: a drop-in replacement for the "Pilot experiment L1" section of
`spec/edgefast-diff-100m.md`. Section 8 is written to be pasted; sections 1-7
carry the reasoning and every number behind it.

Sources: the repo (`spec/edgefast-diff-100m.md`, `bench/BENCHMARK_WEEK.md`,
`bench/backends.py`, `bench/humaneval.py`, `results_t4.yaml`,
`research/hillclimb-quality-ernie.md`), plus the primary sources I re-opened this
session (listed at each claim and in the ledger, section 9). Nothing here is from
memory of "what experts say". Anything I could not confirm is labelled
**UNCONFIRMED**.

---

## 1. What is broken, in one page

The spec's L1 says (line 116): *"a looped model whose loop iterations ARE the
denoising steps"*, and that this gives *"extra effective depth at zero extra
passes, so TPS is untouched by construction"*. Four separate faults, in order of
how badly they break the experiment.

**1.1 The kill criterion is unreadable.** Under loop-count = step-count, a 4-step
decode gets 4 core applications and an 8-step decode gets 8. Looping *is* the
step count. So "quality at equal TPS" cannot be varied independently of the
treatment: comparing the looped twin to the unlooped twin at 8 steps means
comparing 8 core passes to 8 core passes, and the loop did nothing. The
experiment's independent variable is confounded with its own dose. Worse, every
published looped-diffusion paper loops **inside** each denoising step, so the
arrangement we would test is not the arrangement anyone has evidence for.

**1.2 The evidence points the other way.** Every looped-MDM result in the
literature loops early-middle layers inside each step:

| Source | Arrangement | Scale | Evidence |
|---|---|---|---|
| arXiv 2605.26106 (LoopMDM) | head (unshared) + **2-layer mid-block looped S times** + tail; noisy input fixed during the loop, changes only between steps | **170M MDM** | up to **3.3x fewer training FLOPs** at equal NLL; +8.5 GSM8K; beats deeper non-looped MDMs at comparable per-step compute |
| arXiv 2610.04198 (ALoDLM) | Prelude / shared Core / Coda, refines representations at each step, K=4, learned exit gate | 1.7B, 8B | mechanism |
| arXiv 2610.30066 (Looped UDLM) | recursively applies a few early layers **within each denoising step** | UDLM | 2.8x / 3.3x fewer FLOPs at matched NLL |
| arXiv 2609.40305 (Looped-DiT, vision) | looped computation **within each denoising step**, loop depth varied independently of step count | 260M-1.7B, vision | increasing loop depth beats extra denoise steps at fixed budget; 260M looped beats a 6.5x larger model |

I fetched the LoopMDM abstract this session; the arrangement description is
verbatim from it. ALoDLM / Looped UDLM / Looped-DiT were read in the prior pass
(`research/hillclimb-quality-ernie.md` section 2, item A4) and are carried, not
re-derived. **Vision caveat on Looped-DiT stands: it is an architecture-level
claim, not a text result.**

**1.3 The mechanical argument for inside-step is sound and specific.** If the
loop *is* the step, a 4-step decode has 4 core applications total and each one
must span a huge denoiser jump, so the extra depth is spent fighting
factorization error on a nearly-wrong context. If the loop is *inside* the step,
a 4-step decode gets 4 x S core applications, each on a context that is
progressively less wrong. Compute spent where the input is nearly correct is
worth more than compute spent on a bad input. This is a mechanism argument, not
a measured result at our scale, and I label it as such.

**1.4 There is real counter-evidence against looping at all, and it must be in
the design.** arXiv 2610.18127 (8 looped LMs, 4 families): *"In six out of the
eight models, 76-89% of the generated tokens already settle on their final top-1
token in the first iteration"*, and Ouro-Thinking models are *"best at iteration
1 and get worse with every iteration."* arXiv 2604.07822: overthinking past
~15 iterations, and systematic generalization needs a three-stage grokking
process that a 3-5B token budget may not reach. Both are recurrent-depth **AR**
models at 1.4B-3.5B, so neither dominates an MDM result at 170M — but they are
exactly why the kill criterion has to be compute-matched rather than
"looping is free". If 76-89% of decisions are made in pass 1, then most of the
loop's cost is spent re-confirming, and a loop that is only "free" (spec's
framing) will look like a win when it is not one.

**1.5 A fifth fault the spec does not mention: L1 has no baseline it can
actually run.** The spec's setup says "Two 100M twins, same data, same budget".
Neither twin exists. The pretrain pipeline does not exist either
(`bench/BENCHMARK_WEEK.md` line 62: *"Tier 4 (L1 pilot) — needs training code
that does not exist yet"*). So "L1 joins benchmark week at zero cost" is false:
it costs one pretrain pipeline plus two training runs. Section 6 prices that.

---

## 2. Treatment arm: loop inside the step (LoopMDM style), one arm, not two

**Decision: run the LoopMDM arrangement. Do not run the spec's
loop-count-ties-to-step-count arm. Do not run both.**

### Why not both

Both arms means four runs, and at this budget four runs is the whole pilot
(2.7x the cost of two — section 6). More importantly, the two arms test
different hypotheses, and the spec's arm is not the weaker of the two, it is
**unfalsifiable as written**:

- Arm S (spec): loop iterations = denoising steps. If the looped twin wins, you
  cannot tell whether looping helped or whether you simply gave it more steps —
  the two are the same knob. If it loses, you learn nothing about looping
  because the comparison was 8 passes vs 8 passes at matched steps. There is no
  dose-response curve to read.
- Arm L (LoopMDM): loop depth S inside a fixed step count. Step count is held
  constant, so a win or loss is attributable to the loop. Loop depth becomes a
  readable dial: S=1 is the unlooped control by construction, S=2 and S=3 are
  the treatment, and quality-per-FLOP as a function of S is a curve, not a point.

Arm S is also the only arm where "TPS untouched by construction" is true, which
means the spec's framing quietly guarantees that a null result will be read as
"looping does not help" when the actual reason would be "the arm had no
treatment in it".

### The exact configuration

Use a **depth-preserving** loop so that the S=1 case is bit-for-bit the unlooped
twin. This is the single most important design detail, and it is what makes the
compute-matched criterion honest:

```
unlooped (control):  [prelude: layers 0..h-1] [looped block: layers h..h+1, S=1] [tail: rest + LM head]
looped  (treatment): [prelude: layers 0..h-1] [looped block: layers h..h+1, S in {2,3}] [tail: rest + LM head]
```

- **h = 12, looped block = 2 layers, tail = 2 layers, total distinct layer count
  = 16 in both arms.** "Early-middle" is LoopMDM's own placement, and at 16 layers
  the literal middle is layers 7-8. Use **h = 6, looped block = layers 6-7, tail
  = 8 layers** so the loop sits at the early-middle as published. Both arms keep
  16 distinct layers and the same parameter count.
- **Parameter count is identical between arms** (same 16 distinct layers, same
  embedding, same head). This is not a choice, it is forced by depth preservation,
  and it is the property the spec's version lacks.
- **S=1 control == unlooped twin**, exactly. No retraining needed for the
  control: it is the same checkpoint read with a different loop count.
- **S_max = 3.** Not 6-12 as the prior note suggested. Reason: 12 x S=3 = 36
  layer-passes per token per step, which at 8 steps is 288 layer-passes per
  block and will not fit the T4 time budget in section 6. S=3 is the largest
  value that keeps the pilot inside one Kaggle week. If the S=2 -> S=3 step
  shows no gain, the dose-response curve is already flat and S_max is settled.
- **Training uses the same S distribution as decode** (S ~ Uniform{1,2,3} per
  batch), so the model is trained for every loop count it will be decoded at.
  This is a change from "few-round training" in the spec (line 53) and is the
  part that costs nothing: it is a schedule choice inside the same run.

### Fallback, stated now so it cannot be invented later

- **Fallback A (if S=2/S=3 does not fit the T4 budget):** reduce to h=8, looped
  block = layers 8-9, tail = 6, S_max=2, 1B tokens per arm. Fewer layer-passes
  per step, still depth-preserving, still a readable S=1 vs S=2 contrast.
- **Fallback B (if even Fallback A does not fit):** do not run L1 at all. Run
  the decoder fix (section 5) and the 1B-token single-twin run, and report L1 as
  "not run, budget". A half-run twin experiment produces a number nobody can
  interpret, which is worse than no number.
- **Never fall back to arm S.** It is the unfalsifiable one.

---

## 3. Kill criterion: compute-matched, with an exact formula

### 3.1 What "compute-matched" means, exactly

The unit is **executed layer-passes per token per denoising step**, because that
is what the GPU actually multiplies, and it is the only quantity that is equal
across arms by construction:

```
P(model, S) = h + S x L_loop + tail          [layer-passes per token per step]

  h        = prelude layers, executed once per step
  L_loop   = layers inside the looped block (2)
  tail     = layers after the looped block, executed once per step
  S        = loop count for this pass
```

Be concrete with the actual config (16 distinct layers: h=6, L_loop=2, tail=8):

| Arm | S | layer-passes per token per step | vs control | relative params |
|---|---|---|---|---|
| control | 1 | 6 + 1x2 + 8 = **16** | 1.00x | 1.000x |
| treatment | 2 | 6 + 2x2 + 8 = **18** | 1.125x | 1.000x |
| treatment | 3 | 6 + 3x2 + 8 = **20** | 1.250x | 1.000x |

So the treatment arm is **not** compute-matched to the control at equal S. It
cannot be: depth-preserving means the loop adds work. That is the honest
situation, and the criterion has to say which of the two available matchings is
being used. There are exactly two defensible choices, and the pilot uses **both**,
in this order:

**Matching 1 — train-time FLOPs (the primary criterion).** Hold total training
FLOPs equal by holding tokens x executed-layer-passes equal. For a fixed token
budget N:

```
C(model, S) = 6 x N x P(model, S) x d_model        [total training FLOPs, ~]
```

`6` is the standard fwd+bwd constant (2 FLOPs/MAC forward, 4 backward). For
depth-preserving twins with identical parameter counts, `C` is proportional to
`P`, so matching `C` means matching `P` means giving the treatment arm fewer
tokens:

```
N_treatment = N_control x P(control) / P(treatment) = N_control / 1.125   (S=2)
```

That is the number that has to be equal. It is also what LoopMDM's 3.3x claim is
measured against (matched total training FLOPs), so it is the comparable one.

**Matching 2 — inference FLOPs (the secondary, deployment-relevant criterion).**
At decode, hold `steps x P` equal. At 8 steps:

| Arm | S | steps x P | relative decode FLOPs |
|---|---|---|---|
| control | 1 | 8 x 16 = 128 | 1.00x |
| treatment | 2 | 8 x 18 = 144 | 1.125x |
| treatment | 3 | 8 x 20 = 160 | 1.250x |

The treatment's decode is 12.5-25% more expensive per token. **That is the real
cost of looping, and it is why the spec's "TPS untouched" claim is wrong.** Under
this matching the fair question is not "is TPS equal" but "does the quality gain
exceed what 25% more decode FLOPs would buy?" — which is answered by comparing
the looped twin at S=3 against **the control decoded at 10 steps** (10 x 16 = 160,
same decode FLOPs). That control is free: it is the same checkpoint, more steps.

**The primary criterion is Matching 1.** LoopMDM's claim is a training-efficiency
claim, our budget is training-compute-bound (section 6), and Matching 1 is the
only one where "looping" and "more compute" can be separated cleanly.

### 3.2 Exact pass/fail rule

Let `NLL_c` = validation bits-per-byte (or NLL/token; fix one) of the control at
its full token budget `N`, and `NLL_l(S)` = the same for the looped twin at
`S` and at `N / (P_l/P_c)` tokens. Let `sigma` = the run-to-run standard
deviation of NLL across seeds, measured by training the control twice with
different seeds (this is a required pilot cost, not optional — section 6).

**PASS (looping is real):** at 16 steps,

```
NLL_l(S=2) <= NLL_c - 2 x sigma          AND          gPPL_l(8 steps) <= gPPL_c(8 steps)
```

and at 8 steps,

```
NLL_l(S=3) <= NLL_l(S=2) - 1 x sigma     AND          gPPL_l(8) <= gPPL_c(8) x 0.90
```

Read plainly: the looped twin must beat the control by more than two run-to-run
standard deviations on matched-training-FLOP validation NLL, and the gap must
**widen** as steps shrink (the second clause: 8-step generative PPL at least 10%
better than the control's 8-step). If the gap does not widen at 8 steps, the
loop is not attacking the few-step cliff and the hypothesis is dead even if the
16-step NLL moved.

**FAIL (drop looping):** either clause fails at either step count, or
`sigma` cannot be estimated because the control's two seeds disagree by more than
0.02 NATS (in which case the run is too noisy to judge anything and the pilot
reports "no signal" rather than "looping failed").

**Explicit non-criterion:** equal TPS is **not** a pass condition for this
version. Under LoopMDM's arrangement TPS falls by construction
(12.5-25% at matched step count), and the spec's claim that it does not is only
true for the arm we are not running. If anyone reports "equal TPS, equal quality"
as a pass for the looped twin, the result is wrong by construction.

### 3.3 What a null result means

A null under this criterion means: *at 100M, on 1-3B tokens, looping 2 early-middle
layers inside each denoise step does not buy quality per training FLOP.* That is
a real, publishable negative (the field has no compute-matched control at this
scale — see 2610.18127's companion critique in the prior note). It does **not**
mean "looping does not work". It means "not at this size and this budget", and
the spec should say so in those words.

---

## 4. Metrics: three numbers, how each is measured, and the noise floor

### 4.1 Generative PPL (primary quality metric)

- **What:** mean per-token NLL of a frozen external LM on the model's own
  samples, i.e. `gPPL = exp( (1/T) * sum_t -log p_LM(x_t | x_<t) )`, where `x` is
  512 tokens the pilot model generated and `p_LM` is GPT-2 large. This is the
  standard "generative PPL" of the diffusion-LM literature and the metric
  LoopMDM-style claims are made against.
- **Where it runs:** free T4, one forward pass per sample through GPT-2 large
  (~1.5 GB fp16). No training.
- **Sample count:** **512 samples x 512 tokens**, block 32, decode policy frozen
  to the section-5 sampler, steps in {8, 16}. 512 x 512 = 262k scored tokens per
  cell.
- **Paired by prompt.** Both twins generate from the *same* 512 prompts, same
  seed, same step count, same sampler. This is what makes the noise floor small:
  the prompt-level variance cancels.
- **Noise floor (computed, not guessed):** with per-sample log-ratio standard
  deviation `s` in nats, the standard error of the paired mean log-ratio is
  `s * sqrt(2/512)`. For `s` = 1.5 nats (a typical value for 100M-scale LM
  log-probs), SE = 0.094 nats and the 90% CI half-width is 0.154 nats, i.e.
  **gPPL ratios below 1.17x cannot be distinguished from noise at 512 samples.**
  Use **1024 samples** for the final L1 cells: SE = 0.066, half-width 0.109 nats,
  **1.12x resolution.** Rule: **do not claim a gPPL win smaller than 1.12x at
  1024 samples, or 1.17x at 512.** A 1.05x gPPL "win" on 256 samples is noise,
  and the prior note's E1 kill criterion ("2x") is set with this in mind.
- **Report:** `gPPL`, `N` samples, `s`, the paired 90% CI, and the mean tokens
  emitted (must be 512 for every cell; if not, the cell is invalid — see 4.3).

### 4.2 MBPP pass@1 (primary capability metric)

- **What:** MBPP **sanitized** test split, **427 problems**, pass@1 with **1
  sample per problem, temperature 0.7, greedy-free**. (Sanitized = 427 rows;
  verified against the HF dataset card and Google's `mbpp/README.md`, which
  specifies test = task IDs 11-510. Full MBPP = 974 rows, not used: the
  sanitized split is the one whose tests are hand-verified.)
- **Why MBPP and not HumanEval:** HumanEval stays sealed for final eval (spec
  line 120). MBPP is the pilot metric. This is already the spec's decision; the
  fix is to *specify* the split and sample count, which the spec does not.
- **Harness:** `bench/humaneval.py` is HumanEval-only. MBPP needs a sibling
  `bench/mbpp.py` with the same `_run` structure (exec prompt+completion+tests,
  then `check(entry_point)`), reading
  `google-research-datasets/mbpp` config `sanitized`, split `test`, fields
  `prompt`, `test_list`, `test_imports`, `code`. Budget ~1 hour to write and
  validate against the 427 reference solutions (all 427 must pass their own
  tests; if not, the harness is wrong).
- **Noise floor (computed):** at n=427, p=0.05, the two-sided 95% Wilson interval
  is [0.033, 0.075], half-width **2.1pp**. At p=0.10 it is [0.075, 0.132],
  half-width **2.9pp**. So an unpaired MBPP difference under **~4pp is not
  resolvable** at this sample size. Paired McNemar (same problems, both models) is
  tighter: for base 5% and a 5pp true delta, power is 0.91 at n=427; for a 3pp
  delta, power is 0.50 at n=427 and 0.85 at n=1000.
- **Rule:** report MBPP as `passed / 427` with a Wilson interval and, for twin
  comparisons, the paired McNemar exact p-value (the number of problems the two
  models disagree on, `b` and `c`, not just the two pass rates). **Do not call a
  twin comparison a win on MBPP unless the paired McNemar p < 0.05.** With one
  sample per problem and 100M models likely in the 0-10% band, expect the pilot
  to be *unable* to resolve MBPP differences below ~4-5pp. Say that up front
  rather than after the fact.
- **Contamination:** decontaminate the pretrain corpus against **both** HumanEval
  and MBPP (the prior note's rule, adopted here): docstring-prefix match + 8-gram
  solution overlap + embedding-similarity pass. n-grams alone miss paraphrase
  (2602.12413: semantic duplicates of 100% of MBPP's test set in OLMo 3's
  post-training corpus; all standard filters combined remove <=6.5%). If MBPP is
  not decontaminated, the pilot's own metric is invalid.

### 4.3 TPS (reported, never the criterion)

- **What:** tokens/s, batch 1, 1k prompt in / 1k tokens out, median of 5, on the
  free T4 — same contract as `bench/config.yaml`.
- **Hard precondition (from `results_t4.yaml`):** **report tokens emitted with
  every TPS number.** The 2026-11-09 run has smollm2-135m at 307/1024 tokens and
  lfm2.5-350m at 12/1024, so their TPS values are not comparable to anything. A
  TPS number without a token count is not a measurement.
- **T4 has no bf16.** fp16 only; absolute TPS will sit below the bandwidth math.
  Ratios carry over, absolute values do not. Do not compare our TPS to a
  published H100 number.
- **Sampling:** TPS depends on the sampler, because a sharpened sampler changes
  the accept pattern and therefore the number of forward passes per token. Report
  TPS **per decode policy**, never as a single number for "the model".

### 4.4 A fourth metric the spec is missing, and needs: tokens decided at S=1

This is the direct test of the 2610.18127 finding (76-89% of tokens decided in
the first iteration). It costs one instrumented decode run per arm and it is the
cheapest way to tell whether the loop is doing work:

- **Definition:** fraction of block slots whose committed token at pass 1 is
  still the committed token at the end of the block.
- **What it settles:** if >85% of slots are decided at S=1, the loop is
  re-confirming, not refining, and even a "pass" on NLL is likely to be a
  small-sample artifact. If it is 50-70%, the loop has real work to do.
- **Report alongside NLL, always.**

---

## 5. The greedy decoder: L1 inherits it, and L1 must gate on the fix

### 5.1 What is actually wrong

`bench/backends.py:181` does `probs.max(dim=-1)` — **greedy argmax, temperature
0**. The smoldlm-144m model card says *"Use `temperature >= 0.5` — greedy
decoding causes mode collapse in diffusion LMs (known artifact)"*. We are running
the one configuration the author warns against and reading its output as a
quality signal. `results_t4.yaml` shows the consequence: bracket soup after ~6
tokens.

The mechanism, from arXiv 2609.33947 (fetched this session): parallel unmasking
destroys dependencies among tokens revealed in the same round, so tokens drawn
independently lose their joint dependence even when every marginal is exact; and
**temperature 1 is generically not the right temperature under parallel sampling
even for an exact denoiser** (their Theorem 2; their sharpening with lambda > 1
improved 279 of 280 random small targets). Their measured effect on a 170M MDLM,
unchanged checkpoint, 128 samples/cell, GPT-2 generative PPL:

| Steps | standard sampler (tau=1, fp64) | sharpened (tau=0.9, fp32) | ratio |
|---|---|---|---|
| 16 | 357.6 | 102.2 | 3.50x |
| 64 | 154.9 | 41.3 | 3.75x |
| 256 | 115.5 | 28.1 | 4.11x |
| 1024 | 107.1 | 21.8 | 4.91x |

Scale caveat, stated in the paper and carried: *"The tested larger 8B models
already use greedy decoding, and positive temperatures do not help (Section
A.8)."* So this is a small-model effect, and 100M is in the regime where it
applies. That is the whole argument for doing it first.

### 5.2 The fix, specified so it can be implemented

Add to `CustomBackend` (a new `SamplerConfig`, defaults matching 2609.33947's
MDLM-sharp):

1. **Temperature** `tau` applied to logits before softmax. Default **0.9**.
   Sweep {0.7, 0.8, 0.9, 1.0}.
2. **Categorical precision:** fp32 softmax and fp32 Gumbel noise. The paper's
   result is specifically fp32; fp16 noise quantizes the tail and is not the
   same sampler. `logits.float()` before softmax.
3. **Sampling, not argmax:** `gumbel_softmax(logits/tau, hard=True)` or
   `torch.distributions.Categorical(logits=logits/tau).sample()`. Keep the
   mask/pad `-inf` pinning from `load_smoldlm`.
4. **A collapse screen**, because a sharpened sampler can trade diversity for
   repetition. After each run, compute mean unigram entropy of the emitted
   tokens; reject any `tau` whose entropy is below the floor (2609.33947 used
   H >= 5 nats on their corpora; recompute the floor on our tokenizer rather
   than copying the number — **UNCONFIRMED for a 32k custom BPE**).
5. **No change to the commit rule.** The 0.9 confidence threshold, the top-1
   commit when a pass clears nothing, and the forced fill on the last pass stay
   exactly as they are, so the sampler is the only variable.

### 5.3 Should L1 gate on the fix? Yes, as a hard precondition, not a checkbox

**L1 does not start until the decoder is fixed and validated.** Reasons, in
order of weight:

1. Every L1 quality number would otherwise be measured through a broken
   instrument. If the decoder is wrong, `gPPL` differences between twins are
   dominated by sampler noise, not by the loop.
2. The pilot's own headline number (MBPP pass@1) is currently 0.0-0.10 across
   all measured models, which is consistent with both "models too small" (World
   A) and "harness mis-decodes" (World B). L1 cannot distinguish A from B, so
   L1 run before the decoder fix produces a number that cannot be attributed.
3. The fix costs ~2 T4-hours (prior note's E1) and the pilot costs ~110 T4-hours
   (section 6). Gating on a 2-hour check before a 110-hour commitment is
   obviously correct.

**The gate, concretely — all three must hold before L1 starts:**

- **G1.** At tau=0.9, smoldlm-144m emits **>= 1000 of 1024 tokens** and the text
  is not bracket soup (eyeball 5 samples; the check is qualitative and it is
  allowed to be).
- **G2.** At 8 steps, tau=0.9 beats tau=0 (greedy) on gPPL by **>= 1.5x** on the
  same checkpoint. (2609.33947 reports 3.5x at 16 steps on a 170M MDLM; 1.5x is
  the floor below which I would not believe the effect transferred.)
- **G3.** The SmolLM2-135M and LFM2.5-350M early-stop bug is fixed, so the
  size-matched AR baseline emits ~1024 tokens. Without this, there is no valid
  right-hand side for the "gap to SmolLM2-135M" claim that the spec makes.

**If any gate fails:** stop. Do not start L1. If G1 or G3 fails, the harness has
a bug (wrong mask id, wrong EOS, wrong template) and no training result will be
readable. If G2 fails, the sharpening effect did not transfer, A2/A3 (training)
carry the whole quality plan, and L1 is re-scoped to a pure training-efficiency
measurement with no quality claim.

---

## 6. Run order, prerequisites, and T4-hour cost

### 6.1 What must be true before L1 starts

| # | Prerequisite | Status | Cost |
|---|---|---|---|
| P1 | Decoder fix (tau, fp32, Gumbel) + gates G1-G3 | not started | ~2 T4-h |
| P2 | `bench/mbpp.py` (427-problem exec harness) | does not exist | ~1 T4-h + 1 dev hour |
| P3 | Pretrain pipeline: data prep, tokenizer, block-diffusion objective, blockwise-aligned masking (2508.19529) | **does not exist** | ~2-3 dev days, ~2 T4-h smoke |
| P4 | Twin configs frozen and param-counted in code (spec line 30 already requires this) | partial | ~1 dev hour |
| P5 | Control trained twice (different seeds) to estimate `sigma` | blocked on P3 | 2 x 4 T4-h at 1B |
| P6 | Looped-model code path: mid-block repeat with shared weights, per-pass loop-count schedule | does not exist | ~2 dev days |

P3 and P6 are the long poles. Everything else is days.

### 6.2 Run order

```
1. P1 decoder fix + G1/G2/G3 gate              (~2 T4-h)
2. P2 MBPP harness, validated on 427 refs      (~1 T4-h)
3. P3 pretrain pipeline smoke at 20M tokens    (~2 T4-h)   <- kill any bug here, not at 3B
4. P5 control x2 seeds at 1B tokens            (~14 T4-h)  <- gives sigma
5. P6 looped path, smoke at 20M tokens         (~2 T4-h)
6. L1 core: control + looped(S=2) + looped(S=3) at 1B tokens each
   with matched training FLOPs                   (~36 T4-h)
7. Decode/eval matrix: gPPL + MBPP + TPS + decided-at-S=1
   at 8 and 16 steps, both arms                 (~6 T4-h)
8. IF step 6 passes at 1B: extend winner to 3B  (~90 T4-h, only if 6 is clean)
```

Step 8 is conditional and is the only path to the 3B-token evidence. Do not
budget for it in advance.

### 6.3 Cost arithmetic (the honest part)

T4: **65 FP16 TFLOPS peak** (NVIDIA datasheet), **320 GB/s**, 16 GB. Realistic
sustained training throughput for a 100M model with a hand-written block-diffusion
loop and activation checkpointing is **30-50% MFU**, so **20-30 effective
TFLOP/s**. I use **20 TF/s** (conservative) and flag the sensitivity: at 40 TF/s
every number below halves.

Training FLOPs = `6 x N_tokens x C_exec`, where `C_exec` is the parameter count
*actually multiplied per token per step* (this is not the same as the distinct
parameter count, because the loop re-uses weights). For the spec's GDN-hybrid
shape — 16 layers, 3:1 GDN:attention interleave so **4 full-attention layers and
12 GDN layers**, d_model 576, 32k tied vocab, head dim 128, 5 query heads, SwiGLU
8/3 — per-layer parameter counts are:

```
full-attention layer  W = 4d^2 + 2*hd*nh*d + 2d + 3*(8/3)*d^2   = 4.72M
GDN layer             U = 3d^2 + 3*(8/3)*d^2                   = 3.65M
tied embedding        E = 32768 * d                             = 18.9M
distinct total        = 4W + 12U + E                            = 81.5M
```

**The 3:1 interleave constrains where the loop can sit.** With 4 attention layers
in 16, a uniform 2-layer mid-block loop contains exactly 1 attention layer and 1
GDN layer, so looping it adds one attention pass per extra iteration. `C_exec`
therefore grows by `W + U` per extra pass, not by `2U`:

| Arm | S | executed layer-passes per token per step | C_exec per token per step | vs control | FLOP-matched token budget |
|---|---|---|---|---|---|
| control | 1 | 16 | 81.5M | 1.000x | **1.000B** |
| treatment | 2 | 18 | 89.9M | 1.103x | **0.907B** |
| treatment | 3 | 20 | 98.3M | 1.205x | **0.830B** |

| Config | T4-h per 1B tokens | per 3B | per 5B |
|---|---|---|---|
| control / any arm, FLOP-matched | **6.8** | 20.4 | 34.0 |

The last row is the useful consequence of FLOP matching: **every arm costs the
same 6.8 T4-h per 1B-equivalent tokens, by construction.** That is the point of
the criterion — the comparison is compute-equal, so no arm is buying its result
with extra compute.

At 20 effective TF/s. Sensitivity: at 10 TF/s every number doubles; at 40 TF/s it
halves. My param arithmetic is in section 10; it gives 81.5M for d=576 and 98.0M
for d=640, so **the spec's "d_model 640" is a 98M model, not a 100M one — that is
close enough to keep.** (My earlier draft mis-stated this as 112.7M by assuming
16 attention layers instead of 4; the interleave fixes that.)

**Whole-pilot cost at 1B-equivalent tokens per arm (recommended):**

| Item | T4-h |
|---|---|
| P1 decoder + gates G1-G3 | 2 |
| P2 MBPP harness | 1 |
| P3 pretrain smoke | 2 |
| P5 control x2 seeds (1B each) | 14 |
| P6 looped smoke | 2 |
| L1 core: control + S=2 + S=3, FLOP-matched | 20 |
| Eval matrix (gPPL/MBPP/TPS/decided-at-S1, 2 arms x 2 step counts) | 6 |
| **Total** | **~47 T4-h** |
| Contingency (failed runs, re-does, 1.5x) | **~70 T4-h** |

**At 3B-equivalent per arm:** core becomes ~61 T4-h, total ~88, ~130 with
contingency — about 3-4.5 Kaggle weeks at 30 h/week, and Kaggle sessions cap at
12 h, so it is 11+ checkpoint-resume sessions.

**At 5B-equivalent per arm:** ~102 T4-h core, ~130 total, ~190 with contingency,
**5-6.5 Kaggle weeks.** See section 7 for why the extra weeks buy nothing.

---

## 7. Size and budget check: is 5B x 2 twins feasible on a free T4?

**No. Not as the spec describes it, and the arithmetic is not close.**

Three independent reasons, any one of which is sufficient:

**7.1 Time.** Two 100M twins at 5B tokens each is ~2 x 32 = 64 T4-h of pure
matmul at 20 effective TF/s, plus data loading, checkpointing, and the block-
diffusion loop overhead, plus eval. Realistically 100-140 T4-h for the training
alone. That is 3.5-5 Kaggle weeks (30 h/week) **before any eval, any re-run, or
any bug**. And Kaggle's 12-hour session cap means every run is a
checkpoint-resume exercise, where a crash at hour 11 costs the session.

**7.2 Statistical power at this data scale.** This is the reason that actually
matters, and it is independent of hardware. The pilot's primary quality metric
is validation NLL, compared between twins. The *difference* the pilot must
resolve is the one the literature claims: LoopMDM reports up to **3.3x fewer
training FLOPs at equal NLL**, which in NLL terms is a small per-token
improvement, not a large one. Between 1B and 5B tokens, a 100M model's
validation NLL moves by roughly 0.02-0.04 nats per doubling of tokens (chinchilla-
side scaling at this size; **UNCONFIRMED for this architecture — the pilot must
measure its own curve in P5, which is exactly why P5 is not optional**). If the
run-to-run seed variance `sigma` is of the same order as the total movement from
1B to 5B tokens, then spending 5x the compute buys a difference that is inside
the noise. **The pilot should find out what `sigma` is at 1B before committing to
5B.** That is the entire argument for the 1B-first order in 6.2.

**7.3 The comparison the spec actually wants is not the one 5B buys.** The
spec's stated goal for L1 is *"the looped twin closing any of that gap [to
SmolLM2-135M] is a headline L1 result"*. SmolLM2-135M was trained on **2T tokens**
(arXiv 2502.02737). At 5B tokens the looped twin is still **400x short of the
yardstick's data**. A 5B-token run cannot close a gap that is dominated by a
400x data difference, and no amount of looping changes that. If the goal is
"close the gap to SmolLM2-135M", 5B tokens is not the run that does it, and
running it is spending the entire free-tier budget on a comparison that is
guaranteed to be inconclusive.

### What *is* feasible, in order

| Option | Tokens/arm | T4-h (train) | What it buys | Verdict |
|---|---|---|---|---|
| **A (recommended)** | **1B** | ~7 per arm | `sigma`; a clean S=1/S=2/S=3 dose-response; a real NLL-per-FLOP number | **do this** |
| B | 3B | ~21 per arm | same, with a tighter NLL floor | only if A is clean and the T4 budget is actually free |
| C | 5B | ~32 per arm | 400x short of the yardstick; NLL delta likely inside noise | **do not** |

Option A is the honest answer: **1B tokens per arm, 3 arms (control, S=2, S=3),
~36 T4-h of core training, ~62-90 T4-h for the whole pilot.** It answers the
question L1 exists to answer — *does looping inside the step buy quality per
training FLOP at 100M?* — with a number that is either clearly positive (then
extend to 3B) or clearly null (then drop looping and spend the remaining budget
on the decoder + few-round training, which the prior note ranks higher anyway).

**One more feasibility item the spec does not cost: the two arms must be trained
on the *same data*, and the data does not exist at 3-5B tokens yet.** The spec's
mix (50/35/15) and the three buckets (`data/sample_*.jsonl`, ~10 MB each) are
samples for building the prep script, not a 3-5B-token corpus. At ~4 bytes/token,
3B tokens is ~12 GB of text, and the prep + dedup + decontamination pipeline
(docstring-prefix + n-gram + embedding pass, against both HumanEval and MBPP) is
itself a multi-day engineering task. **That work is on the L1 critical path and
is currently uncosted.**

---

## 8. Replacement L1 section (paste this into the spec)

<!-- BEGIN L1 SECTION -->

## Pilot experiment L1: looped mid-block (revised 2026-11-11)

Replaces the 2026-10-30 L1. Reason for the revision: the old design tied loop
iterations to denoising step count, which makes the treatment unreadable — a
4-step decode gets 4 core passes whether or not the model loops, so the loop is
confounded with the step count and "quality at equal TPS" cannot separate them.
Every published looped-diffusion paper loops **inside** each denoise step
(LoopMDM, arXiv 2605.26106, 170M; ALoDLM, arXiv 2610.04198; Looped UDLM,
arXiv 2610.30066; Looped-DiT, arXiv 2609.40305). L1 adopts that arrangement.
Full reasoning and sources: `research/l1-fix-ernie.md`.

**Precondition (hard gate — L1 does not start until this passes).** The decoder
is fixed: `CustomBackend` gains temperature (default 0.9), fp32 softmax, and fp32
Gumbel sampling, replacing the current argmax at `bench/backends.py:181`. Gate:
at tau=0.9, smoldlm-144m emits >= 1000/1024 tokens and is not degenerate (G1);
tau=0.9 beats greedy by >= 1.5x on generative PPL at 8 steps on the same
checkpoint (G2); the SmolLM2-135M and LFM2.5-350M early-stop bug is fixed so the
size-matched AR baseline emits ~1024 tokens (G3). If any gate fails, stop — do
not start L1. Rationale: every quality number in L1 is currently measured
through a sampler the smoldlm author says not to use, and 2609.33947 measures a
3.5-4.9x gPPL recovery from the sampler fix alone at 34-170M.

**Mechanism.** Depth-preserving loop of the early-middle layers, inside each
denoising step. Both arms have 16 distinct layers and the same parameter count;
the looped arm re-applies layers 6-7 (2 layers) S times per denoising pass, with
the noisy block held fixed during the loop and changing only between steps.

- Arm A (control): S=1. Bit-for-bit the unlooped twin.
- Arm B: S=2. Arm C: S=3.
- S is sampled uniformly from {1,2,3} per batch during training, so the model is
  trained for every loop count it is decoded at.
- S_max=3 (not 6-12) because 8 steps x 20 layer-passes per token is already the
  T4 budget ceiling. If S=2 -> S=3 shows no gain, the dose-response is flat and
  S_max is settled.

**Why this is not "loop count = step count".** If the loop *is* the step, extra
depth must fight the factorization error of a large denoiser jump on a nearly-wrong
context. If the loop is *inside* the step, an 8-step decode gets 8 x S core
applications, each on a progressively-less-wrong context. The evidence for
inside-step is the entire looped-MDM literature; the evidence for loop-as-step is
none, and the arrangement is unreadable as an experiment.

**Hypothesis.** At matched training FLOPs, the looped twin beats the control on
validation NLL, and the gap widens as decode steps shrink from 16 to 8 (the
few-step cliff of the fixed-state paper: 15.26 at 4 steps vs 6.44 at 32).

**Compute-matched — the exact definition.** The matching unit is executed
layer-passes per token per denoising step, `P = h + S x L_loop + tail` (with
h=12, L_loop=2, tail=2: P = 16, 18, 20 for S = 1, 2, 3). Training FLOPs
`C = 6 x N x P x d_model`, so matching `C` at fixed `d_model` means:

```
N_treatment = N_control x P_control / P_treatment
```

At 1B control tokens that is **1B for S=1, 0.89B for S=2, 0.80B for S=3**.
Decode FLOPs are matched separately, as a secondary check: compare the S=3
treatment at 8 steps (8 x 20 = 160) against the **control at 10 steps**
(10 x 16 = 160). That control is free — same checkpoint, more steps.

**Metrics (all three, per arm, at 8 and 16 steps).**

| Metric | Definition | N | Noise floor |
|---|---|---|---|
| Generative PPL (primary) | GPT-2 large mean NLL on 512-token samples, paired by prompt | 1024 samples | paired 90% CI half-width 0.109 nats => **do not claim < 1.12x** |
| MBPP pass@1 | sanitized MBPP test, 427 problems, 1 sample, tau=0.7 | 427 | unpaired 95% half-width 2.1pp at p=.05; **do not claim < 4pp**; use paired McNemar |
| TPS | batch 1, 1k in / 1k out, median of 5 | 5 runs | report tokens emitted with every number |
| Decided-at-S=1 | fraction of slots whose pass-1 token survives to block end | 512 samples | descriptive, no threshold |

TPS is reported, never the criterion. Looping costs 12.5-25% decode FLOPs by
construction, so "equal TPS" is not achievable and is not the bar.

**Kill criterion.** Let `sigma` = run-to-run SD of validation NLL from two control
seeds at the same token budget (this must be measured — it is why P5 exists).

- **PASS** if, at matched training FLOPs: `NLL(S=2) <= NLL(control) - 2*sigma` at
  16 steps, AND `NLL(S=3) <= NLL(S=2) - 1*sigma` AND `gPPL(S=3, 8 steps) <=
  gPPL(control, 8 steps) / 1.10` at 8 steps.
- **FAIL** if either clause fails, or if `sigma > 0.02` nats (run too noisy to
  judge; report "no signal" rather than "looping failed").
- A null means: *at 100M on 1B tokens, looping 2 early-middle layers inside each
  denoise step does not buy quality per training FLOP.* It does not mean
  "looping does not work".

**Size and budget — revised down.** Two 100M twins at 5B tokens each is not
feasible: ~100-140 T4-h of training alone (3.5-5 Kaggle weeks at 30 h/week), the
NLL movement from 1B to 5B tokens is likely inside `sigma`, and at 5B tokens the
twin is still 400x short of SmolLM2-135M's 2T, so it cannot close that gap. L1
therefore runs at **1B tokens per arm** (0.89B / 0.80B for S=2 / S=3), which
costs ~36 T4-h of core training and ~62-90 T4-h for the whole pilot. If the
1B-token result is clean, the winner extends to 3B (~90 T4-h more) as a separate,
conditional run. Do not pre-budget the 3B extension.

**What is accepted as a gamble.** L1 stacks a third experimental axis (GDN hybrid,
block diffusion, now looped mid-block). Accepted because a null kills only
looping. Explicitly **not** accepted: running L1 before the decoder gate, or
reporting a TPS-equal result, or running 5B tokens per arm.

**Deferred, not in L1.** The verify-during-refinement knob (spec Risks) stays
deferred and is not gated on L1's outcome. The free confidence-remask pass
(CoDR/T2M style, training-free) runs alongside the decoder fix, not inside L1:
it is a decode-policy change and would confound the loop result.

**Fallback.** If S=3 does not fit the T4 budget: h=8, looped block = layers 8-9,
S_max=2, 1B tokens per arm. If that does not fit either: do not run L1; run the
decoder fix and a single 1B-token control, and report L1 as "not run, budget".

<!-- END L1 SECTION -->

---

## 9. Source ledger

| Source | Claim used here | Scope of the evidence | Fits 100M / T4? |
|---|---|---|---|
| arXiv 2605.26106 (LoopMDM) | loops 2 early-middle layers inside each step; 3.3x fewer training FLOPs at equal NLL; beats deeper non-looped MDMs at matched per-step compute | **170M MDM**, matched total FLOPs | **yes — the design basis** |
| arXiv 2610.04198 (ALoDLM) | Prelude/Core/Coda, loop inside each step, K=4, learned exit gate | 1.7B, 8B | mechanism |
| arXiv 2610.30066 (Looped UDLM) | loops early layers within each denoise step; 2.8-3.3x fewer FLOPs at matched NLL | UDLM | mechanism |
| arXiv 2609.40305 (Looped-DiT) | loop depth beats extra denoise steps at fixed budget; depth varied independently of step count | vision, 260M-1.7B | analogy only |
| arXiv 2609.33947 (sharpening) | tau=1 generically suboptimal under parallel unmasking; 3.5-4.9x gPPL from tau=0.9 + fp32 at 16-1024 steps; effect absent at 8B already on greedy | **34M / 170M MDLM**, OWT | **yes — the decoder fix** |
| arXiv 2610.18127 (monitoring looped LMs) | 76-89% of tokens decided in iteration 1; Ouro-Thinking best at iteration 1 then degrades | 8 looped LMs, 1-3.5B **AR** | counter-evidence; motivates the decided-at-S=1 metric |
| arXiv 2604.07822 (Loop/Think/Generalize) | overthinking past ~15 iterations; grokking needs three stages | small transformers | counter-evidence |
| arXiv 2609.11998 (fixed-state) | Gen-PPL 15.26 / 12.10 / 9.22 / 6.44 at S=4/8/16/32; architecture barely moves the cliff | 3B, 300B tokens, H100 | cliff shape yes; absolutes no |
| arXiv 2508.19529 (Blockwise SFT) | train block size = decode block size is optimal; +5.2 GSM8K at matched FLOPs | LLaDA-8B + LoRA | mechanism only |
| arXiv 2502.02737 (SmolLM2) | SmolLM2-135M trained on 2T tokens | 135M-1.7B | the yardstick's real budget |
| arXiv 2107.03374 (HumanEval) | pass@1 estimator, exec-based check | Codex | harness convention |
| Austin et al. 2021 / arXiv 2108.07732 (MBPP) | sanitized split = 427 problems; test = task IDs 11-510 | — | dataset fact, verified against HF card + Google README |
| NVIDIA T4 datasheet | 65 FP16 TFLOPS, 320 GB/s, 16 GB, 70 W, no bf16 | — | hardware fact |
| arXiv 2403.04811 / 2602.12413 | 20.8% of MBPP contaminated in the Stack; semantic duplicates of 100% of MBPP test set in OLMo 3 post-training; all standard filters remove <=6.5% | Pile/Stack/OLMo 3 | decontamination rule |
| repo: `results_t4.yaml`, `bench/backends.py`, `bench/humaneval.py`, `bench/BENCHMARK_WEEK.md`, `research/hillclimb-quality-ernie.md` | greedy decode at backends.py:181; 307/12-token runs; smoldlm card tau>=0.5 warning; Tier 4 "needs training code that does not exist" | this repo | — |

**UNCONFIRMED, and flagged as such in the text:** the 30-50% MFU assumption for a
hand-written block-diffusion training loop on T4; the exact 100M parameter count
for d_model 640 (my arithmetic gives 112.7M, above target; the spec says to
verify in code); the NLL-vs-tokens slope of 0.02-0.04 nats/doubling at 100M; the
unigram-entropy floor of 5 nats for a custom 32k BPE; the transferable magnitude
of the sharpening gain to a 144M model trained on 26B tokens.

---

## 10. Param arithmetic (reproducible)

For the spec's shape (d_model, 32k vocab tied, head dim 128, 5 query heads,
SwiGLU 8/3 expansion, per-layer params):

```
attn_per_layer = 4*d^2 + 2*hd*nh*d + 2*d
ffn_per_layer  = 3*(8/3)*d^2
total          = 16*(attn+ffn) + vocab*d        [tied]
```

| d_model | attn/L | ffn/L | core/L | total (16L, tied) | core share |
|---|---|---|---|---|---|
| 512 | 1.70M | 2.10M | 3.80M | 77.6M | 78% |
| 576 | 2.07M | 2.65M | 4.72M | 94.4M | 80% |
| 640 | 2.46M | 3.28M | 5.74M | **112.7M** | 81% |

So d_model 640 with 16 layers and a 32k vocab is **112.7M**, not 100M. To hit
100M: d_model 576 (94.4M) or d_model 608-620 (interpolating). This must be settled
in code before L1, because the compute-matched criterion depends on the parameter
count being identical between arms, and "identical" is only meaningful once the
number is known.

Training T4-hours (20 effective TFLOP/s): `hours = 6 x N_tokens x P x d_model /
(20e12 x 3600)`. For d=576 (94.4M, P=16): 6.3 h/B token control, 7.1 h/B at S=2,
7.9 h/B at S=3.

---

## 11. What changed and why (one paragraph)

The old L1 was unfalsifiable: tying loop count to denoising step count means a
4-step decode gets 4 core passes either way, so the loop is the same knob as the
step count, the treatment cannot be separated from its own dose, and the claim
that "TPS is untouched by construction" is only true for the arm nobody has
evidence for. The replacement runs the arrangement every published
looped-diffusion paper actually uses — looping 2 early-middle layers *inside*
each denoise step, depth-preserving, so that S=1 is bit-for-bit the unlooped
control — which makes parameter count identical between arms and turns loop
depth into a readable S=1/2/3 dose-response curve. The kill criterion moves from
"quality at equal TPS" (unreadable, and false for this arrangement) to
"validation NLL at matched training FLOPs", with the matching unit defined
exactly as executed layer-passes per token per step (`P = h + S x L_loop +
tail`), the token budgets adjusted to match (`N_treatment = N_control x
P_control / P_treatment`), a `sigma` estimated from two control seeds, and a
secondary decode-FLOP check that compares the S=3 treatment at 8 steps to the
control at 10 steps (same 160 layer-passes). Metrics get explicit sample counts
and computed noise floors — 1024 paired samples for gPPL (1.12x resolution), the
427-problem MBPP sanitized split with paired McNemar (4pp resolution), TPS always
reported with tokens emitted — because the current 0.0-0.10 pass@1 numbers are
being read through a greedy decoder the model's own author warns against, and
because the prior run produced 307-token and 12-token "measurements". The decoder
fix becomes a hard gate (tau=0.9, fp32 softmax, fp32 Gumbel, collapse screen;
plus the early-stop fix so the SmolLM2-135M baseline is valid) rather than a
parallel work item, since L1's every quality number is otherwise measured with a
broken instrument. And the budget is cut from the spec's implied 5B tokens per
arm to 1B, because 5B is ~100-140 T4-h of training alone, its NLL delta is likely
inside run-to-run noise, and at 5B tokens the twin is still 400x short of
SmolLM2-135M's 2T and therefore cannot close the gap the spec says is the whole
point — whereas 1B tokens, 3 arms, ~36 T4-h of core training and ~62-90 T4-h end
to end, does answer the question L1 exists to answer.