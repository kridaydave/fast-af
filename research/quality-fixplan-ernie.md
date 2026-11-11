# Quality fix-plan: making D1's QUALITY numbers trustworthy, then strong

Author: ernie (research sub-agent, quality angle). Date: 2026-11-11.
Scope: the quality half of D1 (EdgeFast-Diff 100M) on a free T4 / Kaggle T4, zero budget.
Companion to `research/hillclimb-quality-ernie.md` (evidence pass). This doc is the **plan**: what to
build, in what order, at what cost, with what number to hit and what would make us stop.

---

## 0. How to read this

Every number below carries a label:

| Label | Meaning |
|---|---|
| **MEASURED** | from this repo's own files (`results_t4.yaml`, `bench/backends.py`, `modeling_smoldlm.py` in the HF cache) |
| **DERIVED** | arithmetic on a MEASURED number, formula shown |
| **PAPER** | quoted from a primary source, scope stated |
| **ESTIMATE** | my assumption, not measured. Always flagged, never load-bearing alone |
| **UNCONFIRMED** | plausible, no source. Do not build on it |

Cost unit is **T4-hour**: one hour of wall-clock on one T4, excluding my own coding time. The binding
constraint on this project is not T4-hours, it is **free-tier wall clock** (Kaggle 30 GPU-h/week;
Colab free sessions die every few hours) and **session friction**. Every plan below is ordered so that
the cheapest thing that can invalidate the rest runs first.

**Two corrections to the brief, with evidence, before anything else:**

1. **"All pass@1 numbers (0.0-0.10) measure the decoder, not the models."** Half true, and the
   half that is false matters. `results_t4.yaml` has **no pass@1 for smoldlm-144m at all** — the
   roster entry sets `humaneval: false` (`bench/config.yaml:68`) and the row has no `humaneval` key
   (results_t4.yaml:103-121). So there is no diffusion pass@1 to be broken; there is no diffusion
   pass@1, period. The AR numbers (0.0 / 0.06 / 0.02 / 0.10) come from `TransformersBackend` with
   `do_sample=False` (backends.py:77), which is the *normal* thing to measure for an AR model and is
   not a broken decoder. The SmolLM2-135M 0.0 is plausible on its face: the SmolLM2 paper reports
   HumanEval only for the 1.7B (22.6 base, 28.1 instruct) and never for the 135M (PAPER, arXiv
   2502.02737 Tables 3-4), so there is no published 135M number to contradict us.
2. **What *is* broken is narrower and worse.** Three things, all verified this session:
   (a) the diffusion twin is measured on a sampler its own author says not to use
   (`modeling_smoldlm.py:_add_gumbel_noise` documents τ=0 as a degenerate path; model card: "Always
   use temperature >= 0.5");
   (b) there is **no generative-PPL measurement anywhere in the repo** — `bench/` contains speed and
   HumanEval only, so the one metric that would tell us whether the model can write code at all does
   not exist;
   (c) **nothing is calibrated against a published number.** Our only quality anchors are four numbers
   we produced ourselves, from one harness, with no external check.

So the fix is not "re-measure with a better sampler". It is: *build a metric that can be trusted, prove
it agrees with a published number on someone else's model, then point it at ours.*

---

## 1. Sampler fix (Part 1)

### 1.1 What the current loop does, exactly

`bench/backends.py:165-210`. Per denoising pass it computes
`probs = torch.softmax(logits[0, -32:], dim=-1)` on **fp16 logits**, then `conf, tok = probs.max(-1)`
— greedy argmax, τ=0 — and commits every open slot whose max-prob clears 0.9. Three defects, in order
of severity:

| # | Defect | Evidence | Effect |
|---|---|---|---|
| D1 | τ=0 argmax | backends.py:181 | The one configuration the model author names as broken |
| D2 | fp16 softmax over a 49,152-token vocab | `probs` is fp16 because `logits` is fp16 and softmax preserves dtype | ~1e-3 relative error on the sharp tail; with any τ<1 sharpening this is the regime where near-ties flip |
| D3 | Confidence threshold is calibrated to greedy peaks | 0.9 is a max-prob threshold; the mean max-prob under τ=0.7 is materially lower | Under sampling the block stops filling: most passes commit only the single fallback token, and the last pass dumps ~24 tokens in parallel from an under-denoised context. **The effective denoising budget collapses as soon as you add temperature.** This is the trap the naive "just add τ" patch falls into |

D3 is the one nobody has written down. It is a *prediction*, not a measurement: I can state the
mechanism from the code, and the check is in §1.3 (log fills per pass at τ=0 vs τ=0.7). If fills per
pass fall from ~4 to ~1.2, D3 is real and the commit rule must change with the sampler.

### 1.2 The change (minimal, and it isolates the sampler)

Two knobs and one precision fix. Nothing else moves: block size, steps, mask handling, EOS, prompt
handling, timing contract all stay.

**Patch `bench/backends.py`:**

```python
# 1) Generation gains a pass counter so "tokens per pass" is always reportable
@dataclass
class Generation:
    text: str
    num_tokens: int
    decode_seconds: float
    passes: int = 0          # total model forwards in the decode window
    blocks: int = 0          # blocks committed

# 2) CustomBackend.__init__ (backends.py:149): three new args, all defaulted to
#    today's behaviour so no existing call site changes.
    def __init__(self, block_size: int, steps: int,
                 confidence: float, score_fn: Callable, tokenizer,
                 eos_id: int = 1,
                 temperature: float = 0.0,      # 0.0 == old greedy path, bit-identical
                 commit: str = "confidence",    # "confidence" | "topk"
                 noise_dtype: str = "float64",  # smoldlm's own choice
                 seed: int | None = None):
        ...
        self.temperature = temperature
        self.commit = commit
        self.noise_dtype = noise_dtype
        self.seed = seed
        self.passes = 0
        self.blocks = 0

# 3) New method: the checkpoint's own sampler, verbatim in spirit
    def _pick(self, probs):                     # probs: (block, V) fp32
        if self.temperature <= 0.0:
            conf, tok = probs.max(dim=-1)
            return conf.float().cpu(), tok
        # _add_gumbel_noise() from modeling_smoldlm.py, in fp64:
        #   argmax(exp(logits) / (-log U)**tau)  ==  sample from softmax(logits/tau)
        logits = probs.clamp_min(1e-20).log().to(
            torch.float64 if self.noise_dtype == "float64" else torch.float32)
        if self.seed is None:
            noise = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device)
        else:
            g = torch.Generator(device=logits.device)
            g.manual_seed(self.seed)
            noise = torch.rand(logits.shape, dtype=logits.dtype,
                               device=logits.device, generator=g)
        gumbel = (-torch.log(noise.clamp_min(1e-20))) ** self.temperature
        tok = (logits.exp() / gumbel).argmax(dim=-1)
        conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        return conf.float().cpu(), tok
```

```python
# 4) In generate() (backends.py:165), replace the scoring block (lines 174-196):
        self.passes = 0
        self.blocks = 0
        ...
        while emitted < out_len:
            block = [self.MASK_ID] * self.block_size
            seq = torch.tensor([out_ids + block], device=self.device)
            for step in range(self.steps):
                open_slots = [i for i, t in enumerate(block) if t == self.MASK_ID]
                if not open_slots:
                    break
                self.passes += 1
                logits = self.score_fn(seq)                     # (1, L, V)
                # FIX D2: fp32 softmax, matching the model's own confidence math
                probs = torch.softmax(logits[0, -self.block_size:].float(), dim=-1)
                conf, tok = self._pick(probs)                   # FIX D1
                conf = conf.float().cpu()
                if self.commit == "topk":
                    # smoldlm's stock schedule: n_masked // steps per pass, by confidence
                    n = max(1, len(open_slots) // max(1, self.steps - step))
                    order = sorted(open_slots, key=lambda i: -float(conf[i]))
                    fill = order[:n]
                else:
                    fill = (open_slots if step == self.steps - 1
                            else [i for i in open_slots if conf[i] >= self.confidence])
                    if not fill:
                        fill = [max(open_slots, key=lambda i: float(conf[i]))]
                for i in fill:
                    if int(tok[i]) == self.MASK_ID:
                        raise RuntimeError("mask id won at an open slot")
                    block[i] = int(tok[i])
                seq = torch.tensor([out_ids + block], device=self.device)
            ...
            self.blocks += 1
        # and return Generation(..., passes=self.passes, blocks=self.blocks)
```

**Why `topk` is a required arm, not an optional one.** Under `commit="confidence"` the number of
tokens committed per pass is a function of the confidence distribution, which temperature changes. So a
τ sweep at fixed 0.9 confounds "different sampler" with "different number of denoising rounds". The
`topk` arm fixes the schedule at `n_masked // steps` (smoldlm's own stock value,
`modeling_smoldlm.py:414`), so τ is the only variable. Run both; report both.

**What we deliberately do *not* change yet:** the KV cache. Wiring it changes the attention semantics
(the uncached full-sequence forward is itself an approximation of the block-causal training config —
see `research/hillclimb-speed-reverify.md` §3.4) and would confound this experiment. Quality work runs
uncached so its numbers are comparable to the existing 349.979 tok/s baseline. The cache is a separate
queued experiment, and §3 shows it is a *prerequisite* for distillation, not for measurement.

**Defaults after the fix:** keep `temperature=0.0` in `backends.py` so the speed harness is untouched
and every quality run states its τ explicitly in the config. Never let τ be implicit.

### 1.3 Verification that the fix worked

Five checks, in order. All run on the checkpoint we already have, no training.

**(V1) Regression: τ=0 must reproduce today's output.**
Run the patched backend at `temperature=0.0` on the same 1k/1k speed prompt.
Expected: the first 200 chars equal `results_t4.yaml:113-114` (the bracket soup) **token for token**,
because at τ=0 there is no RNG. Tolerance: ≥99% token agreement (the fp32 softmax in D2 can flip a
near-tie). If agreement is <99%, the fp32 change is altering argmax and that must be understood before
any other number is trusted.
Cost: 2 minutes. Command:
`python -m bench --only smoldlm-144m --out results/regress.yaml`

**(V2) Degeneracy metrics (model-free, no judge, no GPU-hours).**
Computed on the 1,024 generated tokens, per config. Definitions are ours; they exist to make
"bracket soup" a number instead of an adjective:
- `soup` = fraction of tokens in `{[ ,] ,( ,) ,{ ,} ,= ,;}` — greedy sample measures >0.30; coherent
  code sits at 0.05-0.15.
- `rep4` = 1 − unique 4-grams / total 4-grams — greedy sample ~1.0 (the sample repeats `]]]]]]`
  indefinitely); coherent text <0.10.
- `parse` = `ast.parse(completion)` succeeds (0/1). Bracket soup is 0. This is the single cheapest
  signal that separates "text" from "code".
- `mask_frac` = mask ids in the final text (must be 0.000 by construction; if not, the last-pass fill
  is broken).
- `entropy_H` = mean over open slots of the predictive entropy at commit time, in nats. Screen: reject
  any τ whose mean H < 5 nats (the sharpening paper's own floor, arXiv 2609.33947).
Report all five for every cell. Never report a quality number without `parse` and `tokens_emitted`.

**(V3) The upper bound: run the author's own decoder.**
`modeling_smoldlm.generate(model, prompt, max_new_tokens=256, steps=10, temperature=0.7)` — cached,
top-k, τ=0.7, EOS-stopping. This is the decoder the checkpoint was validated with. If the author's
decoder gets non-zero MBPP and our loop gets 0.0, the bug is in our loop (wrong mask id, wrong EOS,
wrong prompt handling, wrong attention mode) and *no amount of training research matters until it is
found*. This is the decisive test and it costs ~10 minutes.
Cost: 0.2 T4-h.

**(V4) Generative PPL, before vs after (Part 2 defines the metric).**
Same checkpoint, same 8 steps, same prompts, same judge. `τ=0` vs `τ=0.9`.
Pass condition: **gPPL ratio (τ=0.9) ≤ 0.5 × gPPL ratio (τ=0)** at 8 steps, i.e. the 2x-or-better
improvement the paper reports at 34-170M (arXiv 2609.33947: 3.5-4.9x on absolute gPPL, 16-1024 steps;
we should get less because we are at 8 steps, so 2x is the honest bar).
Cost: 0.15 T4-h for both arms.

**(V5) Can the model write code at all? Teacher-forced likelihood.**
Separate the model from the decoder, which no decode-time experiment can do:
- `NLL_real` = mean per-token NLL of the model on 200 reference MBPP solutions (teacher-forced, no
  masks).
- `NLL_shuf` = same on the same solutions with function bodies shuffled at line granularity.
If `NLL_real` is clearly below `NLL_shuf`, the model *has* code structure in its weights and the
failure is in decoding (World B). If the two are indistinguishable, the model does not have code and
the only lever is training (World A). This costs one forward pass per sequence: **~2 minutes**.
Cost: 0.05 T4-h.

**Kill criteria for the whole of Part 1:**

| Result | Conclusion | Action |
|---|---|---|
| V3 gives MBPP > 0 and our loop gives 0.0 | our harness is buggy | Stop. Debug mask/EOS/prompt/attention. Nothing else counts until fixed |
| V4 passes (≥2x gPPL at 8 steps) | decoder was the problem; World B | Proceed to Part 3 as planned; decode policy is now a quality lever |
| V4 fails at every τ but V5 says the model has code | the sampler is not the lever; the objective is | Skip to Q1 (train at the decode policy) |
| V4 fails and V5 says no code knowledge | World A | Drop pass@1 as an endpoint; the plan becomes gPPL-only and the token budget is the whole story |

### 1.4 Cost of Part 1

| Item | T4-h |
|---|---|
| V1 regression | 0.01 |
| V2 metrics (reuses V1 output) | 0 |
| V3 author's decoder on MBPP-500 | 0.2 |
| V4 gPPL before/after | 0.15 |
| V5 teacher-forced likelihood | 0.05 |
| Full τ × commit × steps grid (§1.5) | 2.4 |
| **Total** | **≈ 3 T4-h, no training** |

That is under one Colab session. **The entire "is our instrument broken" question costs 3 T4-hours.**
Every week spent on training before this runs is a week spent measuring with an unverified ruler.

### 1.5 The grid, exactly

Fixed: smoldlm-144m, uncached, block 32, fp16, batch 1, 1,024-token prompts, 256-token MBPP outputs.
Per cell: MBPP-500 pass@1 (3 seeds, report median and min/max) + gPPL (512×128) + V2 metrics + TPS +
`tokens_emitted` + `passes`.

| Arm | τ | commit | steps | Cells | T4-h |
|---|---|---|---|---|---|
| A (isolation) | 0.0, 0.7, 0.8, 0.9, 1.0 | topk | 8 | 5 | 0.9 |
| B (threshold) | 0.0, 0.7, 0.9 | confidence 0.9 | 8 | 3 | 0.5 |
| C (step count) | best τ, best commit | — | 16, 32 | 4 | 0.7 |
| D (author default) | 0.7 | topk | 10 | 1 | 0.2 |

Do not add cells until arm A is read. The output is one YAML row per cell with the five V2 metrics, the
two quality numbers, TPS, tokens, and passes; nothing is reported as a single scalar.

---

## 2. Eval harness (Part 2)

Today the repo can measure tokens/s and HumanEval pass@1 for AR models. It cannot measure whether a
diffusion model produces better *text*, which is the only quality axis we can move cheaply. Three new
pieces, all new files (no edits to `bench/`): `eval/genppl.py`, `eval/mbpp.py`, `eval/metrics.py`.

### 2.1 Generative PPL — design

**Why this metric and not something else.** It is the field's standard for exactly this question
(PAPER: both the sharpening paper, arXiv 2609.33947, and the fixed-state paper, arXiv 2609.11998, score
masked-diffusion samples with GPT-2 PPL), it is continuous (usable on undertrained models where pass@1
is 0), and it needs no tests, no sandbox, and no human. At 100M on 3-5B tokens, pass@1 will be 0-3%
for most of the project's life; a metric that only resolves at 10% is useless as an endpoint.

**Recipe.**

| Component | Choice | Why |
|---|---|---|
| Judge | `gpt2-large` (774M, fp16, T4) | Field standard; both papers we are anchored to use GPT-2 |
| Prompt source | 512 prompts × 64 tokens from a **held-out validation shard of our own pretrain mix**, never trained on | Must be in-distribution for the model, and must not be MBPP/HumanEval (sealed) |
| Generated length | 128 tokens | Matches the fixed-state protocol ("64→128-token continuations") |
| Sample size | 512 samples → 65,536 scored tokens | See stability rule |
| Score | gPPL = exp(mean token NLL under GPT-2-large over the 512×128 generated tokens, prompt excluded) | |
| Normaliser | **gPPL ratio** = gPPL(model) / gPPL(ground-truth 128-token continuation), same prompts, same judge | Removes judge difficulty. Makes 100M numbers interpretable and comparable across decode policies. Absolute gPPL is not comparable to any published number (different judge corpus, different tokenizer) |
| Uncertainty | bootstrap 95% CI over the 512 samples | |
| Stability rule | if the CI half-width > 5% of the estimate, double to 1,024 samples | |
| Sanity floor | ground-truth continuation must score gPPL < 30. If it does not, the judge or the prompt slice is wrong, and no model number means anything | |

**Known limitation, stated up front.** GPT-2 is a WebText model; it has seen little code, so absolute
PPL on code is high and the judge is a harsher critic of code than of prose. That is fine for
*deltas* (same judge, same prompts, only the decoder changes) and fatal for *absolute* claims. We
report ratios and deltas only. If a code-aware judge is wanted later, `bigcode/starcoder2-3b` fits a
T4 in fp16 (~6 GB) — but it adds a 3B download and an OpenRAIL licence check, and it is not needed for
any decision in this plan.

**Cost.** 512 × 128 tokens at 350 tok/s = 187 s generation; one GPT-2-large forward over 65,536 tokens
≈ 2×7.7e8×6.5e4 = 1.0e14 FLOPs ≈ 5-10 s on a T4 at 10-20 TFLOPS effective. **~3.5 min per config,
0.06 T4-h.** This is 30x cheaper than people assume, which is why it belongs in week 1 and not "after
we have a model".

### 2.2 MBPP pass@1 — the pilot endpoint

- Dataset: `google-research-datasets/mbpp`, **test split = 500 problems** (MEASURED from the dataset
  card; the paper confirms the split: "we hold out 10 problems for few-shot prompting, another 500 as
  our test dataset, 374 problems for fine-tuning" — PAPER, Austin et al. 2021, arXiv 2108.07732 §2).
- Few-shot block: the 10 rows of the `prompt` split, never the test rows.
- Prompt template: copy the MBPP paper's template verbatim (Appendix A of arXiv 2108.07732), freeze it
  as one constant, and record it in the results YAML. **Do not invent a new template.** (I did not
  extract the exact string this session; it is marked UNCONFIRMED here and must be copied from the
  paper before the first run.)
- Check: exec the completion + `test_list` asserts, n=1, no sandbox, same `_run` discipline as
  `bench/humaneval.py:48` (exec `completion + tests` and let a non-zero exit or failed assert be a
  fail).
- Seeds: 3 per config; report median and min/max. Without seeds, a 2-point MBPP difference is noise.
- HumanEval: **164 problems, run once, at the end** (`bench/humaneval.py` already loads all 164; the
  50-problem slice is a smoke test only). Reason it cannot carry a decision: at p=0.05, n=50, the 95%
  interval is ±6.0 points; at p=0.10, ±8.3 points. A 2x improvement from 3% to 6% is inside the noise.
  If HumanEval must carry a decision, use all 164 (±3.4 points at p=0.05) with a paired McNemar test on
  the same problems.

**Cost.** 500 problems × 256 tokens at 350 tok/s = 366 s = 0.10 T4-h per config. HumanEval-164 at 256
tokens = 0.04 T4-h.

### 2.3 Calibration: prove the harness agrees with a published number

This is the piece the repo has never had, and it is cheap. Pick a model whose HumanEval is published,
run it through **our** harness, and compare.

- Model: `HuggingFaceTB/SmolLM2-1.7B` (base). Published HumanEval pass@1 = **22.6** (PAPER, arXiv
  2502.02737, Table 3: SmolLM2 1.7B vs Llama3.2-1B 18.9 vs Qwen2.5-1.5B 37.2).
- Run: all 164 problems, raw prompt, 256-token continuation, greedy, our `_run`.
- Accept: **within ±6 points of 22.6** (prompt-format differences plus n=164 noise; ±3.4 points is the
  pure sampling term, so ±6 allows for the template difference). Note the paper used lighteval, which
  may use a different max length and stop set — record the difference.
- If it lands at 5 or at 45, the harness is wrong and every AR number we have is wrong with it.
- This is the only sanctioned early HumanEval use: a model we did not train, purely to validate the
  instrument. It cannot leak into D1 (our data prep is HumanEval-free by construction) and it teaches us
  nothing about D1.

**Cost.** 164 problems × 256 tokens at an estimated ~20 tok/s for 1.7B fp16 on T4 = 2,100 s = 0.6 T4-h.
(ESTIMATE on the 1.7B speed: the 135M measured 23.9 tok/s, and 1.7B is bandwidth-heavier but also
launch-bound, so 20 tok/s is a guess; it is measurable in the first two minutes of the run.)

### 2.4 Metric registry (freeze these definitions now)

Every results row must carry: `model, step_count, block_size, temperature, commit, confidence, seed,
tokens_emitted, passes, blocks, median_tps, soup, rep4, parse, entropy_H, mbpp_pass@1 (n, seeds),
genppl, genppl_ratio, ci95`. If a cell is missing `tokens_emitted` it is not a result — that single
omission is what produced the 307-token and 12-token rows in `results_t4.yaml`.

Two new fields on `Generation` (`passes`, `blocks`) make `tokens_per_pass = num_tokens / passes` a
first-class number. Under `block 32 / 8 steps` it should be 4.0; if it is 1.2 the block is not being
denoised (defect D3 in §1.1).

---

## 3. After measurement works: the highest-leverage quality work, ranked

Ranking rule: expected MBPP/gPPL gain per (T4-hour + engineering-week + risk), **at 100M on 3-5B
tokens**, after the decoder is fixed. Where the literature's evidence does not reach our scale I say so
instead of borrowing its magnitude.

### Q1. Train at the decode policy you will deploy (blockwise-aligned objective) — do this before the pretrain, not after

**Why.** The few-step cliff is steep and monotone (15.26 → 12.10 → 9.22 → 6.44 Gen-PPL at 4/8/16/32
steps) and **architecture barely moves it** (15.26/17.14/18.50 at 4 steps across attention/hybrid/Mamba;
6.44/6.48/6.52 at 32) — PAPER, arXiv 2609.11998 Table 5. A cliff that no mixer choice moves is a
*training* problem. Classical masked training puts loss on random masks across the whole sequence, while
inference decodes a block from a clean frozen prefix with the suffix fully hidden: noisy prefix, suffix
leakage, diluted supervision. Blockwise SFT fixes all three and its ablation peaks when the training
block size equals the inference block size (PAPER, arXiv 2508.19529).

**Cost.** Zero *extra* runs if it goes into the main pretrain — it is a masking construction, not a
separate experiment. The honest cost is one decision made before the run and one 500M-token twin to
de-risk it (≈10 T4-h, ≈1 week of free-tier wall clock).

**Expected effect.** +5.2 GSM8K / +1.6 MATH at matched FLOPs (PAPER, arXiv 2508.19529) — but that is
LLaDA-8B-Instruct + LoRA, verified in the paper's full text. **At 100M from scratch: no number exists.
Expect +1 to +3 MBPP points at 8 steps and a flatter gPPL curve.** The *shape* of the win (flatter
cliff) matters more than the size: a model that is good at 8 steps is 4x cheaper to run than one that
needs 32, which is the product.

**Kill criterion.** At 500M tokens, the aligned twin must beat the classical twin on **gPPL at 8 steps
by ≥3%** (gPPL, not pass@1 — at 500M tokens pass@1 is 0-1% and cannot resolve anything). If it does
not, keep classical masking and move the effort to Q2.

### Q2. On-policy / transition distillation at 100M — the product's 8-step regime, and the measurement nobody has made

**Why.** Our whole TPS claim rests on decoding in 8-16 steps. Nothing in the literature says a 100M
block-diffusion model is *usable* at 8 steps. And there is a clean, fillable gap: every
trajectory/on-policy distillation paper starts at 1.7B (T3D: SDAR-1.7B/4B, LLaDA-8B; OPTD: LLaDA-8B
only; CD4LM: LLaDA). The only sub-200M few-step evidence is SDTT (169M/424M/863M MDLM, 32-64x fewer
steps, arXiv 2410.21035) and FS-DFM (169M), and neither tested a block-diffusion model on code. **We can
produce the missing measurement at exactly our size.**

**The cost arithmetic that decides the design** (this is the useful part):

| Variant | Teacher passes per 128-token sample | 50M tokens (390,625 samples) at 11.43 ms/pass, batch 1 | same, batched 8 (A2, 2.5x) |
|---|---|---|---|
| Offline sequence KD (SDTT: teacher generates 32 steps) | 4 blocks × 32 = **128** | 159 T4-h | 63 T4-h |
| Student generates (8 steps), teacher scores each state (on-policy, OPTD-style) | 4 × 8 + 4 × 8 = **64** | 79 T4-h | **32 T4-h** |
| Student generates, teacher scores the *final* sequence only (1 pass) | 32 + 1 = **33** | 41 T4-h | 16 T4-h |

(Arithmetic: 390,625 samples × passes × 11.43 ms ÷ 3,600. 128 passes → 1.463 s/sample → 571,000 s →
159 h.)

Two facts fall out of that table and both change the plan:
1. **Scoring is one forward pass; generating is `steps` forward passes.** So on-policy relabelling
   (student rolls out, teacher labels) is ~2-4x cheaper than SDTT's offline teacher rollouts at the
   same data size. At zero budget that decides the design: **on-policy, not offline.**
2. **Even the cheapest variant is 41 T4-h unbatched / 16 T4-h batched, i.e. 1.5-2.5 weeks of free-tier
   wall clock.** This is the single most expensive item in the quality plan, and it is why the KV cache
   is not a speed nicety: at the cache's ~1 ms/pass floor (ESTIMATE, from
   `hillclimb-speed-reverify.md` §3.3) the 33-pass variant drops to ~4 T4-h. **Sequence: cache first,
   then distillation.**

**Expected effect.** T3D reports 17.11 → 47.13 average at TokPS=8 on LLaDA-8B (+175% relative); PlaidQ
reports 4-step pass@10 7.90 → 15.85 at 0.7B but that is *continuous* diffusion, not masked. At 100M:
**unknown, and plausibly small if the model is capacity-bound rather than step-bound.** That is the
whole question, and it is answerable for ~32 T4-h batched (79 unbatched).

**Kill criterion.** The 8-step student must beat the 32-step teacher on MBPP pass@1 at equal-or-better
TPS. If it does not, distillation is not buying quality at our scale: stop after this rung, report the
negative, and put the remaining effort into Q1/Q3.

### Q3. Looped mid-block: buy training FLOPs, which on zero budget means tokens

**Why.** LoopMDM (170M MDM, matched total FLOPs) reports up to **3.3x fewer training FLOPs at equal
NLL** and +8.5 GSM8K at matched training compute (PAPER, arXiv 2605.26106). At our budget, 3.3x fewer
FLOPs is *literally* 3.3x more tokens, and tokens are the binding constraint (§Q5). Loop *inside* the
denoise step (early-middle 2 layers, S_max 6-12), never loop-count = step-count.

**Cost.** One pretrain twin at 500M tokens each: ≈21 T4-h, ≈1.5 weeks wall clock.

**Expected effect.** 3.3x training-FLOP efficiency at 170M (PAPER, and it is the closest scale match in
the whole literature). Counter-evidence exists and is at 1.4-3.5B AR recurrent-depth, not MDM: 76-89%
of tokens decided in iteration 1, Ouro-Thinking best at iteration 1 and degrading after (arXiv
2610.18127); overthinking past ~15 iterations (arXiv 2604.07822). The positive evidence is closer to
our setting than the negative evidence, which is why this ranks above data-mix work — but the risk is
real.

**Kill criterion.** At matched total training FLOPs **and matched inference FLOPs**, the looped twin must
beat the unlooped twin on validation NLL. If it does not, looping is dropped. Explicitly: **do not
accept "equal TPS, equal quality" as a pass condition** — looping inside a step costs TPS by
construction, and the spec's original framing would make a null result unreadable.

### Q4. Confidence-remask (training-free decode policy)

**Why.** A masked model's own per-position confidence is a free, same-forward-pass quality signal. CoDR
(confidence drift, training-free) reports +0.85 to +1.44 points on LLaDA-8B and +0.79 to +1.11 on
Dream-7B at 123-135 NFE; T2M (reset-to-mask) reports +5.92 on CMATH by repairing "correct reasoning,
garbled final answer" — a very plausible description of a 100M code model; PRISM reports at **170M**
with <500 GPU-hours of fine-tuning (PAPER: CoDR EMNLP 2026 Findings, arXiv 2604.18738, arXiv 2510.01384).
This is the one verification result that reaches our scale.

**Cost.** ~30 lines in the decode loop, no training. 0.5 T4-h to measure.

**Expected effect.** Small: +1 to +3 MBPP points at 100M if the signal exists at all. Cheap enough to
run alongside Part 1.

**Kill criterion.** At matched NFE, confidence-remask must beat the sharpened sampler by ≥1.5 MBPP
points. If not, the model has no usable self-verification signal at 100M — which also kills spec recipe
step 3 (STaR/ReST), because that is the same premise.

### Q5. Data: token count and filtering first, ratio second

**Why.** Our 3-5B tokens is **400-600x less than the 135M yardstick's 2T** (PAPER, SmolLM2,
arXiv 2502.02737). SmolLM2-1.7B's HumanEval moves **10.97 → 9.15 → 17.68 → 22.60 across the four stages
of one run** — a 12-point swing from data alone, at 1.7B. Nothing else on this list has that effect
size. Meanwhile the mix ratio is a second-order term: REGMIX puts mixture effects at up to 14.6% on a
single task, and the 92-model meta-analysis puts the HumanEval-friendly code band at 15-25% and shows
penalties above it (PAPER: REGMIX ICLR 2025; arXiv 2503.03862). Our 50% code is above that band.

**What to do, in order:**
1. **Freeze the token budget from a measurement, not a guess** (see §5, week 2 probe). Everything else
   is downstream of this number.
2. **Filter hard, then dedup.** 3B clean tokens beat 10B mixed tokens at 100M. Concrete sources already
   chosen and already filtered (FineWeb-Edu `sample-10BT`, The Stack Dedup Python, FineMath /
   OpenWebMath). Do not add generic CommonCrawl to "get more tokens"; add *epochs* on the clean set if
   more tokens are needed, and monitor val NLL for the epoch where it turns up.
3. **Move the mix from 50/35/15 to 30-35% code / 45-50% web / 15-20% math** (a prep-script constant, no
   GPU cost), and say plainly that this is a judgement call from two secondary sources, not a measured
   win at 100M.
4. **Decontaminate against MBPP as well as HumanEval.** MBPP is our pilot endpoint, so an MBPP leak
   invalidates the pilot. Use docstring-prefix + solution n-gram + an embedding-similarity pass
   (n-grams provably miss paraphrases: arXiv 2602.12413 reports semantic duplicates of **100% of MBPP's
   test set** in a post-training corpus, and that no single standard filter removes more than 6.5% of
   them). Accept the residual and report it; do not claim "decontaminated" as a binary.
5. **Report a seed control at least once** (two runs, same data, different seed) so a 2-3 point effect
   can be separated from run-to-run variance.

**Expected effect.** Token count: large and unbounded upward, bounded by free-tier wall clock. Mix: ≤14.6%
single-task (PAPER, REGMIX), and probably less at 100M. **Do not run a mix A/B** — two 3B pretrains is
4-6 weeks of wall clock to resolve a second-order term (§4).

**Kill criterion.** n/a (hygiene). One rule: if decontamination costs more than ~2 points of HumanEval,
we are removing real signal (HumanEval-shaped style transfer), so report both numbers.

### Q6. Block size 32 vs 8 — a pretraining-efficiency lever with a 4x decode bill

**Why it is on the list.** The compute gap to AR is 7.8x at block 32 but 4.6x at block 8 and 3.0x at
block 4 (PAPER, arXiv 2610.17590, exact likelihood + block diffusion + encoder-decoder, ~1,700 models).
So block 8 is materially cheaper to train for the same likelihood.

**Why it is ranked last.** The decode cost is arithmetic, not opinion: tokens per pass = block/steps, so
at fixed 8 steps, block 32 → 4 tokens/pass and block 8 → **1 token/pass**, i.e. **4x the forward passes
for the same tokens and ~4x lower TPS.** The product is maximum TPS. Block 8 trades the product's one
metric for training efficiency. Also, decoding a block-32-trained model at block 8 is an
*approximation* (positions that were bidirectional in training become clean context), so a decode-time
sweep gives a lower bound on the benefit, not a measurement of it.

**Cost.** Decode-time proxy sweep (block ∈ {8,16,32} × steps ∈ {4,8,16}, no training): 0.5 T4-h. A real
answer needs a pretrain twin: ≈21 T4-h per arm.

**Expected effect.** Training-compute: 7.8x → 4.6x (PAPER). Quality at 8 steps: unknown at 100M.
TPS: −4x at fixed steps (DERIVED).

**Kill criterion.** Only revisit block 8 if the week-2 compute probe shows the 3B run would exceed ~6
weeks of wall clock. If block 8 does not buy ≥2x training throughput in the probe, it is dead on
arrival.

### Q7. Not built: verification-during-refinement, adaptive steps, STaR/ReST

Scored here so the ranking is complete, then killed in §4.

---

## 4. What NOT to do

Plainly, because a plan that only adds is not a plan.

1. **Do not run a data-mix A/B.** Two 3B-token pretrains = 4-6 weeks of free-tier wall clock to resolve
   a term worth ≤14.6% on one task, while the token count is 400-600x below the yardstick. Freeze a
   defensible mix (Q5) and spend the weeks on tokens.
2. **Do not build STaR/ReST self-distillation for v1.** Two independent reasons: (a) the premise is
   false at our scale — the generation-verification gap scales with pretraining FLOPs and is
   non-positive for ≤7B models (PAPER, arXiv 2412.02674), and self-correction is bottlenecked by the
   verifier, not the refiner (PAPER, arXiv 2404.17140); (b) mechanically, at pass@1 ≈ 5% you need ~20
   samples per problem to harvest one passing solution, so the yield is ~0 until pass@1 clears ~10-15%.
   Q4 is the cheap test of whether even the diffusion-specific version of the signal exists. If Q4
   fails, step 3 of the training recipe is dead, not deferred.
3. **Do not build the full verify-during-refinement + adaptive-steps mechanism.** Oversized, and its
   premise is the one Q4 tests for 0.5 T4-h.
4. **Do not run L1 as specified.** "Loop iterations ARE the denoising steps" + "TPS untouched by
   construction" is a configuration nobody in the literature uses, and the equal-TPS framing makes a
   null result unreadable. Loop inside the step (Q3) and use matched-FLOPs.
5. **Do not run Tier 3 (LLaDA-8B default + Fast-dLLM vs Qwen3.5-9B) before the measurement works.** It
   is the biggest friction item in benchmark week and it cannot inform D1's decode policy until we can
   measure a decode policy. Run it last, if at all.
6. **Do not chase int4 QAT quality before there is a quality baseline.** At 100M the int4 cost is paid
   in the same currency as everything else, and we cannot see it yet.
7. **Do not push 32k context.** The fixed-state paper's passkey numbers are weak at 16k for all mixers;
   our evals run at 1k. Long context is a v2 problem.
8. **Do not use HumanEval-50 as a decision metric.** ±6 to ±8 points of 95% interval at the pass rates
   we expect (DERIVED: binomial). Use it as a smoke test; use MBPP-500 and gPPL for decisions.
9. **Do not report TPS without `tokens_emitted` and `passes`.** That omission is what produced the
   307-token and 12-token rows.
10. **Do not try to beat SmolLM2-135M on HumanEval with 3-5B tokens.** The bar is "coherent code-shaped
    text, non-zero MBPP pass@1, and a gPPL curve that is flat in step count". A HumanEval gap to a
    2T-token model is expected and is not a defect.
11. **Do not train a tokenizer and then change it.** Already decided (32k BPE, own corpus). Freezing it
    is what makes token counts comparable across every experiment in this plan.
12. **Do not add GDN-2, adaptive block size, or tool calling to v1.** Each is a new mechanism stacked on
    two already-experimental axes.

---

## 5. Sequencing (zero budget, week by week)

Wall-clock assumptions: Kaggle 30 GPU-h/week; Colab free T4 in ≤12 h sessions with disconnects;
everything above 30 T4-h therefore costs ≥1 week of wall clock regardless of how small the number looks.
A 3B-token pretrain is ≈64 T4-h (ESTIMATE §6) = **≈2-3 weeks of wall clock**, so it must not start
before the recipe is frozen.

### Week 1 — instrument (3 T4-h, 3 sessions)

| Day | Work | Gate |
|---|---|---|
| 1 | Calibration: SmolLM2-1.7B HumanEval-164 through our harness, expect 22.6 ±6 (0.6 T4-h) | If outside ±6, fix the harness before anything else |
| 1 | Build `eval/metrics.py` (V2 metrics) + `eval/genppl.py` + `eval/mbpp.py` | ground-truth gPPL < 30 sanity floor |
| 2 | Apply the §1.2 patch; V1 regression (τ=0 reproduces the bracket soup) | ≥99% token agreement with `results_t4.yaml` |
| 2 | V5 teacher-forced likelihood (does the model have code?) | World A or World B |
| 3 | V3 the author's own decoder on MBPP-500; then the §1.5 grid (arms A+B, 8 cells) | gPPL ratio at τ=0.9 ≤ 0.5x of τ=0 |

**Decision D1 (end of week 1).**
- Author's decoder > 0 and ours = 0 → **harness bug; stop the project's research track and debug.**
- τ=0.9 beats τ=0 by ≥2x gPPL → decoder was the lever; continue.
- No τ helps → the sampler is not the lever; the plan collapses to Q1 + Q5 and the 3B pretrain is the
  only remaining experiment.
- Model has no code likelihood → World A; drop pass@1 as an endpoint for the rest of the project.

### Week 2 — freeze the recipe (1 T4-h)

| Day | Work | Gate |
|---|---|---|
| 4 | Freeze decode policy: τ*, commit, steps, block, from the week-1 grid. Write it into `bench/config.yaml` as explicit numbers | every quality row states τ |
| 4 | Compute probe: 200 steps, batch 8, seq 1024, block 32 — measure tokens/s (0.3 T4-h) | ≥13,000 tok/s → 3B in ≈64 h; <6,000 tok/s → recipe must change |
| 5 | Data prep + decontamination (HumanEval **and** MBPP; prefix + n-gram + embedding); freeze the mix at 30-35% code | 0 exact test-set hits; report residual |

**Decision D2 (end of week 2).** If the probe is below 6,000 tok/s, 3B tokens is 6+ weeks: either cut to
1.5B tokens and accept a weaker model, or switch to block 8 (Q6) and accept 4x decode cost. Do not
start the pretrain hoping the probe was pessimistic.

### Weeks 3-6 — pretrain D1 (64 T4-h, 2-3 weeks wall clock)

- 3B tokens, block 32, context 1024, the Q1 aligned objective if the 500M twin (below) cleared its
  gate, else classical masking.
- Checkpoint every 2,000 steps to Drive, with optimizer state and a resume script. A 62-hour run on a
  free tier *will* die repeatedly; the resume path is part of the deliverable, not an afterthought.
- Eval every 500M tokens: validation NLL + gPPL at 8 and 32 steps (0.1 T4-h). This gives a learning
  curve, catches divergence in week 3 instead of week 6, and produces the cliff curve for the write-up.

**Decision D3 (at 1.5B tokens, mid-run).** If gPPL at 8 steps is not falling toward gPPL at 32 steps
(i.e. the cliff is not flattening), the decode-policy work has failed and the remaining budget should be
spent on tokens (Q5), not on architecture.

**Decision D4 (end of run).** If MBPP pass@1 < 2%, pass@1 is not a usable endpoint at this scale: report
gPPL ratio vs a same-size AR baseline's gPPL ratio, and stop optimizing pass@1.

### Week 7 — distillation (32 T4-h batched / 79 unbatched, 1-2 weeks wall clock) — only if D1/D3 were positive

On-policy transition distillation (Q2), batched, 50M tokens of student rollouts, teacher = the same
weights at 32 steps.
**Decision D5.** If the 8-step student does not beat the 32-step teacher on MBPP at equal-or-better TPS,
distillation does not buy quality at 100M: stop, publish the negative (it is a real result — the gap
below 1.7B is unmeasured), and put the remaining weeks into tokens.

### Week 8 — looped mid-block twin (Q3), only if the pretrain pipeline is reusable and week 7 was positive

**Decision D6.** Matched total training FLOPs and matched inference FLOPs; validation NLL is the
endpoint. If the looped twin does not win, looping is dropped and L1 is closed with a clean negative.

### Week 9 — final eval and write-up

HumanEval-164 (unsealed, once), MBPP-500, gPPL at 8/16/32, TPS with tokens emitted, and the cliff curve
against the fixed-state paper's 15.26/12.10/9.22/6.44.

**Total: ≈120 T4-h, ≈9 weeks of wall clock** (≈98 T4-h for the 10-line list in §8, plus the week-8
looped twin at 21 T4-h). The pretrain and the distillation run are 80% of it; the measurement work that
makes all of it interpretable is 3% of it.

---

## 6. Cost model (assumptions, stated once)

| ID | Assumption | Value | Status |
|---|---|---|---|
| A1 | One forward pass of smoldlm-144m, fp16, batch 1, seq ≤2048, on T4 | **11.43 ms** | MEASURED-DERIVED: 349.979 tok/s and 256 passes per 1,024 tokens (32 blocks × 8 steps) |
| A2 | Batch 8 raises per-pass throughput ~2.5x (the pass is launch/overhead-bound: 11.43 ms vs a ~0.9 ms weight-read floor for 0.29 GB at 320 GB/s) | 4.6 ms/seq/pass | ESTIMATE — must be measured in week 1 |
| A3 | Training throughput at batch 8 / seq 1024 | **13,000 tok/s** (≈8 TFLOPS, 12% MFU); plausible range 5,000-13,000 | ESTIMATE — the week-2 probe replaces it |
| A4 | Free tier | Kaggle 30 GPU-h/week; Colab sessions ≤12 h with disconnects | FACT |

Derived budgets: 3B tokens at A3 = 64 T4-h; 1B = 21 T4-h; 500M = 11 T4-h.
Per-token training FLOPs = 6N = 6e8 for N=100M (standard dense estimate, no attention-sequence term at
seq 1024 — that term is ~1% here).

---

## 7. Source ledger

**Verified this session (fetched or read from disk):**
- `results_t4.yaml`, `bench/backends.py`, `bench/config.yaml`, `bench/humaneval.py`, `bench/run.py`,
  `README.md`, `spec/edgefast-diff-100m.md`, `bench/BENCHMARK_WEEK.md` — repo state, read in full.
- `modeling_smoldlm.py` (HF cache, snapshot `64f1177`): `_add_gumbel_noise` (fp64, exp(logits)/U^τ),
  `generate()` defaults `steps=10, temperature=0.7`, `tokens_per_step = max(1, n_masked // steps)`,
  cache warmup + per-block finalize. `config.json`: 144.47M params, 30 layers, n_embd 576, vocab 49,152,
  block 32, mask 0 / eos 1 / pad 2, seq_len 2048.
- SmolLM2 paper (arXiv 2502.02737, ar5iv HTML): HumanEval **22.6** for 1.7B base, 28.1 for 1.7B
  instruct, **no 135M HumanEval row**; stage table 10.97 → 9.15 → 17.68 → 22.60 (1.7B).
- SmolLM2-135M-Instruct model card (HF): reports IFEval/MT-Bench/HellaSwag/ARC/PIQA/MMLU/BBH/GSM8K,
  **no HumanEval**.
- MBPP (arXiv 2108.07732, ar5iv): 974 tasks; split = 10 few-shot / 500 test / 374 fine-tune / rest
  validation. Dataset card (HF): full = 974 rows, sanitized = 427 rows.
- SDTT (arXiv 2410.21035, abstract): "reduces the number of inference steps by a factor of 32-64",
  "up to 860M parameters", 1.3B scale up to 8x faster than AR with KV caching.

**Carried from `research/hillclimb-quality-ernie.md` §6 ledger (fetched 2026-11-11, primary sources
opened):** 2609.11998 Table 5 (the cliff, 15.26/12.10/9.22/6.44; 512-sample, 64→128-token GPT-2
protocol), 2609.33947 (sharpening, 3.5-4.9x, 34-170M, τ=1 generically suboptimal, no help at 8B),
2508.19529 (blockwise SFT, +5.2/+1.6, LLaDA-8B LoRA only), 2605.26106 (LoopMDM 170M, 3.3x),
2610.04198 / 2610.30066 / 2609.40305 (loop-inside-step), 2610.18127 / 2604.07822 (looping
counter-evidence), 2412.02674 (GV gap ≤7B), 2404.17140 (verifier bottleneck), 2510.01384 (PRISM 170M),
CoDR / 2604.18738 (T2M), 2602.12262 (T3D), 2608.02942 (OPTD), 2509.20624 (FS-DFM), 2605.00161 (CDLM
110M), 2610.17590 (7.8x/4.6x/3.0x), 2603.22075 (AR vs MDLM at 123M), 2511.03276 (super data learners),
2503.03862 (15-25% code), REGMIX, 2602.12413 (soft contamination), 2407.07565 (contamination
prevalence), 2504.11393 (DataDecide).

**UNCONFIRMED, do not build on:** the exact MBPP few-shot prompt string (must be copied from arXiv
2108.07732 Appendix A before the first MBPP run); the 1.7B's expected tok/s on T4 (20 tok/s is an
estimate); batch-8 throughput scaling (A2); training MFU (A3).

---

## 8. Action list (10 lines, in order)

| # | Task | T4-h | Number to hit |
|---|---|---|---|
| 1 | Calibrate the harness: SmolLM2-1.7B, HumanEval-164, our `_run` | 0.6 | pass@1 = 22.6 ±6 |
| 2 | Build `eval/genppl.py` (512×128, GPT-2-large, gPPL ratio) + `eval/mbpp.py` (500 problems) | 0.3 | ground-truth gPPL < 30; 500/500 problems load |
| 3 | Patch `CustomBackend`: fp32 softmax, τ/Gumbel (fp64), `commit=topk`, pass counter | 0.2 | τ=0 output ≥99% identical to `results_t4.yaml` |
| 4 | Run smoldlm's own `generate()` on MBPP-500 as the author-intended upper bound | 0.2 | MBPP pass@1 > 0 |
| 5 | Teacher-forced likelihood: real solutions vs shuffled | 0.05 | NLL_real at least 0.5 nats below NLL_shuf |
| 6 | τ × commit grid at 8 steps (10 cells) | 2.4 | gPPL ratio at τ=0.9 ≤ 0.5x of τ=0 |
| 7 | 200-step pretrain probe (batch 8, seq 1024) | 0.3 | ≥13,000 tok/s → 3B tokens in ≤64 h |
| 8 | Decontaminate (HumanEval + MBPP, 3 passes) and freeze the mix at 30-35% code | 0 | 0 exact test-set hits; residual reported |
| 9 | Pretrain D1 on 3B tokens (aligned objective, block 32, 8-16 steps) | 64 | val NLL falling at 500M-token checkpoints; gPPL@8 within 2x of gPPL@32 |
| 10 | On-policy distillation, 50M student tokens, teacher = 32-step self | 32 | 8-step student ≥ 32-step teacher on MBPP at ≤ equal TPS |

Total ≈98 T4-h (≈120 with the week-8 looped twin), ≈9 weeks of free-tier wall clock. Lines 1-8 are
4 T4-h and decide whether lines 9-10 are worth running at all.