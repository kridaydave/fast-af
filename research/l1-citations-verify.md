# L1 citation verification (web search unavailable; direct arXiv fetch only)

Task: verify citations carried into `spec/edgefast-diff-100m.md` L1v2 (`spec/edgefast-diff-100m.md:110-156`) and cited in `research/l1-final.md`. Source policy followed: each arXiv `abs` page fetched; HTML full text fetched where abstract was insufficient. No content guessed. If a needed detail could not be confirmed from fetched pages, verdict is `UNCONFIRMED`/`MISLEADING` with note.

## Per-citation findings

### 1. arXiv:2605.26106 — LoopMDM (Looped Diffusion Language Models)
- Fetched: `https://arxiv.org/abs/2605.26106`, `https://arxiv.org/html/2605.26106v1` (partial HTML retrieved in this run).
- Spec claims checked:
  - "170M MDM": **CONFIRMED**.
    - Quote (HTML): "We train two models, LoopMDM and a non-looped MDM baseline... using a 170M-parameter diffusion transformer..." and "All models are iso-parameter (170M)..."
  - "looped layers INSIDE each denoising step": **CONFIRMED** (architecture applies looping inside the denoising network forward pass).
    - Quote (HTML): "LoopMDM selectively loops a small shared block inside masked diffusion language models..." / "selectively looping the early-middle transformer layers..." (architecture section: head, looped mid-block applied S times, tail per forward pass).
  - "up to 3.3x fewer training FLOPs at matched FLOPs/performance": **CONFIRMED**.
    - Quote (abstract/HTML): "LoopMDM matches the performance of same-size MDMs with up to 3.3× fewer training FLOPs..." Figure text: "reaching the same final NLL with 3.34×, 2.95×, and 2.34× fewer training FLOPs..." on LM1B/OWT/FineWeb-Edu.
  - "2-of-16 style mid-block loop arrangement": **MISLEADING / PARTIALLY WRONG for stated form**.
    - Quote (HTML): "12-layer LoopMDM loops two (first and second) mid-layers 1-2 in zero-based indexing..." (also math/reasoning setup: "14 layers... loops two mid-layers 1–2..."). The paper's core experiments use 2 looped layers in a 12-layer (or 14-layer) stack, not "2 of 16" as a canonical arrangement stated in the paper. The principle (small early-middle mid-block looped) matches, but "2-of-16 style" is not a paper-stated configuration and should not be presented as a quoted paper detail.
- Verdict: **CONFIRMED (core), MISLEADING (2-of-16 wording)**.
- Action on spec: replace "2-of-16 style mid-block loop arrangement" wording in mechanism text with paper-accurate description: "loop a small early-middle mid-block (e.g. 2 layers in a 12-layer 170M MDM) inside each denoising step" or cite as analogous; keep 170M and 3.3x claims.

### 2. arXiv:2610.04198 — ALoDLM (Adaptively Looped Diffusion Language Models)
- Fetched: `https://arxiv.org/abs/2610.04198` (abstract). HTML not required beyond abstract for this claim.
- Spec claims checked:
  - "same loop-inside-step arrangement exists here": **CONFIRMED (arrangement class)**.
    - Quote (abstract): "At each denoising step, ALoDLM iteratively refines latent representations and allocates computation according to token difficulty. Tokens ready to commit are fed back as discrete context, while unresolved tokens retain and further refine their latent states through additional recurrent passes."
  - Scale/context note: paper reports 1.7B and 8B models (not 100M). Arrangement applies to diffusion denoising steps but with adaptive/token-wise recurrence mechanism different from fixed S mid-block in L1v2.
- Verdict: **CONFIRMED (loop/refinement inside denoising step)**, with scale/mechanism caveat.
- Action on spec: keep citation for "same loop-inside-step arrangement exists" but can clarify "adaptive token-wise latent recurrence inside each denoising step (1.7B/8B)" to avoid implying identical fixed mid-block mechanism.

### 3. arXiv:2610.30066 — Looped Uniform Diffusion Language Models
- Fetched: `https://arxiv.org/abs/2610.30066` (abstract).
- Spec claims checked:
  - "same loop-inside-step arrangement exists here": **CONFIRMED**.
    - Quote (abstract): "selective looping, which recursively applies a few early Transformer layers within each denoising step, has been shown to improve training efficiency in MDLMs. We investigate looping in UDLMs while keeping the training objective and sampler unchanged."
- Verdict: **CONFIRMED**.
- Action on spec: none required for arrangement claim.

### 4. arXiv:2610.18127 — Monitoring latent reasoning in looped language models
- Fetched: `https://arxiv.org/abs/2610.18127`, `https://arxiv.org/html/2610.18127v1`.
- Spec claims checked:
  - "8 looped AR LMs, 1-3.5B": **CONFIRMED**.
    - Quote (HTML/intro): "We apply it to examine eight looped LMs from four families." / "Several research groups have now released such looped LMs at the 1-3.5B scale..." (Ouro, LoopUS carry; Huginn, Rec-OLMo, Parcae re-read families; table lists 1.4B–3.5B class models).
  - "76-89% of tokens settled on final top-1 in iteration 1 in 6/8 models": **CONFIRMED (as stated in paper intro/findings)**.
    - Quote (HTML): "The answer is decided early. In six out of the eight models, 76-89% of the generated tokens already settle on their final top-1 token in the first iteration."
  - "Ouro-Thinking degrades after iteration 1": **CONFIRMED**.
    - Quote (HTML): "More iterations are not always better... The reasoning-tuned Ouro-Thinking models are best at iteration 1 and get worse with every iteration." / "Ouro-Thinking gets worse with every iteration... read the prompt best at iteration 1 or 2 on most datasets. Their log-loss at the last iteration is 5% (1.4B) and 22% (2.6B) higher than at the first..."
  - "model class (AR vs diffusion) - does it even apply to us?": **MISLEADING for direct applicability to 100M block-diffusion**.
    - These are looped language models studied as recurrent/looped transformers in latent reasoning (AR-style token generation setting in this analysis context), not masked/block-diffusion denoising models. Spec cites it as negative-risk evidence; the numerical finding is real but transfer to L1v2's MDM/block-diffusion, 100M, loop-inside-denoising-step setting is not established by this paper.
- Verdict: **CONFIRMED (numbers/facts), MISLEADING (direct applicability class)**.
- Action on spec: in accepted-risks line, explicitly label as "looped AR LMs, 1-3.5B (not diffusion)" to prevent over-reading transfer.

### 5. arXiv:2604.07822 — Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers
- Fetched: `https://arxiv.org/abs/2604.07822`, `https://arxiv.org/html/2604.07822v2`.
- Spec claims checked:
  - "small transformers from scratch": **CONFIRMED**.
    - Quote (abstract/HTML): "Through controlled studies with models trained from scratch..." (synthetic tasks, decoder-only transformers in experiments).
  - "overthinking past ~15 iterations": **PARTIALLY CONFIRMED but MISLEADING as stated generally**.
    - Quote (HTML): "Despite strong generalization performance, we observe performance degradation when using excessively large inference-time iterations. For example, increasing iterations beyond 15 for the R=dynamic does not improve OOD performance..." This is an observed example in a specific depth-extrapolation setting (dynamic recurrence model), not a universal threshold "~15 iterations" for all small transformers/tasks. Spec's phrasing overstates generality.
  - "three-stage grokking": **CONFIRMED**.
    - Quote (abstract/HTML): "Systematic generalization emerges through a three-stage grokking process, transitioning from memorization to in-distribution generalization and finally to systematic generalization..." / "We observe a three-stage dynamic..."
- Verdict: **CONFIRMED (trained from scratch, three-stage grokking), MISLEADING (blanket '>~15 iterations' threshold)**.
- Action on spec: change "overthinking past ~15 iterations" to reflect paper accurately, e.g. "overthinking at excessive recurrence (e.g. beyond ~15 iterations in their dynamic-recurrence depth-extrapolation setting)".

### 6. arXiv:2502.02737 — SmolLM2
- Fetched: `https://arxiv.org/abs/2502.02737`, `https://arxiv.org/html/2502.02737v1`.
- Spec claim checked (G0): "SmolLM2-1.7B base scores 22.6 on HumanEval (Table 3)". 
  - Score and model variant: **CONFIRMED as base SmolLM2-1.7B = 22.6 on HumanEval in Table 4 of fetched paper version** (number and base-vs-instruct distinction supported).
    - Quote (HTML Table 4, base model evaluation): "| HumanEval | 22.6 | 18.9 | 37.2 |" under columns "SmolLM2 1.7B | Llama3.2 1B | Qwen2.5 1.5B" (base comparison table). Section states "Evaluation results in Table 4..." for base models.
  - Table number: **WRONG in spec text** — spec says "Table 3"; fetched paper shows HumanEval 22.6 in **Table 4**.
  - Eval setting / HumanEval version / shots: **UNCONFIRMED from fetched excerpt for full protocol detail**.
    - Fetched text states general evaluation setup ("Evaluations are conducted using lighteval and in a zero-shot setting unless otherwise specified") and ablation setup used HumanEval/MultiPL-E via BigCode-Evaluation-Harness in specific contexts, but the exact HumanEval variant (e.g. original HumanEval pass@1 protocol, n, temperature, prompt format) for Table 4 is not fully specified in the snippets retrieved. Spec must not assume unstated details.
- Verdict: **MISLEADING/WRONG (table number) + CONFIRMED (22.6, SmolLM2-1.7B base) + UNCONFIRMED (exact HumanEval protocol details)**.
- Action on spec: correct "Table 3" to "Table 4" (per 2502.02737v1), state "SmolLM2-1.7B base, HumanEval 22.6 (Table 4, base model comparison; evaluation conducted with lighteval, zero-shot unless otherwise specified — confirm exact HumanEval pass@1 protocol when running G0)".

### 7. arXiv:2108.07732 — Program Synthesis with Large Language Models (MBPP)
- Fetched: `https://arxiv.org/abs/2108.07732`, `https://arxiv.org/html/2108.07732v1` (partial; Appendix A body text not fully retrievable from fetched tail in this run).
- Spec claims checked:
  - "Appendix A contains the few-shot/task template we must copy": **UNCONFIRMED from fetched content**.
    - Appendix A heading exists in HTML ("Appendix A Appendix", A.1/A.2 instructions visible in retrieved sections), but the exact "few-shot/task template" text to copy was not directly confirmed in fetched excerpts. Paper does describe prompt/few-shot setup in main text (e.g. holding out examples for prompts, dialog/assert formatting), but spec's claim that Appendix A specifically contains the canonical template to copy cannot be verified from what was fetched here.
  - "test split = 500 problems": **CONFIRMED**.
    - Quote (HTML): "The Mostly Basic Programming Problems (MBPP) dataset contains 974 programming tasks..." and "In the experiments described later in the paper, we hold out 10 problems for few-shot prompting, another 500 as our test dataset (which is used to evaluate both few-shot inference and fine-tuned models), 374 problems for fine-tuning, and the rest for validation."
- Verdict: **CONFIRMED (500 test split), UNCONFIRMED (Appendix A as copyable template source)**.
- Action on spec: keep "MBPP test split = 500 problems (2108.07732)". Revise prerequisite to: "use the MBPP task/few-shot template as defined in arXiv:2108.07732 (verify exact template text from paper Appendix/prompt description before first run; UNCONFIRMED in repo and not fully retrieved here), freeze it, and record text/hash in every results YAML."

### 8. arXiv:2510.01384 — PRISM (Fine-Tuning Masked Diffusion for Provable Self-Correction)
- Fetched: `https://arxiv.org/abs/2510.01384`, `https://arxiv.org/html/2510.01384v4`.
- Spec claims checked:
  - "confidence-remask works as a training-free diagnostic": **WRONG/MISLEADING as attributed to PRISM**.
    - Quote (HTML): "PRISM–Plug-in Remasking for Inference-time Self-correction of Masked diffusions–a lightweight, model-agnostic approach..." / "attaches a lightweight adapter to any pretrained MDM and fine-tunes with this loss..." This is a fine-tuning framework, not training-free. Spec sentence groups "PRISM reports at 170M...; CoDR training-free" which can blur responsibility, but current wording "cheap training-free confidence-remask diagnostic..." followed by PRISM citation is misleading.
  - "reports at 170M": **CONFIRMED**.
    - Quote (abstract/HTML): "Empirically, PRISM advances MDM inference across domains and scales: Sudoku; unconditional text (170M); and code with LLaDA (8B)." / setup: "take a DiT-based MDM with 170M parameters..."
  - Mechanism relevance: PRISM learns per-token quality scores for remasking/self-correction (fine-tuned); confidence/remask signals exist, but as a learned module in this work.
- Verdict: **CONFIRMED (170M report), WRONG/MISLEADING (training-free diagnostic attribution)**.
- Action on spec: separate claims clearly: e.g. "a cheap training-free confidence/remask diagnostic (CoDR-style, training-free) may run separately...; PRISM (2510.01384, fine-tuned, reports at 170M MDM) is separate evidence that remask/self-correction signals can be useful" or drop PRISM as example of training-free diagnostic.

## Required spec lines that must change (`spec/edgefast-diff-100m.md`)

1. **L1v2 mechanism, line ~118**: "loop a small early-middle mid-block inside each denoising step (LoopMDM-style, PAPER arXiv 2605.26106, 170M MDM; same arrangement in 2610.04198 and 2610.30066, PAPER, repo ledger)..."
   - Change "LoopMDM-style..." wording to avoid non-paper "2-of-16 style" implication (if present elsewhere) and cite paper-accurate setup: LoopMDM selectively loops early-middle layers inside denoising step (e.g. 2 layers in 12-layer 170M MDM, up to 3.3x fewer training FLOPs at matched performance). Keep arrangement citations, but note 2610.04198/2610.30066 differ in mechanism/scale (adaptive/token-wise or UDLM, 1.7B/8B for ALoDLM).
2. **G0, line ~123**: "Pass: within +/- 6 points of 22.6 (PAPER, arXiv 2502.02737 Table 3)."
   - Change to "within +/- 6 points of 22.6 (SmolLM2-1.7B base, HumanEval, arXiv 2502.02737v1 Table 4; confirm exact HumanEval pass@1 protocol/version when running G0)."
3. **MBPP template prerequisite, line ~130** (and metric line ~144): "copy the exact few-shot/task template from arXiv 2108.07732 Appendix A (PAPER)..."
   - Change to require copying/freezing the MBPP template as defined by arXiv:2108.07732, but mark exact Appendix A template text as UNCONFIRMED from this verification pass; must verify/copy from paper source before first MBPP run. Keep "test split (500 problems, PAPER arXiv 2108.07732)" (confirmed).
4. **Accepted risks, line ~146**: "... (PAPER arXiv 2610.18127: 76-89% of tokens settled in iteration 1 in 6/8 looped AR LMs; 2604.07822: overthinking/grokking risk)..."
   - Clarify 2610.18127 as looped AR LMs, 1-3.5B (not diffusion) so transfer risk is explicit. Adjust 2604.07822 to "overthinking at excessive recurrence (e.g. beyond ~15 iterations in a reported dynamic-recurrence setting) and three-stage grokking in small transformers trained from scratch" (avoid blanket threshold claim).
5. **Verification skill (deferred), line ~155**: "a cheap training-free confidence-remask diagnostic may run separately, at matched NFE, because confidence is a same-forward-pass signal (PRISM reports at 170M, PAPER arXiv 2510.01384; CoDR training-free)."
   - Do not present PRISM as training-free. Rewrite to separate: training-free confidence/remask diagnostic (e.g. CoDR, training-free) may run separately; PRISM (2510.01384) is a fine-tuned self-correction/remasking method reporting 170M MDM results, not a training-free diagnostic.

## Final verdict table

| Citation | Verdict | Quote (fetched) | Action on spec |
|---|---|---|---|
| arXiv:2605.26106 (LoopMDM) | CONFIRMED (170M, loop inside denoising step, 3.3x FLOPs); MISLEADING ("2-of-16 style") | "using a 170M-parameter diffusion transformer..." / "12-layer LoopMDM loops two (first and second) mid-layers 1-2 in zero-based indexing..." / "matches the performance of same-size MDMs with up to 3.3× fewer training FLOPs..." | Reword mechanism to paper-accurate looped mid-block (2 layers in 12-layer 170M), remove/qualify "2-of-16 style". |
| arXiv:2610.04198 (ALoDLM) | CONFIRMED (iterative refinement/recurrence inside denoising step) | "At each denoising step, ALoDLM iteratively refines latent representations..." / "unresolved tokens retain and further refine their latent states through additional recurrent passes." | Keep as arrangement evidence; note adaptive/token-wise recurrence at 1.7B/8B (not identical to fixed S mid-block). |
| arXiv:2610.30066 (Looped UDLM) | CONFIRMED | "selective looping, which recursively applies a few early Transformer layers within each denoising step..." | No change needed for arrangement claim. |
| arXiv:2610.18127 (Monitoring latent reasoning in looped LMs) | CONFIRMED (facts/numbers); MISLEADING (AR vs diffusion transfer) | "examine eight looped LMs from four families... 1-3.5B scale..." / "In six out of the eight models, 76-89% of the generated tokens already settle on their final top-1 token in the first iteration." / "The reasoning-tuned Ouro-Thinking models are best at iteration 1 and get worse with every iteration." | Explicitly label as looped AR LMs (not diffusion) in risk text. |
| arXiv:2604.07822 (Loop, Think, & Generalize) | CONFIRMED (trained from scratch, three-stage grokking); MISLEADING (general ">~15 iterations") | "Through controlled studies with models trained from scratch..." / "Systematic generalization emerges through a three-stage grokking process..." / "increasing iterations beyond 15 for the R=dynamic does not improve OOD performance..." | Qualify overthinking threshold as setting-specific example, keep grokking/scratch claims. |
| arXiv:2502.02737 (SmolLM2) | CONFIRMED (22.6, SmolLM2-1.7B base HumanEval) + WRONG table ref ("Table 3") + UNCONFIRMED (exact protocol detail) | Table 4 (base): "| HumanEval | 22.6 | 18.9 | 37.2 |" (SmolLM2 1.7B base column) / "Evaluation results in Table 4..." (base models). | Change spec G0 reference from Table 3 to Table 4; record protocol caveat. |
| arXiv:2108.07732 (MBPP) | CONFIRMED (974 tasks, 500 test split); UNCONFIRMED (Appendix A copyable template text) | "The Mostly Basic Programming Problems (MBPP) dataset contains 974 programming tasks..." / "hold out 10 problems for few-shot prompting, another 500 as our test dataset..." | Keep 500-split claim; require verifying/copying exact MBPP template from paper before run (do not assume Appendix A text). |
| arXiv:2510.01384 (PRISM) | CONFIRMED (reports at 170M MDM) + WRONG/MISLEADING (training-free diagnostic attribution) | "PRISM... attaches a lightweight adapter to any pretrained MDM and fine-tunes with this loss..." / "Empirically, PRISM advances MDM inference across domains and scales: Sudoku; unconditional text (170M); and code with LLaDA (8B)." | Do not cite PRISM as training-free diagnostic; separate from CoDR/training-free confidence-remask. |

## Spec lines that must change (concise list)

- `spec/edgefast-diff-100m.md:118` (L1v2 mechanism): remove/replace non-paper "2-of-16 style" detail; make LoopMDM description paper-accurate (170M MDM, 2 early-middle layers looped inside denoising step in 12-layer setup, up to 3.3x fewer training FLOPs); qualify 2610.04198/2610.30066 as same broad arrangement class.
- `spec/edgefast-diff-100m.md:123` (G0): `Table 3` -> `Table 4` (arXiv:2502.02737v1), add protocol-confirmation note.
- `spec/edgefast-diff-100m.md:130,144` (MBPP template): mark exact template text as UNCONFIRMED until copied from 2108.07732; keep test split = 500 (confirmed).
- `spec/edgefast-diff-100m.md:146` (Accepted risks): label 2610.18127 as looped AR LMs (1-3.5B, not diffusion); qualify 2604.07822 overthinking threshold as setting-specific.
- `spec/edgefast-diff-100m.md:155` (Deferred verification skill): stop calling PRISM a training-free confidence-remask diagnostic; cite PRISM as fine-tuned 170M remasking/self-correction evidence, separate from training-free CoDR-style diagnostic.

## Fetch notes

- Direct fetches succeeded for all eight arXiv abs pages and HTML pages for 2605.26106v1, 2610.18127v1, 2604.07822v2, 2502.02737v1, 2108.07732v1, 2510.01384v4 (HTML content partially retrieved by tool truncation in some cases; claims above use only text visible in fetched outputs). No web search used.
- 2108.07732 Appendix A template body could not be fully confirmed from retrieved HTML excerpts in this pass -> marked UNCONFIRMED.
- 2502.02737 exact HumanEval protocol details for Table 4 not fully recoverable from retrieved excerpts -> marked UNCONFIRMED detail.

Commit made by iceman-free (opencode/iceman-free) in opencode on behalf of Kriday.
