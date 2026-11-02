# EdgeFast-Diff 100M — block-diffusion hybrid spec (v3)

Revised 2026-11-01: target size retargeted from 1B to 100M (user call). With zero budget, the 100M is the real, trainable model — not just a pilot. A 1B scale-up stays future work if compute appears. Supersedes v2 of this spec (1B target) and `edgefast-1.5b.md` (AR edge model, kept for reference). Written 2026-10-28 after the hybrid-attention and diffusion-decoding research passes.

## Goal

Maximum single-stream TPS. One guardrail: pass@1 on HumanEval tracked at every step, because a model that outputs garbage at 50k tok/s wins nothing.

The governing formula:

> TPS ≈ bandwidth ÷ bytes read per forward pass × tokens emitted per forward pass

The design attacks both terms: a mostly-delta-rule backbone shrinks the per-pass read (no growing KV cache), and block-diffusion decoding emits several tokens per pass (Mercury's trick, LLaDA's missed opportunity).

## Baseline

B0: Qwen3.5-0.8B (researched 2026-10-30: Qwen3.5 small series is 0.8B / 2B / 4B / 9B; 0.8B is the closest small hybrid). fp16, vanilla decode, batch 1. Everything is measured against it: same GPU, batch 1, 1k in / 1k out, median of 5 runs. B1 (int4) and B2 (EAGLE-3) build on the same Qwen model, not on a separately trained baseline. Known caveat: D1 is 100M vs B0's 0.8B — 8x smaller, so a TPS win over B0 is expected from size alone and proves little by itself. The honest size-matched comparison is against SmolLM2-135M (see benchmark week); B0 is the "latest small hybrid" reference point. The real guardrail is quality: D1's pass@1 will be far below B0's, and the gap is tracked, not hidden. No official code benchmarks exist for Qwen3.5-0.8B, so its HumanEval pass@1 is measured by us. Reference numbers in this doc are H100-scale; all actual benchmark runs happen on Colab's free T4 (see hardware section).

| Step | Config | Expected vs B0 |
|---|---|---|
| B0 | vanilla AR (Qwen3.5-0.8B fp16) | 1x (the reference) |
| B1 | B0 + int4 QAT | ~2x |
| B2 | B0 + int4 + EAGLE-3 | 3-6x, the honest off-the-shelf competitor |
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

1. Reproduce AR baselines at ~100M: SmolLM2-135M (yardstick) and the latest small hybrid Qwen3.5-0.8B (GDN backbone, same family as D1) + LiquidAI LFM2.5-350M. B1 (int4 QAT) and B2 (int4 + EAGLE-3, budget a day of friction) build on the Qwen3.5-0.8B. No Qwen model exists near 100M; size gap is acknowledged.
2. Size-matched diffusion vs AR at ~100M: HoangHa/smoldlm-144m (144M, block diffusion, block_size 32, Apache 2.0, latest) vs SmolLM2-135M. OpenSML/OpenSML-150M (150M, AR, Sep 2026) only if CUDA-portable (MLX by default). These numbers are the whole scoreboard for D1's own scale.
3. Paradigm check at scale: LLaDA-8B (default config AND Fast-dLLM decoding) vs Qwen3.5-9B as AR defender. Dream 7B double run optional. DiffusionGemma 26B-A4B optional (skip honestly if no fit). If diffusion can't win after tuning decode policy at 8B, D1 pivots to B2 + competence training.
4. Optionally run the fixed-state authors' setup if code exists (nearest proxy for D1).

## Hardware reality

Budget status (updated 2026-10-30): no training budget. Lightning AI's free GPU tier is gone. Everything runs on free tiers until credits or money appear.

- Benchmarks run on Colab free T4 (16 GB, sm75, ~320 GB/s), chosen for ease of use. Identical silicon to Kaggle's T4, and the second T4 was never used (batch 1, one benchmark per GPU), so the original math is unchanged.
- The 100M pretrain is the plan now (Kaggle T4 free tier), not a rehearsal for one. A 1B run (rented H100, ~$1-1.5k at 200-300B tokens) stays future work if credits or grants appear. The speed thesis (TPS) is fully testable at 100M; only 1B-scale pass@1 quality needs the bigger run.
- Zero-budget sequencing: benchmark week on Colab → pretrain D1 (100M, same architecture as specced) free on Kaggle T4 (30h/week, checkpoints to Drive every N steps; Colab's short sessions make it bad for training) → with proof the physics works, apply for compute grants or rent for the 1B scale-up.

T4 consequences (unchanged):

- T4 has no bf16 and no AWQ support. Full-precision tier is fp16; quantized tier is GPTQ int4 or llama.cpp Q4_K_M. Turing also misses the Ampere+ attention kernels, so absolute TPS will sit below the bandwidth math; ratios between configs are the deliverable, and those carry over.
- 8B models fit: Q4 is ~4.5-5 GB. LLaDA-8B runs via the diffuse-cpp GGUF fork (community conversion with the diffusion decode loop). EAGLE-3's reference stacks target newer GPUs; B2 may need llama.cpp spec-decode or an older vLLM. Budget a day of friction for B2.
- The TPU is excluded from everything: the entire stack (vLLM, SGLang, llama.cpp, EAGLE, Fast-dLLM, diffuse-cpp) is CUDA, and porting decode loops to JAX/XLA buys nothing even for free. This project is PyTorch/CUDA.

## Pilot experiment L1: looped core (decided 2026-10-30)

User decision on record: if D1 passes, the looped + verification direction is the follow-up, no further debate.

Standard pilots rehearse a proven design at small scale. Ours can't: 100M models barely think, so a scaled-down rehearsal hides nothing but proves little. L1 gives the pilot a real job instead: test the idea that makes small models smarter per parameter.

**Mechanism.** Block diffusion already runs the model 8-16 times per block (denoising passes). L1 ties those passes to a shared, weight-tied core: a looped model whose loop iterations ARE the denoising steps. Extra effective depth at zero extra passes, so TPS is untouched by construction.

**Hypothesis.** At equal TPS, a looped block-diffusion twin beats an unlooped twin on quality, and the gap widens as steps shrink. Looping attacks the few-step quality cliff (more compute per pass should flatten the 15.3@4-steps vs 6.4@32 curve from the fixed-state paper).

**Setup.** Two 100M twins, same data, same budget: (a) unlooped, all params distinct; (b) looped core re-used across denoising steps (fewer params, same inference cost). Metrics: generative PPL, pass@1 on MBPP (HumanEval stays sealed for final eval), TPS. Run both at 16 and 8 steps. Reference yardstick: SmolLM2-135M (researched 2026-10-30, the only real open model at this size; still HF's smallest as of late 2026). The twins are not expected to beat it, but the gap to it is the meaningful quality number, and the looped twin closing any of that gap is a headline L1 result. Runs free on a T4, so it joins benchmark week at zero cost.

**Kill criterion.** If (b) doesn't beat (a) on quality at equal TPS at either step count, looping is dropped and the pilot reverts to a plain pipeline rehearsal. One clean negative result, one weekend spent, nothing else changes.

**Explicit accepted gamble:** this stacks looping on top of the two already-experimental axes (GDN hybrid, block diffusion). Accepted because the pilot is cheap and a negative result kills only looping, not the project.

**Verification skill (deferred knob, not in L1).** Judging is easier than doing; a model can verify code far better than it writes it. The follow-up experiment trains an explicit verify-during-refinement signal (model scores its own masked slots; low-verified tokens re-mask; this doubles as the adaptive-steps-per-difficulty mechanism flagged in Risks). Not stacked onto L1, two new mechanisms in one run would make the result unreadable. Verification gets its own run after L1 lands.

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

Three buckets, user-approved: general web = FineWeb-Edu (`sample-10BT` slice); code = The Stack Dedup, Python subset (mined); math = FineMath or OpenWebMath. Mix for the ~3-5B token run: ~50% code / ~35% web / ~15% math (web stays a third — the cure for code-biased pretrain). Decontamination (n-gram check vs HumanEval) runs inside the prep script on everything. SFT data (Magicoder class) is recipe step 2, separate from this. Tokenizer: train our own 32k BPE on a corpus sample (pending user confirm). Workflow: ~100MB samples locally to build/test the prep script, full downloads on Colab/Kaggle where training happens.

## Deliverables

- B0/B1/B2 benchmark harness and results (week 1)
- Pretrained 100M checkpoint (GDN hybrid, block-diffusion objective)
- Custom decode loop (confidence-threshold unmasking, block-level early stop at 0.97, prompt cache, block overlap, block scheduler with tunable steps)
- TPS + pass@1 report against B0/B1/B2
