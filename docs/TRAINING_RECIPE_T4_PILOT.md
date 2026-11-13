# D1 training recipe and T4 pilot budget

Draft for review, 2026-11-13. This document is the contract for the training pipeline being built on `t3/training-pipeline-v1-shard-mask-resume`. It sets the model and masking assumptions, the staged pilot, the checkpoint and resume rules, the evaluation gates, and the T4 compute budget with its uncertainty. It changes no code. It touches no benchmark, cache, decode, or evaluation file.

Labels follow repo practice. MEASURED means a repo run produced it. PAPER means a cited primary source. DERIVED means arithmetic from repo numbers. ESTIMATE means a guess that the probe must replace. UNCONFIRMED means not yet checked.

## 1. What this covers

In scope: the D1 100M block-diffusion pretrain recipe, a staged pilot on the training-v1 corpus, checkpoint and resume, evaluation gates, and the T4 compute budget with a measurement plan.

Out of scope: code in `bench/`, decode backends, KV-cache logic, and eval harnesses. Those belong to other tracks, and their outputs arrive here frozen. The sampler track owns temperature, commit mode, step count, block size, and confidence. The pipeline track owns loading, masking, training, and checkpointing.

The project has no paid compute. GPU time comes from free tiers: Kaggle T4 at 30 GPU-hours per week, per `research/quality-fixplan-ernie.md` section 6, or Colab T4 in sessions of up to 12 hours that die without warning.

## 2. Inputs the recipe starts from

### 2.1 Corpus: training-v1

All facts below come from `artifacts/training-v1/final/manifest.json`, `audit.json`, `tokenizer_manifest.json`, and `docs/TRAINING_DATA.md` in the project checkout. The corpus is gitignored, so the pipeline consumes the artifact folder, not the repo. The bundle currently lives at `/home/kriday/Code/AI/fast-af-boi/artifacts/training-v1/final`; the training job must be given that path or a verified copy, and must recheck the manifest fingerprint and shard hashes at load.

| Fact | Value |
|---|---|
| Train tokens | 99,983,360 |
| Validation tokens | 983,040 |
| Train sequences | 97,640 |
| Validation sequences | 960 |
| Seq len | 1024 |
| Mix by tokens | 50% code, 35% web, 15% math |
| Interleave | seeded 20-sequence cycles, 10 code, 7 web, 3 math |
| Shards | 10 train, last one partial with 5,480 sequences; 1 validation |
| Format | little-endian uint16, one row per 1024-token sequence |
| Tokenizer | 32,000 byte-level BPE, SHA-256 `5bb6b95687e59ae2c8a52c7ec8e664574aaffae8875f567a7335d9d666e3de0b` |
| Special ids | MASK 0, EOS 1, PAD 2, UNK 3, BOS 4 |
| Audit | passed, 0 detected benchmark overlaps, 0 exact or near duplicates among 123,064 docs |
| Roundtrip | 0 tokenizer failures, 0 unknown tokens in validation |
| Manifest fingerprint | `6f62a175ac05648cdef20684fba9038105a906333a0bcea57f81c3f48a172ab4` |

The audit scope is 13-word n-grams, normalized substrings of at least 32 characters, and conservative 64-bit SimHash with Hamming distance at most 3. It covers HumanEval-164 and all 974 MBPP rows. Zero detections does not prove zero semantic contamination. Research on soft contamination, arXiv 2602.12413, reports that no single standard filter removes more than 6.5% of paraphrased duplicates. One stronger pass, docstring prefix or embedding similarity, must run before G3, and the residual must be reported.

Source pools at the pinned revisions currently bind the unique-token ceiling. Web candidate tokens are 48,534,206, and web is 35% of the mix, so the current exports support about 138.7M unique training tokens (DERIVED). Code supports 155.7M and math 206.1M at their quotas. A 100M-token smoke run needs no new exports. A 200M run does unless epochs on the clean set are accepted; past that, either fetch more pinned shards or accept epochs.

### 2.2 Model: D1

From `spec/edgefast-diff-100m.md`:

| Component | Assumption |
|---|---|
| Size | about 100M params, expected near 105M, exact count verified in code before freezing |
| Blocks | 16, d_model 640 |
| Linear layers | 12 Gated DeltaNet |
| Attention layers | 4, full attention, GQA with 2 KV heads, head dim 128, NoPE, at blocks 4, 8, 12, 16 |
| FFN | SwiGLU, hidden 1728 |
| Norm | RMSNorm, pre-norm |
| Vocab | 32,000, tied input and output embeddings |
| Train context | 1024 for the pilot; architecture supports 32k, but 32k is not tested here |
| Precision | fp16 training, fp32 master weights for P0-P2; int4 QAT deferred (see below) |

The tied 32k table is about 20.5M params, roughly a fifth of the target. GDN internals are not pinned in the spec, so the final count could land anywhere from about 100M to about 130M. The pipeline must print the exact count and hash the config before the first training step. Every compute number below is quoted per 100M and scales linearly with the verified count.

Two spec/recipe reconciliations. First, the spec's architecture table lists int4 QAT from the start, but the quality fix-plan explicitly says not to chase int4 QAT before a quality baseline (`research/quality-fixplan-ernie.md` section 4, item 6). This recipe therefore freezes fp16 for P0-P2 and treats int4 QAT as a post-P2 product decision; if it is required, run a separate QAT pass after P1, not before. Second, the spec's train context is 32k, but the quality plan says long context is a v2 problem and all evals run at 1k (section 4, item 7). The pilot trains and evaluates at 1024; extending context is a P3 decision.

### 2.3 Decode policy

Frozen elsewhere. The pipeline receives a decode-policy hash covering temperature, commit mode, steps, block size, confidence threshold, and prompt templates. The frozen policy is an input to G3 and to every evaluation gate. This recipe does not tune it.

## 3. Masking and objective

Stored token ids are clean targets. They contain no MASK, PAD, UNK, or BOS. Corruption happens in the training loop and is never written back to the shards. The corpus is already built this way.

Default for P1, classical random masking:

1. Take one packed 1024-token sequence from a shard.
2. Sample one corruption rate r per sequence, drawn uniformly from 0.10 to 0.90 (PROPOSED; the probe can narrow it).
3. Replace each token with MASK with probability r, except EOS and reserved ids.
4. Score cross-entropy on masked positions only, averaged over masked tokens in the batch.
5. No causal mask. Every position may attend to every position.

This is the proven default and the one the spec calls boring on purpose. It trains the model at all noise levels. The loss is plain mean cross-entropy, with no importance weighting by r. No chain of denoising steps is unrolled during pretraining. Few-round behavior is taught later in SFT and self-distillation, per spec recipe steps 2-3.

Alternative, blockwise-aligned masking:

1. Split the sequence into aligned blocks of 32 tokens.
2. Choose one active block, or cycle through blocks by step.
3. Prefix stays clean. Active block becomes MASK. Suffix stays clean in storage but is hidden by an attention mask.
4. Score cross-entropy on the active block only.

This matches the decode-time pattern: clean prefix, bidirectional active block, hidden suffix. It costs a little more compute per scored token because the suffix is processed but not scored. Evidence is LLaDA-8B-Instruct with LoRA, arXiv 2508.19529, where the gain peaks when the training block size equals the decode block size. The magnitude at 100M from scratch is unmeasured. Treat it as a pilot decision, not a default.

Attention mask contract: the corpus ships all-ones masks. The pipeline must build the mask the chosen objective needs. Classical masking uses all ones. Blockwise alignment uses a custom mask. The mask mode must be logged in every run config.

EOS is never masked. Documents stay separated by EOS inside a packed sequence. Never pack across shard boundaries.

## 4. Training recipe

Values marked PROPOSED are standard defaults, not measured on T4. The probe can move them. Values marked FREEZE are decided here or by G3.

| Setting | P1 value | Status |
|---|---|---|
| Precision | fp16 compute, fp32 master, AMP scaler | FREEZE for P0-P2; int4 QAT deferred |
| Optimizer | AdamW, betas 0.9 and 0.95, eps 1e-8 | PROPOSED |
| Weight decay | 0.1 on matrices, not norms, biases, or embeddings | PROPOSED |
| Peak learning rate | 1e-3, range 5e-4 to 2e-3 | OPEN, probe screen |
| Schedule | linear warmup 2% of steps, cosine decay to 10% of peak | PROPOSED |
| Minimum learning rate | 1e-4 | PROPOSED |
| Global batch | 64 sequences x 1024 tokens, 65,536 tokens per step | FREEZE |
| Micro-batch | 8 sequences, accumulation 8; accumulation adjusted for 4 or 16 to keep global 64 | PROPOSED, optional 50-step screen at 4/16 |
| Gradient clipping | global norm 1.0 | PROPOSED |
| Dropout | 0.0 | PROPOSED |
| Init | normal 0.02, zero bias, residual outputs scaled by 1/sqrt(2*n_layers) | PROPOSED |
| Weight tying | input and output embeddings shared | FREEZE |
| Data order | seeded shuffle, data seed logged, corpus seed 17 recorded | FREEZE |
| Run seeds | P1 base 1001; twin arms 2001/2002; add 3001/3002 for a second seed per arm if affordable | PROPOSED |
| Steps | P1 100M = 1,525; P1 200M = 3,050; P2 500M = 7,629 per arm | FREEZE |
| Checkpoint | every 250 steps (16,384,000 tokens) or 30 minutes, whichever comes first | FREEZE |
| Eval cadence | validation NLL every 250 steps; gPPL and MBPP at the end | FREEZE |
| Logging | every 10 steps | FREEZE |

Step math: 1,525 steps x 65,536 tokens = 99,942,400 tokens, just under one epoch of the 99,983,360-token corpus. 3,050 steps = 199,884,800 tokens. 7,629 steps = 499,974,144 tokens per arm. Small overshoots are fine, but the run must log actual tokens seen.

Memory: about 1.6 to 2.5 GB for weights, master copy, gradients, and Adam states, plus activations. Micro-batch 8 should fit the 16 GB T4 with fallback kernels. The probe confirms and picks the micro-batch.

## 5. Pilot stages

### P0: compute probe and config screen

Run 200 steps at the production batch, sequence length, and mask mode, with micro-batch 8: 50 warmup steps, then three 50-step timed windows. Measure throughput. Optionally run a 50-step screen at micro-batch 4 and 16; that adds 100 steps (6.6M tokens) and 0.1-0.4 T4-h, outside the 22.9M base. Optionally run 50-step screens at three learning rates to pick the peak LR.

Protocol:

1. 50 warmup steps.
2. Three timed windows of 50 steps each (150 timed steps, 200 total).
3. Sync CUDA before and after each window.
4. Count tokens as global batch times steps in the window.
5. Report median tokens per second, spread, effective TFLOPS, and model FLOPs utilization.

Model FLOPs utilization, or MFU, is effective TFLOPS divided by the T4's 65 TFLOPS fp16 peak. It tells us how much of the chip we actually use.

Gate bands:

| Measured throughput | Decision |
|---|---|
| At least 13,000 tok/s | green. Run the pilot at planned tokens. |
| 6,000 to 13,000 tok/s | yellow. Cut the pilot token budget or change block and sequence settings. |
| Below 6,000 tok/s | red. Stop. Change config before spending weeks. |

This matches the week-2 probe in `research/quality-fixplan-ernie.md` and `research/l1-fix.md`. The probe costs about 0.3 T4-hours at the optimistic throughput and 0.7 at the floor. The LR screen adds about 0.2 and 0.5.

### P1: base D1 smoke pretrain

Train the unlooped D1 on the training-v1 corpus, 100M tokens first. This proves the whole path: shard loading, dynamic masking, forward and backward, checkpoint, resume, validation NLL, one gPPL cell, and one decode sanity check.

Gates:

1. No NaN and no loss explosion. Smoothed loss falls.
2. Validation NLL at start, midpoint, and end decreases. Score all 983,040 validation tokens with fixed mask seeds and the training mask mode.
3. Decode sanity at T=8 and T=16 on a small fixed text set. tokens_emitted near the target length, tokens_per_pass near 4.0 at T=8 and 2.0 at T=16 for block 32, entropy above the frozen floor (default 5 nats), no collapse into repeated punctuation. This uses the frozen decode policy and does not tune it.
4. Checkpoint and resume verified by a controlled kill and restart. Step, tokens seen, shard cursor, and RNG states restore correctly.
5. Exact param count, FLOPs per token, and config hash logged and matched.
6. Eval GPU-hours logged separately from training GPU-hours.

Pass: proceed to P2. Fail: fix the pipeline and rerun P1 once. A second failure stops the pretrain plan, and the project reviews the architecture and data assumptions.

### P2: L1v2 twins, only if P1 passes

Two about-100M twins on identical shards, identical optimizer family, identical training plan.

- Arm A: unlooped control, S=1.
- Arm B: 2 of 16 layers looped inside each denoising step, S_train fixed before the run, default 4. S_decode swept over 1, 2, 4.
- Decode at T=8 and T=16 with the frozen decode policy.
- Token budget 500M per arm if the probe supports it. 1B per arm only as a stretch after a stable 500M run.
- Matched total training FLOPs within plus or minus 10%, ratio reported. Matched inference FLOPs within plus or minus 10% for any win claim.
- One seed is allowed but must be labeled single-seed, low confidence, and needs larger margins. Two seeds preferred.

Endpoints and kill criteria come from `spec/edgefast-diff-100m.md`:

- PASS: arm B beats arm A on validation NLL at 8 steps at matched F_train, in both seeds, or by at least 1.0% in a single-seed run with the gPPL direction agreeing. Convergence diagnostics show S greater than 1 contributes. Arm B is no worse at 16 steps.
- FAIL: no validation NLL win at matched F_train at 8 steps, delta below 0.5% single-seed or split across seeds.
- INCONCLUSIVE: divergence, gate failure, protocol violation. One rerun at pilot scale per root cause. Equal TPS is never a pass condition.

### P3: full pretrain, outside the pilot

Three billion tokens is the planning target from the quality fix-plan, inside the spec's 2-10B range, not the pilot size. Before starting it:

- P2 must produce a clean result.
- The corpus question must be settled. Current exports support about 138.7M unique tokens at the fixed mix. More needs new pinned shards or accepted epochs.
- The probe must show throughput that makes the run fit the free-tier calendar.
- The masking question from section 3 must be settled, ideally with a small mask twin.

## 6. Checkpoint and resume

Free-tier sessions die. Resume is part of the deliverable, not an afterthought.

Every checkpoint stores:

- fp16 weights and fp32 master weights.
- Adam moments, gradient scaler, and scheduler state.
- Python, NumPy, PyTorch CPU, and CUDA RNG states.
- Dataloader position: shard index, offset in shard, shuffle permutation state, and the seed that produced it.
- Global step, tokens seen, wall time, cumulative GPU-hours, and average throughput.
- Config hash, git commit, library versions, and GPU type.

Interval: every 250 steps or 30 minutes, whichever comes first. At the optimistic throughput that is about 16.4M tokens and at most 21 minutes of lost work. A full checkpoint is roughly 1.4 to 2 GB for a 100M model, so retention matters. Keep the three most recent locally and on Drive, plus one snapshot every 500M tokens. Prune older files after verifying the newest loads.

Atomic write: write to a temporary file, flush and fsync, rename, then write a SHA-256 sidecar. Upload to Drive after the rename. Verify the hash on load. A corrupt checkpoint is discarded, and the previous good checkpoint is used.

Resume checks, all mandatory:

- Config hash matches the run config.
- Tokens seen is consistent with step count and global batch.
- Shard cursor is in range and reproduces the same sample order from the recorded permutation seed.
- RNG states are present and load without error.

If any check fails, the run resumes only for continued training if the loss can be bounded. Any twin comparison that cannot prove equal effective tokens and config is marked INCONCLUSIVE.

## 7. Evaluation gates

Full definitions live in the spec and in `research/l1-final.md`. Summary:

| Gate | What it proves | Pass condition |
|---|---|---|
| G0 | Harness calibration | SmolLM2-1.7B base scores 22.6 plus or minus 6 on HumanEval-164 through our harness. |
| G1 | Sampler regression | Temperature 0 reproduces the captured baseline with at least 99% token agreement. |
| G2 | Decode and world check | Decode supports temperature and commit, logs tokens emitted, passes, blocks, fills, and EOS stats. Author decoder and likelihood check separate World A from World B. |
| G3 | Recipe and pipeline freeze | Probe measured, smoke pretrain works, data and decode frozen, decontamination done with residual reported. |

If G2 lands in World A, MBPP is dropped as a decision endpoint. Validation NLL and gPPL carry the pilot.

Pilot-specific gates:

| Gate | Pass condition |
|---|---|
| P0 throughput | Green, yellow, or red band from section 5. |
| P1 val NLL | Decreases across three checkpoints. |
| P1 decode sanity | tokens_per_pass in band, entropy above floor, no degenerate repetition. |
| P1 resume | Controlled kill and restart reproduces step, tokens, cursor, and RNG state. |
| P2 quality | PASS or FAIL per spec, with matched FLOPs logged. |

HumanEval stays sealed for all D1 decisions until final evaluation. MBPP uses the test split of 500 problems, a frozen prompt template copied from arXiv 2108.07732 Appendix A, 256 max new tokens, and three seeds with median and min/max reported. The template text and hash must be recorded in every results file. It is UNCONFIRMED in the repo until copied.

Eval budget: bound it. Validation NLL is cheap, a few minutes per checkpoint. gPPL costs about 0.06 T4-hours per config. MBPP costs about 0.10 T4-hours per config, so three seeds is 0.3. Run MBPP only at final checkpoints, at T=8 and T=16, with the S_decode grid on arm B only. Plan 5 to 8 T4-hours for all evaluation, and log actual cost.

## 8. Compute budget

### Cost model

Training FLOPs per token = 6N for forward and backward, plus an attention-sequence term at seq 1024 of about 0.2N forward (0.4N with its backward). For N=100M that is about 6.6e8 FLOPs per token; the dense 6N floor is 6.0e8. Recompute from the logged param count before freezing. The token-based T4-hour table below is insensitive to this 10% term; the FLOPs number is for the P2 matched-compute check. This matches `research/quality-fixplan-ernie.md` section 6, which used 6N and called the attention term small.

Throughput anchors, all ESTIMATE until P0 measures them:

- A1: 11.43 ms per forward pass, fp16 batch 1, smoldlm-144m on T4. MEASURED-DERIVED from `results_t4.yaml`, 349.979 tok/s and 256 passes.
- A2: batch 8 raises per-pass throughput about 2.5x. ESTIMATE.
- A3: training at 13,000 tok/s, about 8-8.6 effective TFLOPS, about 12-13% MFU on the T4's 65 TFLOPS fp16 peak. ESTIMATE. The same source gives 5,000 tok/s as the floor.

These anchors are why the budget is a range and not a number.

### Budget table

Tokens per T4-hour: 46.8M at 13,000 tok/s, 18M at 5,000 tok/s.

| Item | Tokens | T4-h at 13,000 | T4-h at 5,000 |
|---|---:|---:|---:|
| P0 probe and LR screen | 22.9M | 0.5 | 1.3 |
| P1 smoke, 100M | 99.9M | 2.1 | 5.6 |
| P1 smoke, 200M | 199.9M | 4.3 | 11.1 |
| P2 twins, 500M per arm | 999.9M | 21.4 | 55.6 |
| Evaluation, bounded | | 5-8 | 5-8 |
| Contingency, 30% of P0-P2 training | | 7-8 | 19-21 |
| P3 pretrain, 3B, training only | 3,000M | 64.1 | 166.7 |
| P3 pretrain, 3B, with 30% contingency | 3,000M | 83 | 217 |

The P0 base is 200 probe steps plus 3x50 LR-screen steps = 350 steps / 22.9M tokens. The optional micro-batch screen at 4 and 16 adds 100 steps / 6.6M tokens and 0.1-0.4 T4-h. The budget-table contingency row is 30% of the full P0-P2 training subtotal (0.5+2.1+21.4 T4-h), not of P3; the minimal pilot uses 30% of its smaller P0+P1 subtotal. The P3 row adds its own 30%.

Plans:

| Plan | Content | T4-h, optimistic | T4-h, conservative | Wall clock at 30 h/week |
|---|---|---:|---:|---|
| Minimal pilot | P0 + P1 100M + eval + contingency | 8-11 | 14-17 | about 1 week |
| Full pilot | P0 + P1 100M + P2 + eval + contingency | 36-39 | 86-90 | 1.5-3 weeks, likely 2-5 with churn |
| Single 3B pretrain | P3 + 30% contingency | 83 | 217 | 2.5-4 weeks optimistic, 6-8 conservative |

The 3B row includes its own 30% contingency; the P0-P2 rows are training-only and use the separate contingency row.

The full pilot (P0 + P1 100M + P2) is the recommended commit. It answers whether looping helps at our scale, proves the pipeline, and leaves a clean negative or positive. The 3B run is a separate decision after P2 and after the corpus grows.

### How to measure and update

The probe is the instrument. After P0:

1. Replace 13,000 with the measured median throughput.
2. Recompute every line in the budget table.
3. Freeze the revised table in the run config before P1 or P2 spends real hours.
4. During training, log cumulative tokens seen, wall time, and GPU-hours at every checkpoint. Compute the trailing average throughput. If it drifts more than 20% from the probe, flag it and re-estimate.
5. Report final GPU-hours separately for training, evaluation, and failed or restarted work.

Uncertainty, stated plainly:

- The dominant term is throughput. The 5,000 to 13,000 range is a factor of 2.6 and it is an ESTIMATE from secondary arithmetic.
- The second term is overhead: checkpoint I/O, evaluation, and session deaths. The 30% contingency covers it.
- The third term is the token budget for P2, which is chosen after the probe.
- After P0, throughput uncertainty drops to measurement noise, about plus or minus 10%, and the remaining budget uncertainty is roughly plus or minus 30% including overheads. Quote ranges, not points.

## 9. Assumptions and open decisions

Assumptions this recipe makes:

1. The pipeline branch builds and owns loading, masking, training, and checkpointing. This document is its contract.
2. fp16 training on T4, because T4 has no bf16; int4 QAT is deferred until after P2, and the pilot trains at 1024 context rather than the spec's 32k.
3. The corpus is reused as built for P1. The audit passed, but semantic contamination is not ruled out. One stronger pass must run before G3.
4. Classical random masking is the P1 default. Blockwise alignment is the untested alternative.
5. Throughput anchors are ESTIMATES until P0 replaces them.
6. Free-tier wall clock is 30 GPU-hours per week on Kaggle, with Colab sessions dying early.
7. The decode policy arrives frozen from the sampler track. This recipe does not tune it.
8. HumanEval stays sealed until final evaluation.

Open decisions for Kriday:

1. Corpus mix. Keep 50/35/15 as built, or shift toward 30-35% code, 45-50% web, 15-20% math, as `research/hillclimb-quality-ernie.md` recommends. Shifting means rebuilding and reauditing the corpus, and it invalidates the current hashes.
2. Mask mode. Classical random or blockwise-aligned. The cheapest answer is a 100-200M-token mask twin after P1, or folding the question into P2 planning.
3. Peak learning rate. 1e-3 proposed, range 5e-4 to 2e-3. A 50-step screen settles it for about 0.2 to 0.5 T4-hours.
4. Pilot size. 100M for P1, then 500M per arm for P2, subject to the probe. 1B per arm only as a stretch.
5. GDN internals and exact param count. The spec requires verification in code before freezing.
6. Micro-batch. The probe picks among 4, 8, and 16.
7. Number of seeds. Two is better. One is allowed with larger margins and a low-confidence label.
8. Kernel stack. Installing flash-linear-attention and causal-conv1d on T4 may raise throughput materially and would change the probe target. The current repo measurements are on the lean stack without them.
9. Checkpoint retention and Drive space. A checkpoint is 1.4 to 2 GB, so retention needs a decision.
10. Whether to run the full 3B pretrain at all if P2 fails. The fallback is a single base pretrain and a published negative result.
11. Precision. The spec lists int4 QAT from the start; the quality plan says defer it until a baseline exists. This recipe freezes fp16 for P0-P2. Decide whether int4 QAT is a P3 gate or a post-P2 product run.
12. Context. The spec's train context is 32k; the pilot uses 1024. Decide whether to extend context before P3, and budget the extra compute.

## 10. Source ledger

Repo facts: `artifacts/training-v1/final/manifest.json`, `audit.json`, `tokenizer_manifest.json`, `docs/TRAINING_DATA.md`, `spec/edgefast-diff-100m.md`, `research/l1-final.md`, `research/l1-fix.md`, `research/quality-fixplan-ernie.md`, `research/hillclimb-quality-ernie.md`, `results_t4.yaml`.

External sources cited: arXiv 2508.19529 (blockwise SFT, LLaDA-8B-Instruct LoRA), arXiv 2602.12413 (soft contamination; no single standard filter removes more than 6.5%), arXiv 2503.03862 and REGMIX (mix guidance), arXiv 2605.26106 (LoopMDM, 170M), arXiv 2610.18127 and 2604.07822 (looping counter-evidence), arXiv 2108.07732 (MBPP template), arXiv 2502.02737 (SmolLM2 calibration). Full ledgers live in `research/hillclimb-quality-ernie.md` and `research/l1-final.md`.

Numbers not from those sources are labeled DERIVED or ESTIMATE in place. Nothing in this document was measured on a GPU in this session.
