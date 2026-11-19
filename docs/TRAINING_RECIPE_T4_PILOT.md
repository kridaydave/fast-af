# D1 training recipe and T4 pilot budget

**Supervision update:** [all-block training](D1_SUPERVISION.md) now computes a
shared differentiable clean stream and trains every noisy block. Select it with
`--d1-supervision all-blocks`; legacy frontier masking remains the compatibility
default. The inherited pilot token counts and timings below describe frontier
supervision and must not budget the new path. Use its own scored-target rate,
measured memory and matched-compute learning checks before approving a run.


Updated 2026-11-15 with measured T4 infrastructure results; pilot authorization remains gated. This document is the contract for the training pipeline being built on `t3/training-pipeline-v1-shard-mask-resume`. It sets the model and masking assumptions, the staged pilot, the checkpoint and resume rules, the evaluation gates, and the T4 compute budget with its uncertainty. The implemented hybrid and blockwise objective are described in `D1_ARCHITECTURE.md`; the native chunk-v2 backend passed the 200-attempt P0 speed gate. The evaluation policy and production wall-time, checkpoint and upload controls still block P1. Current quota planning uses 30 Kaggle GPU-session hours per week for the T4 pair, governed by PROJECT_PLAN.md; the former six-hour cumulative cap is superseded. Review findings and evidence are recorded in `TRAINING_INTEGRATION_REVIEW.md`.

The [project plan](PROJECT_PLAN.md) governs the general-purpose base, specialists
and eventual roughly 2B-total-parameter MoE. **D1's 100M label refers to model
parameters**; the instantiated model has 109,137,632. This document's separate
approximately 100M-token pilot is an inherited infrastructure proposal using
training-v1. It is not the user's model-size requirement or an approved data
recipe/token budget for the new general base. The historical six-hour allocation below described that conditional D1 smoke
plan. It is superseded by the renewable weekly budget; it is not an active
project or provider quota ceiling.

The newer Tensor Core and two-T4 variants have a separate
[bounded measurement record](../.verify/d1-t4-parallel/README.md) and
[execution guide](T4_PARALLEL.md). They require their own P0 and quality
requalification. The current project ledger includes subsequent R&D; the old
future reservation below is historical and must not be added to a fresh budget.

Labels follow repo practice. MEASURED means a repo run produced it. PAPER means a cited primary source. DERIVED means arithmetic from repo numbers. ESTIMATE means a guess that the probe must replace. UNCONFIRMED means not yet checked.

## 1. What this covers

In scope: the D1 100M block-diffusion pretrain recipe, a staged pilot on the training-v1 corpus, checkpoint and resume, evaluation gates, and the T4 compute budget with a measurement plan.

Out of scope: code in `bench/`, decode backends, KV-cache logic, and eval harnesses. Those belong to other tracks, and their outputs arrive here frozen. The sampler track owns temperature, commit mode, step count, block size, and confidence. The pipeline track owns loading, masking, training, and checkpointing.

The project has no paid compute. The budget assumes 30 GPU-hours per week and interruptible T4 sessions, as carried from `research/quality-fixplan-ernie.md` section 6. These are planning assumptions, not verified current provider quotas or hardware guarantees. Recheck availability before booking the pilot.

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

The audit scope is 13-word n-grams, normalized substrings of at least 32 characters, and conservative 64-bit SimHash with Hamming distance at most 3. It covers HumanEval-164 and all 974 MBPP rows. Zero detections does not prove zero semantic contamination. Research on soft contamination, arXiv 2602.12413, reports that no single standard filter removes more than 6.5% of paraphrased duplicates. The existing normalized-substring pass satisfies the substring option in the current G3 minimum. A docstring-prefix or embedding-similarity pass would be an additional semantic check, proposed before a long run; it has not been performed in this integration. Report that residual risk even when the existing audit passes.

Source pools at the pinned revisions currently bind the unique-token ceiling. Web candidate tokens are 48,534,206, and web is 35% of the mix, so the current exports support about 138.7M unique training tokens (DERIVED). Code supports 155.7M and math 206.1M at their quotas. A 100M-token smoke run needs no new exports. A 200M run does unless epochs on the clean set are accepted; past that, either fetch more pinned shards or accept epochs.

### 2.2 Model: D1

From `spec/edgefast-diff-100m.md`:

| Component | Assumption |
|---|---|
| Size | 109,137,632 parameters in the instantiated default 32k-vocabulary model |
| Blocks | 16, d_model 640 |
| Linear layers | 12 forward GDN; 4 heads, key dim 128, value dim 256, output gate, causal conv width 4 |
| Attention layers | 4 block-causal GQA, 10 query / 2 KV heads, head dim 64, NoPE; blocks 4, 8, 12, 16 |
| FFN | SwiGLU, hidden 1728 |
| Norm | RMSNorm, pre-norm |
| Vocab | 32,000, tied input and output embeddings |
| Train context | 1024 for the pilot; longer-context behavior is unvalidated |
| Precision | fp16 training, fp32 master weights for P0-P2; int4 QAT deferred (see below) |

The previous five-query/two-KV layout was invalid for standard GQA. The implemented ten-query/two-KV layout preserves width 640. Forward GDN supplies recurrent state; attention provides bidirectional context inside each block. Every layer is block-causal across blocks. Prefix states remain independent of later noisy blocks, and each denoising round reuses the same entry state before a final clean commit pass. This is a specific engineering choice, not a claim of reproducing bidirectional Mamba. Measured T4 infrastructure results are recorded below. See `D1_ARCHITECTURE.md` for code and validation commands.

The tied vocabulary table is 20,480,000 parameters. The exact full-model count includes GDN gates, convolution and normalization. Compute must be measured on that implementation and objective; parameter count alone is insufficient.

Two spec/recipe reconciliations. First, the earlier spec listed int4 QAT from the start, but the quality fix-plan explicitly says not to chase int4 QAT before a quality baseline (`research/quality-fixplan-ernie.md` section 4, item 6). This recipe therefore freezes fp16 for P0-P2 and treats int4 QAT as a post-P2 product decision; if it is required, run a separate QAT pass after P2, not before. Second, the earlier spec targeted 32k context, but the quality plan says long context is a v2 problem and all evals run at 1k (section 4, item 7). The pilot trains and evaluates at 1024; extending context is a P3 decision.

### 2.3 Decode policy

Frozen elsewhere. The pipeline receives a decode-policy hash covering temperature, commit mode, steps, block size, confidence threshold, and prompt templates. The frozen policy is an input to G3 and to every evaluation gate. This recipe does not tune it.

## 3. Masking and objective

Stored token ids are clean targets. They contain no MASK, PAD, UNK, or BOS. Corruption happens in the training loop and is never written back to the shards. The corpus is already built this way.

The implemented D1 objective is blockwise frontier masking:

1. Sample one shared active 32-token block per micro-batch, uniformly over blocks with eligible targets.
2. Keep the prefix clean. Sample a corruption rate from 0.10 to 0.90 per sequence and mask eligible positions in the active block independently. Force its first eligible mask when that row draws none.
3. Crop every block after the active block. Attention sees its own block and earlier blocks. GDN scans forward.
4. Score masked cross-entropy only in the active block, token-weighted over the accumulated global batch. Prefix tokens are processed but not scored.
5. At inference, reuse the same immutable clean-prefix cache for each denoising round and run a final clean block pass before committing the cache.

The default small smoke transformer retains classical all-position Bernoulli masking. Its objective is distinct from D1. Full bidirectional attention over every block is not the D1 training function.

The implementation reports `masked_ce` and `val_masked_ce`, not a sequence NLL or diffusion NELBO. No importance weighting or unrolled denoising chain is claimed. EOS remains an unmasked target boundary; the development decoder is fixed-length and does not establish learned EOS termination.

One sampled block gives far fewer scored tokens than full-sequence classical masking. Logs and checkpoints separately track consumed corpus tokens, processed prefix-plus-active tokens, and scored masked tokens. The following token counts and hour estimates are inherited context-consumption planning numbers, not measured scored-token budgets. They must be revised after P0 for this objective before P1 can be approved.

`--eval-full` visits every frontier of every validation batch with fixed corruption seeds. That covers the validation contexts and all frontier positions using sampled masks; it does not score every token with probability one. It is more expensive than the old one-forward validation estimate.

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
| Micro-batch | 8 sequences, accumulation 8 | Validated by production P0 on T4 |
| GDN backend and AMP scale | `chunk` version 2; `--amp-init-scale 8` | Production warmup adapted from 256 to 8; fresh scale-8 parity/resume passed |
| Gradient clipping | global norm 1.0 | PROPOSED |
| Dropout | 0.0 | PROPOSED |
| Init | normal 0.02, zero bias, residual outputs scaled by 1/sqrt(2*n_layers) | PROPOSED |
| Weight tying | input and output embeddings shared | FREEZE |
| Data order | seeded shuffle, data seed logged, corpus seed 17 recorded | FREEZE |
| Run seeds | P1 base 1001; paired twins both use 2001, then both use 3001 if affordable; log data and corruption seeds separately | PROPOSED |
| Steps | P1 100M = 1,525; P1 200M = 3,050; P2 arm A reference = 7,629, arm B set by matched FLOPs | P1 nominal; P2 OPEN pending probe |
| Checkpoint | every 250 steps (16,384,000 tokens) or 30 minutes, whichever comes first | FREEZE |
| Eval cadence | full-frontier validation at initialization, attempt 763 and attempt 1525; gPPL/MBPP and decode checks at the end | Six-hour P0/P1 plan; replaces the old 250-step full-validation cadence |
| Logging | every 10 steps | FREEZE |

Step math: 1,525 steps x 65,536 tokens = 99,942,400 tokens, just under one epoch of the 99,983,360-token corpus. 3,050 steps = 199,884,800 tokens. 7,629 full steps = 499,974,144 tokens for the unlooped reference arm. The loader emits a partial micro-batch at shard ends, so these are nominal token counts, not exact counts for arbitrary batch sizes. In particular, the real 5,480-row tail is not divisible by micro-batch 16. P1 should use 4 or 8 until a full-global-batch policy is implemented for 16. Small overshoots are fine, but the run must log actual tokens seen.

Memory: native AMP keeps fp32 parameters, fp32 gradients, and two fp32 Adam moments, about 1.6 GB for 100M parameters, plus activations and temporary autocast copies. It does not require a separately checkpointed fp16 master copy. The earlier 1.6 to 2.5 GB range is a planning allowance, not a measurement. The old backend rejected batch 8 under its 85% allocation margin. Version 2 fits batch 8: production P0 peaked at 12.23 GiB allocated. Allocator reservation reached 14.31 GiB; reservation includes reusable cached segments. Keep batch 8 / accumulation 8 for the measured configuration, and measure any fallback separately.

## 5. Pilot stages

### P0: compute probe and config screen

Run 200 steps at the production batch, sequence length, and mask mode, with micro-batch 8 and accumulation 8: 50 warmup steps, then three 50-step timed windows. Measure throughput. Optional capacity screens must fit the T4 memory margin and have a separate update and wall-time cap. The bounded screen already tried 1, 2, 4 and 8; it did not test 16. Optionally run 50-step screens at three learning rates to pick the peak LR.

Protocol:

1. 50 warmup steps.
2. Three timed windows of 50 steps each (150 timed steps, 200 total).
3. Sync CUDA before and after each window.
4. Count actual consumed, processed and scored tokens in each window, including partial batches; report throughput for each denominator. Log attempted batches, successful optimizer updates, and AMP overflows separately. Skipped updates consume tokens and compute but do not advance the LR schedule.
5. Report median tokens per second, spread, effective TFLOPS, and model FLOPs utilization.

Model FLOPs utilization, or MFU, is effective TFLOPS divided by the T4's 65 TFLOPS fp16 peak. It tells us how much of the chip we actually use.

Gate bands use **processed** tokens/s for hardware throughput. A loader token
that is cropped as future context does not count as model work. Use consumed
corpus tokens/s separately when budgeting a corpus epoch. Version 2 production P0
measured 7,763 to 7,862 processed tokens/s, clearing the 6,000 stop threshold.
Its 14,678 to 15,096 consumed tokens/s is a separate corpus-budget denominator.

Gate bands:

| Measured throughput | Decision |
|---|---|
| At least 13,000 tok/s | green. Run the pilot at planned tokens. |
| 6,000 to 13,000 tok/s | yellow. Cut the pilot token budget or change block and sequence settings. |
| Below 6,000 tok/s | red. Stop. Change config before spending weeks. |

This matches the week-2 probe in `research/quality-fixplan-ernie.md` and `research/l1-fix.md`. The 200-attempt production P0 completed in 0.25 T4-hours including warmup and its final checkpoint. Three 50-step LR screens project to about 0.18 to 0.19 hours before startup/checkpoint overhead; they were not run.

### P1: base D1 smoke pretrain

Train the unlooped D1 on the training-v1 corpus, 100M tokens first. This proves the whole path: shard loading, dynamic masking, forward and backward, checkpoint, resume, validation NLL, one gPPL cell, and one decode sanity check.

Gates:

1. No NaN and no loss explosion. Smoothed loss falls.
2. Validation masked CE at start, midpoint, and end decreases. Visit all validation contexts and frontiers with fixed mask seeds and the training mask mode; report scored-token counts.
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
- Use arm A at nominal 500M tokens as the compute-budget reference if the probe supports it. Set arm B tokens to `tokens_A * FLOPs_per_token_A / FLOPs_per_token_B`, rounded to whole updates, and compare at matched cumulative FLOPs. A 1B arm-A reference is a stretch only. Equal tokens are a separate, unmatched-compute diagnostic.
- Matched total training FLOPs within plus or minus 10%, ratio reported. For illustration only, equal-cost layers would give `(14 + 2*4)/16 = 1.375` times compute per token for B, and about 364M tokens for B at a 500M-token A budget. Actual hybrid layer costs, output projection, and backward unrolling must replace that illustration before freeze. Matched inference FLOPs within plus or minus 10% for any win claim.
- One seed is allowed but must be labeled single-seed, low confidence, and needs larger margins. Two seeds preferred.

Metric definition: validation masked CE is a teacher-forced measurement with no decode-round argument. It is the primary training comparison at matched cumulative training FLOPs. Generative PPL and decode sanity are evaluated separately at T=8 and T=16, under the frozen policy and matched inference FLOPs. The development decoder does not replace that policy.

The revised decision rule preserves the existing CE margins while separating generation evidence:

- PASS: B improves validation masked CE in both paired seeds, or by at least 1.0% for a labeled single-seed run. Generated-quality direction at T=8 must agree, with no deterioration at T=16 and matched training/inference compute. Diagnostics must show that extra iterations contribute. A CE win alone is not evidence of a few-round generation win.
- FAIL: no masked-CE improvement at matched training FLOPs, improvement below 0.5% for a single-seed run, or conflicting seeds. A generation win that requires higher unmatched inference compute cannot establish PASS.
- INCONCLUSIVE: a margin between those single-seed thresholds, divergence, gate failure, missing generation evidence or a protocol violation. Freeze the comparison protocol before P2; no new quality result is asserted here.

### P3: full pretrain, outside the pilot

Three billion tokens is the planning target from the quality fix-plan, inside the spec's 2-10B range, not the pilot size. Before starting it:

- P2 must produce a clean result.
- The corpus question must be settled. Current exports support about 138.7M unique tokens at the fixed mix. More needs new pinned shards or accepted epochs.
- The probe must show throughput that makes the run fit the free-tier calendar.
- The masking question from section 3 must be settled, ideally with a small mask twin.

## 6. Checkpoint and resume

Free-tier sessions die. Resume is part of the deliverable, not an afterthought.

Every checkpoint stores:

- fp32 model parameters for native AMP, with temporary fp16 compute casts recreated on load. No duplicate fp16 checkpoint weights are required.
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

If any check fails, reject resume. A deliberate warm start from model weights is a new run with reset optimizer and counters, and must not be labeled a continuation. Twin comparisons must prove matched cumulative training FLOPs and frozen protocol settings; token counts may differ by design and must both be reported.

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

Eval budget is bounded by section 8. Cached full-frontier validation is measured at 353.09 seconds for all 960 contexts, 365.42 seconds including checkpoint loading. The inherited gPPL estimate is 0.06 T4-hours per config and MBPP 0.10 per config, neither validated for D1. Three MBPP seeds at both T=8 and T=16 would therefore estimate 0.6 hours. Reserve at most 0.75 hours for external gPPL, MBPP and decode checks, and stop with incomplete gates if it is exhausted. The old 5-to-8-hour whole-study evaluation allowance is superseded for this capped P1. HumanEval remains sealed.

## 8. Compute budget

### Version 2 measurements, 2026-11-15

The optimized backend preserves the 109,137,632-parameter model, context 1024,
blockwise objective and global batch 64. It batches GDN work across chunks,
avoids repeated slice-gradient accumulation, uses chunks up to 128 tokens, and
projects only scored positions to the vocabulary. T4 now supports micro-batch
8 with accumulation 8 within the same 85% allocation criterion.

The real-T4 parity, checkpoint/resume, short configuration probe, production P0
and validation evidence is in [the optimization report](../.verify/d1-t4-speed/README.md).
PyTorch was 2.14.1+cu130, CUDA 13.0, Tesla T4 SM75. FP32 recurrent state and
triangular solves remain in use under fp16 autocast.

| Production P0 window | Consumed corpus tokens/s | Processed tokens/s | Scored tokens/s | Applied / attempted updates |
|---|---:|---:|---:|---:|
| Attempts 51 to 100 | 15,096 | 7,827 | 236.5 | 50 / 50 |
| Attempts 101 to 150 | 14,678 | 7,862 | 231.6 | 50 / 50 |
| Attempts 151 to 200 | 15,066 | 7,763 | 235.0 | 50 / 50 |

P0 consumed the actual full training-v1 bundle at fingerprint
`6f62a175ac05648cdef20684fba9038105a906333a0bcea57f81c3f48a172ab4`, not a repeated
small shard. All source hashes were checked. The first 50 attempts warmed up;
three 50-attempt windows followed, with CUDA synchronization at their boundaries.
All 150 timed updates applied. Five overflow skips occurred during warmup,
at attempts 19, 21, 24, 38 and 39. GradScaler handled them safely and the LR
scheduler did not advance for them. The 200 attempts therefore applied 195
updates, consuming 13,107,200 corpus tokens and processing 6,881,024.

Peak allocation was 12.23 GiB and reservation 14.31 GiB. Total P0 wall time was
896.6 seconds; the last checkpoint write, checksum and finalization took 14.6
seconds. These are MEASURED. The short full-context probe was 8,464 processed
tokens/s median, about 2.43x the old recommended configuration's 3,490.

The 6,000 processed-token/s stop gate is passed. The 13,000 green band is not.
This supports a capped P1 quality smoke pilot, not the old full P2/P3 commitment.

### Validation cost and semantics

The original full validation recomputed the entire clean prefix for all 32
frontiers. It exceeded a 630-second outer cap on the 960-context split, so its
whole-split time is **not measured**. That timeout is preserved in the report.

The checkpoint evaluator reuses immutable clean-prefix state, discards every
noisy candidate, and commits clean blocks before the next frontier. Every
context and frontier is still scored with the same mask RNG order and token
weighting. Its T4 comparison on 16 full contexts scored the same 8,078 tokens and differed
by 5.79e-6 CE. All 960 contexts then completed in 353.09 seconds, 365.42 seconds
including checkpoint/config loading. The split scored 491,072 positions. These
are MEASURED; three full checks project to 18.27 minutes including loads. This cache is for evaluation only; training prefix
computation and gradients are retained.

### Historical six-GPU-hour P0/P1 plan: superseded

The following table preserves the former planning assumption for reference.
Its six-hour cumulative ceiling and remaining-balance arithmetic are withdrawn.
Use [PROJECT_PLAN.md](PROJECT_PLAN.md) and
[weekly-budget.json](../.verify/d1-t4-parallel/weekly-budget.json) for current
30-hour weekly session-time planning. This does not authorize P1 or waive gates.

A 1,525-attempt P1 consumes 99,942,400 corpus tokens. Applying the three P0
window rates gives **1.84 to 1.89 GPU-hours for training alone**, DERIVED. Budget
corpus tokens with consumed throughput, rather than confusing them with model
work or loss tokens. A 200M-token P1 is not the default under this ceiling.

| Future stage | Maximum GPU-hours allocated |
|---|---:|
| P1, about 100M corpus tokens, including training slowdown/restart margin | 2.50 |
| Three optional 50-attempt LR screens | 0.25 |
| Full validation at initialization, midpoint and end | 0.40 |
| Bounded gPPL, MBPP and decode checks | 0.75 |
| Checkpoint I/O, transfer and runtime setup | 0.25 |
| Total future-stage caps | 4.15 |

These are allocation limits, not all measured durations. The training cap is
about 32% above the slowest measured projection and includes restart margin;
do not add another blanket 30% on top. The validation cap must use the measured
cached path. External generation/MBPP and network upload costs remain unmeasured
for D1. Stop and report incomplete gates if either exhausts its allocation.

The old ledger's Colab event durations remain evidence, but Colab use does not
consume Kaggle quota. The provider's weekly runtime counter is authoritative;
previous weeks' usage does not reduce a fresh allowance. Enforce current-week
remaining runtime and session chunks before P1 is enabled. The `--p1` guard
remains closed for operational and evaluation prerequisites, not an imposed
six-hour lifetime project cap. Do not reuse this historical table as a new
reservation or deduct it from the weekly budget.

P1 full validation moves to initialization, attempt 763 and attempt 1525. This
preserves the start/mid/end quality gate while replacing the old 250-step full
validation cadence. No quality threshold, scored frontier or validation context
is dropped. Checkpoints remain every 250 attempts plus the final checkpoint.

P2's matched twins at a 500M arm-A reference remain far beyond six hours: about
18.4 to 18.9 hours of unlooped-equivalent training alone at these rates, before
evaluation or contingency. The looped arm is unimplemented and unmeasured.
These historical estimates did not authorize P2 or a 3B pretrain. The weekly
quota now permits multi-session planning, but P2 still needs implementation,
measurements and quality gates. No MFU, FLOPS, convergence,
generation quality or MBPP result is inferred from the speed measurements.

## 9. Assumptions and open decisions

Assumptions this recipe makes:

1. The pipeline branch builds and owns loading, masking, training, and checkpointing. This document is its contract.
2. fp16 training on T4, because T4 has no bf16; int4 QAT is deferred until after P2, and the pilot trains at 1024 context with longer context deferred.
3. The corpus is reused as built for P1. The audit passed with n-gram and substring checks, but semantic contamination is not ruled out. An additional semantic pass is proposed, not completed.
4. D1 uses the implemented blockwise frontier objective; classical masking remains a separate smoke profile. Quality is unmeasured; production P0 throughput is measured in section 8.
5. Section 8 uses three production P0 windows; any kernel, precision or batch change needs a new probe.
6. Planning uses the account-verified 30 Kaggle session hours per week for T4×2 and resumable sessions. Query current allowance before each launch; hardware availability is not guaranteed.
7. The decode policy arrives frozen from the sampler track. This recipe does not tune it.
8. HumanEval stays sealed until final evaluation.

Open decisions for Kriday:

1. Corpus mix. Keep 50/35/15 as built, or shift toward 30-35% code, 45-50% web, 15-20% math, as `research/hillclimb-quality-ernie.md` recommends. Shifting means rebuilding and reauditing the corpus, and it invalidates the current hashes.
2. Mask mode is now implemented as blockwise frontier masking for D1. A classical-mask comparison would be a separate, explicitly budgeted experiment.
3. Peak learning rate. 1e-3 proposed, range 5e-4 to 2e-3. Three 50-step screens project to about 0.18 to 0.19 T4-hours before overhead and can reject unstable settings; they do not establish a convergence-optimal LR.
4. Training volume. The reference D1 smoke proposal consumes about 100M corpus tokens; this is independent of the roughly 100M-parameter model size. Freeze or revise it against the current project plan before launch. The historical P2 500M-token arm-A reference needs a separately measured allocation under the renewable weekly quota.
5. Architecture and metric definitions are implemented and recorded in `D1_ARCHITECTURE.md`. The native T4 backend, parity, AMP resume and production P0 speed gate passed. The evaluation policy, total-time controller and operational readiness still block G3/P1.
6. Micro-batch. Use 8 and accumulation 8 from production P0; a fallback needs its own probe.
7. Number of seeds. Two is better. One is allowed with larger margins and a low-confidence label.
8. Kernel stack. The pinned modern FLA/Triton stack requires SM80+ and explicitly rejects T4/SM75. The native PyTorch `chunk` version 2 passes T4 checks and the 6,000 processed-token/s stop threshold. It does not reach the 13,000 green band. Any kernel-stack change requires new parity and throughput evidence.
9. Checkpoint retention and Drive space. A checkpoint is 1.4 to 2 GB, so retention needs a decision.
10. Whether to run the full 3B pretrain at all if P2 fails. The fallback is a single base pretrain and a published negative result.
11. Precision. The spec lists int4 QAT from the start; the quality plan says defer it until a baseline exists. This recipe freezes fp16 for P0-P2. Decide whether int4 QAT is a P3 gate or a post-P2 product run.
12. Context. The pilot uses 1024; the earlier target was 32k. Decide whether to extend context before P3, and budget the extra compute.

## 10. Source ledger

Repo facts: `artifacts/training-v1/final/manifest.json`, `audit.json`, `tokenizer_manifest.json`, `docs/TRAINING_DATA.md`, `spec/edgefast-diff-100m.md`, `research/l1-final.md`, `research/l1-fix.md`, `research/quality-fixplan-ernie.md`, `research/hillclimb-quality-ernie.md`, `results_t4.yaml`.

External sources cited: arXiv 2508.19529 (blockwise SFT, LLaDA-8B-Instruct LoRA), arXiv 2602.12413 (soft contamination; no single standard filter removes more than 6.5%), arXiv 2503.03862 and REGMIX (mix guidance), arXiv 2605.26106 (LoopMDM, 170M), arXiv 2610.18127 and 2604.07822 (looping counter-evidence), arXiv 2108.07732 (MBPP template), arXiv 2502.02737 (SmolLM2 calibration). Full ledgers live in `research/hillclimb-quality-ernie.md` and `research/l1-final.md`.

Numbers not from those sources are labeled DERIVED or ESTIMATE in place. The T4 measurements in section 8 are backed by `.verify/d1-t4/`; other estimates retain their labels.
