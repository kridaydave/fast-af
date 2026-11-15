# Project plan: specialist prototype and eventual MoE

Current direction, 2026-11-15. This document supersedes conflicting product,
training-data and sequencing decisions in the historical D1 and swarm specs.
It is a plan, not evidence that the proposed architecture or training will work.
The D1 pipeline now has measured T4 infrastructure results. Those establish
feasibility of that implementation, not a general-purpose base or a working MoE.
This plan does not authorize a training run.

## Goal

Build a general-purpose assistant that communicates clearly and codes well.
The intended final direction is a roughly 2B **total-parameter** mixture of
experts (MoE), assembled where practical from smaller pretrained models. Active
parameters, compute, memory and latency are separate measures. A 2B MoE is not
assumed to match a dense 2B model. Architecture, number of experts and sizes are
open; do not treat this target as an approved training budget.

## Model size, data volume and compute

Keep these quantities separate:

- **Model size:** the small test model is in the roughly 100M-parameter class.
  The existing D1 implementation contains **109,137,632 parameters**. Its
  measured backend is infrastructure we can evaluate, not an automatic choice
  of base architecture for the agreed general-purpose prototype.
- **Training data:** the existing bundle contains 99,983,360 training tokens.
  The old recipe's approximately 100M-token pilot is a separate smoke-run
  proposal based on that bundle, not a token target implied by the model name
  or a user-approved general-base training volume.
- **Compute:** use the renewable **30 Kaggle GPU-session hours per week**
  available to this account. One hour of a T4×2 runtime uses about one hour of
  that allowance. Query the provider's actual remaining quota before a run.
  The earlier six-hour cumulative project cap and 2.52-hour balance were an
  incorrect planning interpretation and are withdrawn. Model size alone does
  not determine training time; the token budget and objective remain separate.

Do not use bare `100M` for a data or training budget. State parameters, unique
corpus tokens, consumed tokens, processed tokens and scored tokens explicitly.
The final 2B goal means total parameters; active parameters remain separate.

## First experiment

1. Train and assess a small general base. Around 100M parameters is a proposed
   starting point, not a requirement. Establish coherent answers and short
   multi-turn instruction retention before committing to branches. If these
   fail, investigate data, training and SFT, then consider a larger seed.
2. Create compatible siblings from that checkpoint, retaining general data in
   each: **communication**, **code**, and **reasoning**. Use the same tokenizer,
   architecture and shapes for the first upcycling test. The code branch covers
   multiple programming languages; communication tests whether clean SFT can
   teach direct answers at small size. Their data, sizes and budgets remain open.
3. Each branch has one primary qualification goal, evaluated against a chosen
   larger model on fresh held-out tasks under a frozen comparable protocol:
   - Communication: follow instructions across short conversations.
   - Code: solve and debug small tasks, checked with hidden tests.
   - Reasoning: use supplied facts and constraints correctly.
   Each must beat the larger reference in its arena before its outputs are
   candidates for distillation. Choose the reference, sample size, scoring,
   uncertainty and pass margin before training; inspect outputs manually and
   filter errors. A narrow win alone does not guarantee useful teacher data.
4. Test tool calling separately: select the right tool or no tool, supply correct
   arguments and use returned results over turns. Do not assume a 100M model or
   an MoE can do this reliably.
5. Build a small upcycled MoE and compare it with the base, siblings and a
   whole-model router. Measure behavior, training cost, active and total
   parameters, memory and latency. Check whether the mixture retains general
   ability and improves on its components. The architecture, integration method
   and required joint training remain research questions.
6. If integration works, consider more experts or capacity for code and
   reasoning, and scale toward the final target. Same-shaped siblings are the
   initial test, not a requirement for equal capacity in the final system.
   Code-focused RL is a later option after a solid general baseline, with
   general-behavior regression checks.

Upcycling retains selected specialist weights in the final model. Distilling all
branches back into the small base is a separate experiment and does not make it
larger; its capacity may constrain what it retains. A whole specialist checkpoint
is not necessarily one MoE expert: shared components and retained weights must be
counted explicitly. See [BAM](https://proceedings.neurips.cc/paper_files/paper/2024/file/665bb142d4b9f55660cb89bb56a66fe1-Paper-Conference.pdf)
and [Symphony-MoE](https://arxiv.org/abs/2509.18542); neither establishes success
for this project's architecture or compute budget.

## Data and evaluation

Design a broad general-language foundation with diverse code languages, math and
technical material; source selection, proportions, tokenizer and token budget
are open. The existing training-v1 50% Python-code / 35% web / 15% math bundle
is a frozen pipeline and smoke-test artifact, not the final model's data recipe.
Its tokenizer is suitable for that artifact; compatibility across a new base and
its siblings must be designed and frozen before training them. Preserve source
revisions, licenses, attribution, hashes and splits, and decontaminate all stages.
No new dataset download or modification of the existing corpus follows from this
plan.

Behavioral evaluation is primary: privately held-out, manually inspectable
cases for instructions, supplied-fact reasoning, uncertainty, conversation,
code and debugging, with prompt variations and full outputs recorded. Hold out
repositories, related variants and code tests; assess tool selection and results
separately. Public benchmarks including MMLU and HumanEval remain comparison
points, not sole targets; contamination and repeated tuning limit their value.
Use matched protocols, report uncertainty, inspect failures and protect a final
holdout from development and distillation. Record regression results for general
behavior whenever specializing, upcycling or applying RL.

## Infrastructure status and weekly Kaggle budget

The [native T4 evidence](../.verify/d1-t4-speed/README.md) records CPU/T4
output, gradient, cache and AMP resume parity, a 200-attempt P0 whose three
50-update windows cleared 6,000 processed tokens/s, and cached validation over
all 960 contexts and 32 frontiers in 353.09 seconds. All 150 timed updates
applied; five overflows were skipped during warmup. Those timings belong to
native chunk version 2 and are not a v3 or two-GPU measurement.

The optional `chunk-tc` v3 backend, fused AdamW and single-node DDP preserve the
109,137,632-parameter model, context 1024 and global batch 64. FP32 residuals
and normalization were retained after an FP16 candidate failed cached/full
parity on Kaggle; the tolerance was not loosened. The
[final single-T4 probe](../.verify/d1-t4-parallel/colab-final/uniform.json)
measured **8,436 processed input tokens/s**, **15,308 consumed corpus tokens/s**
and approximately **241 scored tokens/s**. Single-T4 output, gradient,
1024-token cache and process-resume checks passed. These are infrastructure
results, not evidence of coherent conversation, tool use or a trained base.

As of 2026-11-15T17:12:00.712435+00:00, the final two-T4 qualification
request is still **QUEUED**, with no provider GPU reservation and no published
outputs. Earlier two-T4 gradient checks used another candidate. Final v3
**two-T4 throughput and resume parity remain unmeasured**. See the
[measurement record](../.verify/d1-t4-parallel/README.md) and
[execution guide](T4_PARALLEL.md). No long pretraining or P1 run has been launched.

### Weekly quota policy

Use one Kaggle runtime with two T4s and the
[current weekly ledger](../.verify/d1-t4-parallel/weekly-budget.json):

- **30 runtime hours per week**, refreshed weekly. Both T4s can be used during
  those hours; do not divide the allowance by two. A six-hour T4×2 run consumes
  about six hours of Kaggle quota. Physical device-hours may be logged for
  compute analysis, but they are not the quota planning unit.
- Charge this week's experiments, GPU-enabled idle/CPU work, validation,
  checkpoints, uploads and failed sessions to this week's allowance. Previous
  weeks' usage does not reduce the fresh allowance. Colab usage is separate.
- The API snapshot reports 0.05258 hours used and
  0.00 provider-reserved hours, leaving **29.95 hours**.
  Reserving the already queued test's 20-minute cap once leaves **29.61
  hours** for other work in this snapshot. Replace a local reservation with the
  provider reservation when allocated; never subtract both for the same job.
- The API reports the next reset at **2026-11-21T00:00:00 UTC**. Read the
  actual quota again on every runtime start and after a reset. A queued job
  that starts in another quota window belongs to that window.
- The old [six-hour ledger](../.verify/d1-t4-parallel/six-hour-budget.json),
  its 2.52-hour balance, and the inherited 4.15-hour future-stage reservation
  are historical, superseded planning assumptions. They are not additional
  deductions or a lifetime project ceiling. Six hours may be chosen as an
  experiment duration later; it is not an active cumulative quota cap.

Kaggle documents the [weekly allowance](https://www.kaggle.com/docs/efficient-gpu-usage)
and [12-hour CPU/GPU session limit](https://www.kaggle.com/docs/notebooks).
Plan resumable sessions below that limit and export verified checkpoints before
termination. Preserve model/optimizer/scheduler/scaler state, both ranks' RNG,
corruption RNG, loader cursor and token counters across sessions and weeks.
Pin source, backend, environment, data and tokenizer: a runtime reset must not
silently change the resume configuration. Checkpoint retention/upload and final
two-T4 resume qualification remain prerequisites for a long run.

### Provisional pretraining duration on two T4s

The token target for the general base is **not selected**. The following are
illustrative consumption budgets for the fixed 109.1M-parameter D1, not a
100M-token substitute for a 100M-parameter model, or a quality guarantee.

The measured single-T4 consumption rate is 15307.63 corpus tokens/s. For an
assumed aggregate two-T4 speedup `s`, estimate:

`runtime hours = consumed corpus tokens / 15307.63 / s / 3600 × 1.30`

The extra 30% is an **unmeasured planning allowance** for evaluation,
checkpointing, restarts and setup. It is already included in the table and must
fit inside the weekly allowance, not be added above it. Queue delays and waits
for a quota reset add calendar time and are not included here.

| Consumed corpus tokens | Provisional 1.5× scaling | Ideal 2× scaling | No scaling gain |
| --- | ---: | ---: | ---: |
| 1B | 15.7 h | 11.8 h | 23.6 h |
| 2B | 31.5 h | 23.6 h | 47.2 h |
| 3B | 47.2 h | 35.4 h | 70.8 h |

All two-GPU columns are **estimates, not benchmarks or guaranteed bounds**.
The 1.5× case is a working assumption relative to the measured Colab single-T4
rate; software differences and communication may make it optimistic. Replace it
with the matched-work Kaggle measurement before choosing a production budget.
The exact inputs and calculations are in
[pretraining-time-scenarios.json](../.verify/d1-t4-parallel/pretraining-time-scenarios.json).

For example, 2B consumed tokens projects to roughly 31.5 runtime hours in the
working case, just over a fresh weekly allowance; ideal scaling would put it
near 23.6 hours. A 3B consumption budget projects to 47.2 hours in the working
case, requiring a quota refresh. These are execution hours, not guaranteed
calendar completion dates or proof of sufficient pretraining. Consumed frontier-
objective corpus tokens are not autoregressive loss-token equivalents; repeated
passes do not create additional unique data.

The Python-heavy training-v1 corpus and inherited approximately 100M-token pilot
remain conditional infrastructure artifacts. Choose broad-base sources,
tokenizer, unique-data volume, consumption target and behavioral gates before
training a base. New backend, batch and world-size configurations require their
own P0 and quality requalification. Specialists, upcycling and the eventual
roughly 2B-total-parameter system need separately measured budgets; they are not
automatically covered by one base-pretrain estimate.

## Next work, in order

1. Specify the base feasibility suite before training: coherent answers, short
   multi-turn instruction retention and honest uncertainty. Keep tool calling
   as a separate test and freeze held-out cases and failure criteria.
2. Choose broad-base sources, license/provenance checks, mixture and tokenizer.
   Include general language, multiple code languages, reasoning and technical
   material. Freeze clean splits and contamination controls before packing.
3. Freeze model size/architecture and a **separate token or attempt budget** from
   measured compute. The current D1 timing is a reference; changed data, context,
   architecture, batch or objective may require a new bounded probe.
4. Select the measured T4 configuration and freeze its micro-batch/frontier
   grouping. A global micro-batch of 8 on two GPUs preserves the single-GPU
   grouping; 16 changes the number of frontier draws per update and needs
   requalification. Implement weekly Kaggle runtime accounting and checkpoint
   retention/upload. Persist run progress across restarts, but debit quota to
   the provider's current weekly window. Include validation, transfers and
   failed work; refresh the allowance after the provider reset. Refuse a stage
   if this week's remaining quota cannot cover its allocated session chunk.
   `--p1` remains blocked until these controls and the evaluation protocol are
   ready. Do not launch training solely because the speed gate passed.
5. Train one bounded base candidate and inspect the feasibility results. Only
   if it passes, allocate a separately reviewed budget to the compatible
   communication, code and reasoning branches and their larger-model tests.
6. Attempt a small upcycled MoE only after the branches qualify and its
   integration budget is explicit. Scaling toward 2B total parameters and code
   RL require evidence and resources beyond the current infrastructure pilot.

## Open decisions

Base architecture and size; whether the current D1 is an appropriate seed;
broad-base sources, licenses, mixture, tokenizer and token volume; feasibility
thresholds and larger reference model; specialist recipes and budgets; MoE layout,
router and integration training; generation settings and final evaluation splits.
The renewable 30-hour Kaggle weekly allowance and the agreed general-purpose
goal are constraints, not permission to assume a Python-only model, sufficient
pretraining data, or unrestricted runtime availability.
