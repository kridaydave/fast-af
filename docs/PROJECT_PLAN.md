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
- **Compute:** the current ceiling is six GPU-hours. Run length depends on data,
  objective, measured throughput and evaluation overhead. A parameter count
  alone does not determine training time or establish that a model is trained
  well enough. The 1.84-to-1.89-hour projection applies only to the existing
  D1 recipe consuming 99,942,400 corpus tokens, not to training every proposed
  100M-parameter base, specialist, or the eventual 2B system.

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

## Infrastructure status and six-hour boundary

The [D1 implementation](D1_ARCHITECTURE.md) and
[T4 evidence](../.verify/d1-t4-speed/README.md) establish:

- Native PyTorch GDN version 2 runs on a real T4. Output, gradient, cached-state
  and AMP process-resume checks passed within their recorded tolerances.
- A 200-attempt P0 on the original training-v1 corpus cleared the 6,000
  processed-token/s stop gate in all three 50-update timing windows. All 150
  timed updates applied; five overflows were safely skipped during warmup.
- Cached full-frontier validation covered all 960 contexts and 32 frontiers in
  353.09 seconds, 365.42 seconds including checkpoint loading and process work.
- The measured configuration is context 1024, global batch 64, micro-batch 8,
  accumulation 8 and FP16 autocast with FP32 state/solve. Warmup calibrated the
  loss scale from 256 to 8; fresh scale-8 parity and resume checks passed.

These are infrastructure results for the current 109.1M-parameter D1. They do
not demonstrate coherent conversation, tool use, branch qualification, useful
MoE integration, inference speed, or completion of a general-base pretrain.
The Python-heavy corpus remains a frozen infrastructure fixture.

Additional bounded optimization work is recorded in the
[new T4 evidence](../.verify/d1-t4-parallel/README.md) and
[current allocation ledger](../.verify/d1-t4-parallel/six-hour-budget.json).
The historical ledger's 1.7894 GPU-hours is included there. Its unspent 4.15-hour
reservation is withdrawn: subsequent R&D must also fit within the same six-hour
ceiling. Use the new ledger's remaining balance rather than adding a new
six-hour allowance. Provider billing and quota units are separate from physical
GPU-hours; startup uncertainty is reserved explicitly.

The available two-GPU arrangement is **one Kaggle runtime with two T4s**.
Both allocated devices count throughout that runtime, including CPU reference
checks, one-rank controls, validation and checkpoint work. Six physical GPU-hours
therefore permit three runtime hours with two T4s before subtracting prior work.
A six-hour two-T4 runtime is twelve physical GPU-hours. No P1 run has been launched.

The optional `chunk-tc` v3 backend, fused AdamW and single-node DDP preserve the
109.1M-parameter model and context 1024. FP32 residuals/norms were retained after
an FP16 candidate failed cached/full parity on Kaggle; the tolerance was not
loosened. The final single-T4 bounded probe measured 8,436 processed input tokens/s
and 15,308 consumed corpus tokens/s. Those are different from its approximately
241 scored tokens/s. See [execution details](T4_PARALLEL.md) and the evidence for
paired two-T4 results, memory, resume checks and conservative budget arithmetic.

At the measured single-T4 rate, consuming one billion corpus tokens projects to
about 18.1 runtime hours before evaluation/checkpoint overhead. This is a
projection of this frontier objective, not a sufficient pretraining budget or
autoregressive loss-token equivalent. A model's parameter count alone cannot
establish a six-hour complete pretrain. Do not reduce model size or silently
substitute a 100M-token target to make the budget appear to fit.

These allocations do not cover three specialists, router training, upcycling,
code RL, or a roughly 2B-parameter system. External quality-evaluation and upload
costs remain unmeasured and must stop at their caps if exhausted. New backend,
batch and world-size configurations require P0 and quality requalification.

The [T4 pilot recipe](TRAINING_RECIPE_T4_PILOT.md) remains a conditional
infrastructure recipe. Its token-volume assumptions and allocations must be
revisited against the broad-base data plan before adopting it for this prototype.
Do not relabel one pass over training-v1 as completion of the general base.
The older multi-billion-token pretrain and matched looping twins are outside the
current six-hour plan.

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
   requalification. Implement cumulative GPU-time accounting and checkpoint retention/upload.
   Persist spent time across restarts; include validation, transfers and failed
   work. Refuse another stage if the remaining balance cannot cover it.
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
The six-hour current ceiling and the agreed general-purpose goal are constraints,
not open invitations to assume a Python-only model or unlimited compute.
