# Training integration review

Reviewed on 2026-11-13 by a single worker in
`t3/shard-mask-resume-t4-pilot-integration`. No subagents or other threads were
started. Only this bound worktree was changed. Nothing was pushed or merged into
the project-root master checkout.

## Commits and preserved work

- Fetched master base: `cde26573920c7ae5bd6aab06617952ba282bbde9`.
- Pipeline input: `dfae0b68f7f3c2d560548a991fd5151d1138ec00`.
- Recipe input: `e61cf251f83bb6d34600f17dc31281df303e7111`.
- Pipeline merge onto master: `aad0d571eda983206fa88675adcb48129a5f59b1`.
- Recipe merge: `3c7e347bc7415bd00f10f70a4f226128fc900ff0`.
- Pipeline corrections and regression tests:
  `ff333528487fdce0947599daa87b067bbed66947`.

Both input commits merged cleanly. The interrupted checkout at
`/home/kriday/.t3/worktrees/fast-af-boi/t3-8a46ff0f-9fc27c4d565e9888`
was at `1bf87efe2b72e21328bd254c56ae112666355ef0`. It already contained both
merges, plus an uncommitted 19-line patch in `scripts/train_diffusion.py` and
`tests/test_train_diffusion.py`. That patch added positive whole-sequence shard
checks and FP32 non-finite gradient detection. It was read, copied into this
worktree, and extended here. The source checkout was not changed. Its original
binary diff SHA-256 was
`336d80708abe08d04c5b467b18b2c81261d5cd0c0811e808545ee96efd567520`.

The project root stayed on master at the base above, retaining its modified
`results_smoke.yaml` and untracked `g1-baseline.json`, `g1-candidate.json`, and
`g1-comparison.json`. Its tracked diff SHA-256 was
`33a04ebf577ee77be89083121818d83c9c1c3331c1f267879ca931072526b5f7`.

## Correctness findings and changes

1. Non-finite loss was checked after the optimizer update, and FP32 gradient
   norms could silently poison weights. Loss is now checked before backward;
   non-finite FP32 gradients fail before the update. Tests verify unchanged
   weights and absence of a checkpoint on failure.
2. AMP overflow skipped the optimizer while advancing the LR schedule. The
   schedule now advances only after an update. Non-finite gradient norms also
   skip the update and back off the scaler, covering norm-reduction overflow
   even when each gradient is finite. Logs distinguish attempted batches,
   successful optimizer updates, skipped updates, and consumed tokens. The
   real CPU GradScaler test verifies skipping and recovery, with no CUDA claim.
3. Resume accepted plausible but incorrect token totals and in-range cursors.
   It now checks exact consumed micro-batches and tokens from the seeded shard
   order, including partial batches and completed epochs. Non-batch-boundary
   offsets are rejected. CPU tests resume exactly across partial batches,
   accumulation, evaluation, and multiple epochs.
4. Manifest loading did not require whole sequences. Added checks cover positive
   integer sequence lengths, positive divisible shard token counts, shard
   sequence counts and split totals, and empty shards/splits. The hard-coded
   special IDs are checked against both manifests. Vocabulary bounds and a
   missing declared tokenizer model are rejected too.
5. The cosine schedule reached its stated floor after the final update, and
   each record logged the next LR. The final update now uses 10% of peak and
   logs its actual LR. A one-step run uses peak LR. Non-finite LR, clipping,
   decay, and epsilon values are rejected.
6. A combined-batch reference check verifies token-weighted accumulation when
   micro-batches contain three and one sequences. This checks the weighting,
   beyond merely counting consumed tokens.

Checkpoint format is now 3. Earlier formats are rejected because schedule and
update semantics changed; existing format-2 files are not exact resumptions of
this implementation. Library versions are recorded but not compatibility-
checked, so exact reproducibility still assumes the same runtime.

## Recipe and spec review

The integrated code is a small bidirectional smoke transformer. The D1 and P1
refusal guards remain in place. The recipe now separates implemented behavior
from unresolved requirements:

- The spec's 640-dimensional query projection with head dimension 128 gives
  five query heads. Two KV heads cannot serve those heads in standard GQA.
  The spec and recipe now leave head layout open. A separate query projection
  width is also possible, but no architecture choice was silently made.
  [PyTorch's GQA contract](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)
  requires query heads to be divisible by KV heads.
- Causal GDN recurrence does not implement all-position bidirectional masking.
  Directionality and the GDN mask/cache contract must be designed and tested.
- Equal-token looped and unlooped runs do not imply matched compute. P2 now
  budgets arm B tokens from measured FLOPs per token, with the table expressed
  in arm-A-equivalent compute. The table covers one paired seed; a second seed
  requires another training and evaluation allotment. Paired seeds are explicit.
- Teacher-forced masked CE has no decode-step T argument. The inherited
  "validation NLL at 8 steps" criterion needs a defined estimator or an amended
  evaluation protocol before G3. It cannot currently serve as an executable gate.
- Native AMP checkpoints fp32 parameters and optimizer moments, with temporary
  fp16 compute casts. A separately stored fp16 parameter copy is not required.
  Failed resume checks now reject continuation.
- The real shard tail has 5,480 sequences. Micro-batches 4 and 8 divide it;
  micro-batch 16 produces partial global batches. Step-token arithmetic is
  nominal; throughput and budgets must use actual token counts.
- The forced first mask when Bernoulli draws select none is documented as a
  short-row bias. Only corruption rates 0.10 to 0.90 are sampled, rather than
  all noise levels.
- Existing n-gram and normalized-substring audit checks meet the documented
  G3 substring option. An additional semantic pass remains proposed and
  unperformed. A clean mechanical audit cannot establish no semantic leakage.
- LR screens establish initial stability, not optimal long-run convergence.
  Provider quotas, sm75 kernel support, D1 throughput, and the extra attention
  FLOPs term remain assumptions to verify. The legacy uncached inference timing
  is not evidence of D1 training speed. QAT is consistently deferred past P2.

The inherited research-paper ledger was not fully re-audited against every
paper. This review checks repository alignment, arithmetic, local corpus facts,
and the relevant primary PyTorch contracts. The budget remains an estimate.

## Checks and limits

- `/usr/bin/python -m pytest -q`: **187 passed in 149.72 seconds**. No failures
  or skips. The pre-fix merged suite also passed, 163 tests.
- `/usr/bin/python -m pytest tests/test_train_diffusion.py -q`: **43 passed in
  11.68 seconds** on the final training code.
- `/usr/bin/python -m compileall -q scripts tests bench` and `git diff --check`:
  passed. No repository type-checker configuration was found.
- A fresh fetch still put `origin/master` at the base recorded above, and it
  is an ancestor of the integrated branch.
- Both protected checkout heads, status lists, and tracked-diff hashes were
  checked again and matched the preservation record. The imported patch hash
  matches the untouched interrupted checkout's diff.

The environment was Python 3.14.8, PyTorch 2.15.0+cpu, NumPy 2.5.4, with
`torch.cuda.is_available() == False`.

The real-bundle process check ran the committed training code at `ff333528`.
Each six-step run used batch size 2, accumulation 2, sequence length 1024,
warmup 1, checkpoint/evaluation interval 3, and one validation batch per eval.
All 11 real shards and tokenizer provenance were hash-verified on load. The
model was the default smoke transformer, not D1. A process was SIGKILLed after
logging step 4, exited -9, and restarted from checkpoint 3. Its loss records,
model, optimizer, scheduler, scaler, loader and torch RNG matched the
uninterrupted run exactly at step 6 and **24,576 consumed tokens**. The two
sampled validation losses were 45.2665 and 42.9459; these are a smoke check,
not convergence evidence.

Durable local evidence lives under `.verify/training-integration/`: `pytest.log`,
`cpu-process-report.json`, the full/interrupted/resumed process logs, and
`preserved-checkouts.json`. Raw checkpoints remain in the temporary artifact
folder named in the CPU report.

The source branch attributed its earlier 109-pass/1-failure run to the
`sys.executable -S` subprocess using an AppImage executable. That original
process is unavailable. Here, the bare `pytest` launcher has a shebang pointing
to the missing `/home/kriday/Applications/Synara-1.0.7-x86_64.AppImage` and exits
127. The same CLI regression passes under `/usr/bin/python -m pytest`, including
its real `-S` child interpreter and expected success/failure exit codes. No test
was weakened, and no global launcher was modified.

No GPU, T4, D1, P0 or P1 validation is claimed. The branch's historical T4
training numbers have no raw logs or checkpoints in the source commit and
were not accepted as evidence. There was no full-corpus training epoch or
real full-validation sweep, no fresh contamination audit, and no gPPL/MBPP
quality evaluation. Runtime support for a 30-minute checkpoint cap, retention
and remote upload, wall-time/GPU-hour/FLOPs accounting, decode-policy hashes,
D1 initialization and GDN architecture, and production probe measurement is
still absent. Smoke defaults also differ from the proposed P1 LR, warmup,
seed and evaluation configuration; the P1 guard prevents this model being
misrepresented as a production run.
