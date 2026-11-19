# D1 implementation evidence

Implementation commit: `a640397c126a5dc5d70460e52ae24e869bde6c94`.
Hardware compatibility guard: `3f080ba886fd72aee760900a6333d3ce422452eb`.

- `full-model-cpu.json`: actual 109,137,632-parameter default model, synthetic
  three-block input, CPU FP32 forward/cache equivalence, immutable prefix and
  future-block isolation. No backward, GPU or throughput result in this check.
- `mini-cpu-backward.json`: 14,916-parameter, reduced-dimension CPU configuration;
  real forward, backward through every parameter and an AdamW update. This is
  not a full-width training result.
- `real-corpus-cpu-resume.json` and the full/interrupted/resumed logs: a
  522,740-parameter reduced-width D1 on real training-v1 contexts of length
  1024. Six CPU updates, batch 1, accumulation 1, no validation in this process
  check. SIGKILL after logging step 4, exit -9, restart at checkpoint 3.
  Loss records, model, optimizer, scheduler, scaler, cursor and all saved RNG
  states matched exactly. Totals: 6,144 consumed, 4,032 processed and 92 scored
  tokens. A development decode preserved 35 prompt tokens and emitted 13
  tokens with 1 prefill, 8 denoising and 1 clean-commit pass. No quality claim.
- `cuda-unavailable.json`: SKIP, exit 2, no CUDA checks executed.
- `focused-tests.log`: 64 CPU tests passed and 2 GPU tests skipped before the
  final hardware-policy guard. That additional guard is covered by the final
  suite, using a simulated rejection path, not an executed CUDA kernel. The
  skipped GPU parity checks additionally enforce relative output/state and
  gradient errors, rather than relying only on permissive absolute tolerances.
- `implementation-suite.log`: 208 passed and 2 GPU tests skipped on the initial
  implementation. The final suite after the compatibility guard is recorded
  separately in `final-suite.log`.
- `typecheck.json`: scoped basic type checks for `d1_model.py` and
  `validate_d1.py`, 0 errors and 0 warnings. Ruff F checks, compilation and
  `git diff --check` also passed.
- `preserved-checkouts.json`: protected checkout heads, dirty status and tracked
  diff hashes were checked again after implementation and were unchanged.

Two initially concurrent verification runs terminated with exit 143 before
completion. Their cause was not established, and they were not counted as
passes. The whole suite and real-corpus process check were rerun separately.
The final single pytest log completed with 209 passed and 2 skipped, but its
runner returned 143 after that summary. To get unambiguous successful exit
statuses, all tests were also run in separate groups, recorded below.

No GPU/FP16/T4 performance or full-width training claim is made. The pinned FLA
stack uses modern Triton, which documents SM80+ support. T4 is SM75, and the
compatibility guard rejects it before kernel import/model allocation in the
CLI. A T4-compatible accelerated backend or an explicitly revised hardware
plan is still required before P0/P1.

Confirmed exit-0 groups on the final CPU implementation:

- Core suite, excluding only the two separately exercised SmolDLM modules:
  **193 passed, 2 skipped**.
- `test_smoldlm_cache.py`: **12 passed**.
- `test_smoldlm_wiring.py`: **4 passed**, using the real cached model assets.

Together these cover **209 passed, 2 GPU skips**, matching the full-run summary.
`checks.json` records counts, scopes and the runner-interruption caveat.
