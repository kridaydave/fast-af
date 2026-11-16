# Dense two-T4 qualification

The saved Kaggle credential is outside the repository in the standard credential
store, with directory mode 0700 and file mode 0600. Authentication was verified
from a fresh process. No token value or credential file is part of the notebook,
this record, or the repository.

The private bounded job is
[d1-dense-two-t4-qualification-20261116](https://www.kaggle.com/code/kridaydave/d1-dense-two-t4-qualification-20261116).
Each attempt has a 1,200-second provider runtime cap and a 1,100-second internal
phase budget. These are qualification jobs, not long pretraining. Kaggle quota
is charged in GPU-enabled session wall time, not multiplied by two devices.

## Runtime and checks

The first two attempts ran on actual Tesla T4 ×2 / SM75, Torch 2.11.0+cu130 and
CUDA 13.0. Both native and Tensor Core dense parity passed against the small CPU
reference and full 109M-model context-1024 cache path. The real two-rank gradient
check passed for all model parameter tensors, including a rank with zero scored
targets. This does not establish full-model CPU gradient parity at context 1024.

## Resume checker defects

Attempt v1 failed when `communicate()` returned multiple JSON records in one
stdout chunk. Inspecting the interrupted trace also showed that killing only
the torchrun launcher did not reliably kill workers in separate sessions.
Attempt v2 added an explicit pause after update 4; it then exposed the surviving
workers by timing out while their stdout remained open. Neither failed attempt
qualifies as successful checkpoint resume. See [attempt-v1](attempt-v1) and
[attempt-v2](attempt-v2).

The corrected checker:

- Parses newline-delimited records after joining buffered chunks.
- Holds rank zero at the update-4 log boundary, preventing a later collective
  optimizer update before the controller intervenes.
- Captures the launcher's descendants and sends SIGKILL to the launcher and
  workers even when they have separate process groups.
- Requires exact observed update sequences 1–6, 1–4 and 4–6 for uninterrupted,
  interrupted and restarted processes, and refuses a final checkpoint from the
  interrupted process.

The final CPU suite passed **302 tests**, with two GPU-only skips. The three new
regressions cover buffered log tails, actual small-model CPU training held and
killed after update 4, and a real nested worker that starts a separate session.
See [the test log](full-suite-final.log) and
[source alignment](source-alignment-final.json). Six measured runtime/numerical
scripts remain byte-identical to the single-T4 implementation. Only the
qualification checker changed; no model, objective, optimizer or trainer
arithmetic changed.

## Earlier frontier request

The earlier non-dense final qualification request eventually ran and failed at
single-GPU micro-batch 8: Torch allocated 13.53 GiB and a further 302 MiB request
could not fit. Its parity, cache and two-rank gradient phases passed, but it never
reached two-GPU resume. This is frontier evidence, not dense timing or resume.
[prior-frontier](prior-frontier) preserves those reports. The dense job uses
single-GPU micro 2 and two-GPU global micro 4, two sequences per GPU, with global
batch 64 for timing and a separate global-batch-2 resume check.

The controller runs required correctness and resume before timing. It then
compares one-GPU and two-GPU dense timing on the same Kaggle runtime. A matched
global-micro-2 two-GPU control runs if the internal budget permits. Each timing
configuration has one untimed warmup and three timed successful updates.
Reports record actual consumed, processed and scored tokens, peak per-rank
memory and skipped AMP updates. These short screens do not qualify production
P0, trained-model quality, broad-base data or sufficient pretraining volume.

Version 3 contains the corrected checker. Its final allocation/execution status
is recorded separately; no dense resume or throughput success is claimed until
those actual reports pass.

Latest version-3 snapshot: **QUEUED** at 2026-11-16T17:40:47.138460+00:00; provider
reserved hours 0.0. Resume and throughput remain unqualified.
