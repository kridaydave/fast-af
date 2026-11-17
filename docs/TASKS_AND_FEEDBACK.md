# Tasks and feedback

This first version builds short Python tasks and grades recorded completions.
It does not generate model completions or train a model. Eight authored families
cover lists, strings, counting and a small repair. This is a seed curriculum,
not enough data for the proposed 10% training mixture or evidence of transfer.

Linux, `/usr/bin/python3` and `bwrap` are required. Bubblewrap must be allowed to
create user, PID, mount and network namespaces. The CLI probes real execution
before building or grading and refuses an unsandboxed fallback. Unsupported
hosts must fix their sandbox configuration; they cannot substitute the existing
HumanEval executor. The task catalog uses only the Python standard library.
Production context checking additionally uses the project's `tokenizers` package.

## Build and inspect

For a quick CPU plumbing check:

```bash
python3 -m tasks build --output artifacts/tasks-smoke --variants 2 --seed 17 --smoke
```

For the pinned benchmark and tokenizer checks, using existing local artifacts:

```bash
python3 -m tasks build --output artifacts/tasks-v1 --variants 4 --seed 17 \
  --benchmarks artifacts/training-v1/sources.json \
  --tokenizer artifacts/training-v1/final/tokenizer.json \
  --tokenizer-manifest artifacts/training-v1/final/tokenizer_manifest.json
```

Outputs are immutable directories. Choose a new path for another build.
`--variants` accepts 1 through 16. Variants change task parameters within the same
family; they are not new independent skills. The split registry never hashes
individual variants into different splits.

| Output | Use |
|---|---|
| `prompts/train.jsonl` | Training-task prompts and starter code for generation |
| `prompts/development.jsonl` | Development prompts for measurement only |
| `references-train.jsonl` | Training-family reference examples, with provenance and solution character offsets |
| `private/tasks.jsonl` | References, cases, expected answers and sealed evaluation tasks |
| `private/audit.jsonl` | Executed reference passes and rejected-mutant results |
| `manifest.json` | Family assignments, hashes, exclusions, context checks and review status |

Private files have restricted permissions and are never mounted into candidate
workers. Do not give them to a model generator. Hidden expected answers remain
in the parent grader; workers receive code and argument values only. Test inputs
are observable during execution, so passing a finite test set is not a proof of
general correctness. All current hidden cases are authored in the catalog;
independently sourced transfer tasks still need to be added.

Every reference must pass and every known incorrect implementation must return
a wrong answer. A crash is not accepted as proof of a discriminating verifier.
Repair cases include broken behavior and behavior the repair must preserve.
Production builds exclude matches to complete pinned HumanEval and MBPP suites
using the existing 13-word n-gram and normalized-substring checks. These checks
do not rule out semantic overlap. The saved tokenizer checks complete prompt and
solution length without truncation; overlong tasks are counted and excluded.

References are marked `pending-human-review` by default. After reviewing
`tasks/catalog.py`, a human reviewer can supply an attestation JSON containing
`catalog_sha256`, `reviewer`, and `approved: true` with `--review-file PATH` when
building a new bundle. The hash must match the current catalog. The CLI never
creates that approval automatically. File hashes detect changes; they are not
cryptographic signatures or proof of a reviewer's identity.

## Grade a frozen checkpoint's attempts

Save full Python source defining `solve(values)` in JSONL. For example:

```json
{"attempt_id":"run1-0001","task_id":"<ID from prompts/train.jsonl>","completion":"def solve(values):\n    return []\n"}
```

This example is a deliberately unverified candidate, not a passing solution.
Use a unique attempt ID for every sample. Each batch belongs to one frozen
checkpoint and one split.

```bash
python3 -m tasks grade --bundle artifacts/tasks-v1 \
  --attempts artifacts/tasks-rollout-v1/attempts.jsonl --output artifacts/tasks-feedback-v1 \
  --checkpoint '<model revision or checkpoint hash>' \
  --generation-seconds 120 --gpu-seconds 120 \
  --min-accepted 10 --min-successes-per-gpu-hour 30
```

The numbers above illustrate the interface. Set timing from real measurements
and freeze yield thresholds before generation; these are not recommended
thresholds. GPU seconds are summed over devices for the entire generation batch,
including failed attempts. They exclude CPU-only grading. The report labels
generation timings as externally reported and grading wall time as measured.
Use zero GPU seconds for CPU fixtures; the rate is then unknown and the yield
gate remains false. The CLI does not authenticate the supplied checkpoint ID.

The grader writes `feedback.jsonl`, `accepted.jsonl` and `report.json`. Only
verified training-family successes enter `accepted.jsonl`. Development successes
are measured with `--split development` and never exported for training. The CLI
does not expose sealed evaluation grading for this feedback loop.

Candidates run without network, home, project or bundle access. Only the system
Python runtime and worker are mounted read-only, with a private temporary
filesystem. Each worker has a 1-second CPU limit, 3-second wall limit, 256MiB
address-space limit, 64KiB escaped-input/output caps, no child processes and no regular-file
writes. Host-side grading checks returned values with strict types; candidate
output cannot declare itself passed. Syntax errors, resource failures and
malformed output fail the attempt without counting as success. Code that exceeds
the parent's structural-normalization limits is also rejected per attempt.

Deduplication uses task ID plus normalized solution structure, including renamed
local variables and numeric literals. It is conservative and may merge distinct
passing approaches. Supply `--previous-accepted PATH` with the accumulated accepted
records from this same bundle to exclude earlier rounds too. Novelty means a new
accepted task/structure pair, not a newly learned capability. This version does
not implement a curriculum scheduler; `by_family` solve rates and failure statuses
are the measurements for choosing subsequent tasks.

`eligible_for_training` requires a non-smoke bundle, recorded reference approval,
a training split and the configured yield gate. Smoke records always retain
`smoke_only: true`. Eligibility applies to this candidate data only; it does not
approve model training, close the hardware gate or establish learning gains.
The exports are not yet wired into the three-bucket token-mixture builder.

Run the CPU checks with `python3 -m pytest -q tests/test_tasks.py`. Sandbox-dependent
tests explicitly skip if genuine bubblewrap execution is unavailable.
