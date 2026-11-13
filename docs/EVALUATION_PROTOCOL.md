# HumanEval evaluation protocol

This document freezes the prompt and extraction protocol used by
`bench/humaneval.py` for HumanEval pass@1. It exists because a score is only
meaningful with the prompt that produced it: a base continuation prompt and an
instruction-tuned chat prompt are different experiments, and mixing them makes
the resulting pass@1 numbers unauditable.

Protocol version: `humaneval-eval-v1`
Extraction policy version: `humaneval-extract-v1`

## Modes

Every model that has `humaneval: true` must declare `eval_prompt` explicitly in
`bench/config.yaml`. The harness never infers the mode from the repo name.

| mode | prompt sent to the model | expected output | program used for exec |
|---|---|---|---|
| `completion` | the raw HumanEval prompt, unchanged | a continuation of that prompt | prompt + continuation |
| `chat` | `tokenizer.apply_chat_template` with a fixed instruction | one complete program in one Python fence | the extracted program alone |

`completion` is the historical protocol and is preserved byte-for-byte for base
models. `chat` is for instruction-tuned models whose raw continuation is prose.
The current SmolLM2-135M-Instruct row uses `chat`; the Qwen/LFM rows use
`completion`.

## Chat prompt

The chat prompt is built from exactly these two messages. The instruction is
prewritten and must not be tuned against HumanEval scores.

```python
CHAT_SYSTEM = "You are an expert Python programmer."
CHAT_INSTRUCTION = (
    "Write one complete, self-contained Python program that solves the problem. "
    "Put the entire program inside a single ```python fenced code block. "
    "Include every import the program needs and the required function signature. "
    "Do not write any explanation, prose, or text outside the code block."
)
```

Messages:

```python
[
    {"role": "system", "content": CHAT_SYSTEM},
    {"role": "user", "content": prompt + "\n\n" + CHAT_INSTRUCTION},
]
```

The rendered prompt is `tokenizer.apply_chat_template(messages,
tokenize=False, add_generation_prompt=True)`. A tokenizer without a chat
template is a hard error for `eval_prompt: chat`; the harness does not silently
fall back to a completion prompt.

The fixed template hash is recorded at runtime. Current values:

- `prompt_template_id: humaneval-chat-instruction-v1`
- `prompt_template_sha256: ab691e5f92efdf769a7518de8a5193426b95825bad4e25625db25160a295fd3f`
- completion mode uses `prompt_template_id: humaneval-raw-completion-v1` and the
  SHA-256 of the empty string
  (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`),
  because there is no wrapper text.

## Extraction policy v1

The raw model completion is always preserved in the results row
(`cases[*].completion`). Extraction is a separate step and never rewrites the
raw text.

### Completion mode

1. If the exact stripped prompt appears in the completion, remove everything up
   to and including its first occurrence.
2. Otherwise return the completion unchanged.

No other surgery is performed: no `def` is prepended, no cut is made at the next
`def`, imports are not stripped, and signatures are not rewritten. The prompt is
then prepended once, exactly as the historical executor did.

### Chat mode

1. If the response contains one or more Markdown fences:
   - choose exactly one Python-tagged fence (`python`, `py`, `python3`);
   - if there is one Python-tagged fence, use its body verbatim, even if other
     non-Python fences are present;
   - if there are multiple Python-tagged fences, reject the case as ambiguous;
   - if there are no Python-tagged fences, accept exactly one untagged fence
     only when its body parses as Python; otherwise reject;
   - ignore all text outside the chosen fence, including trailing prose.
2. If there are no fences, the whole response must parse as a Python module with
   at least one top-level function. If it does not, and it starts with the exact
   echoed prompt, remove that leading echo and retry the remainder. Prose-only
   or unparseable output is rejected.
3. Inside a fence the prompt is treated as part of the model's program, not as
   a wrapper: an echoed prompt is kept so the function signature survives.
4. A chat program is standalone. The HumanEval prompt is not prepended a second
   time, so a full function is never double-appended.
5. `ast.parse` must succeed and the module must contain a top-level
   `def`/`async def`. This rejects prose, truncated bodies, and empty output
   instead of letting the executor fail obscurely.

Multiple Python fences are rejected rather than resolved by "longest fence" or
similar task-specific heuristics. Plain code with trailing prose is rejected
unless the prose is outside a fence; this is a deliberate ambiguity boundary,
not an attempt to guess where code ends.

## Results record

Every HumanEval result row contains:

```yaml
humaneval:
  n: 50
  passed: 3
  pass@1: 0.06
  protocol:
    protocol_version: humaneval-eval-v1
    protocol_id: humaneval-eval-v1:chat:humaneval-chat-instruction-v1:humaneval-extract-v1
    mode: chat
    prompt_template_id: humaneval-chat-instruction-v1
    prompt_template_sha256: ...
    prompt_template: ...
    extraction_policy_version: humaneval-extract-v1
    tokenizer_chat_template: "<effective tokenizer Jinja template, for chat>"
    tokenizer_chat_template_sha256: ...
    protocol_sha256: ...
    generation:
      max_new_tokens: 256
      do_sample: false
      force_length: false
      add_special_tokens: false
  cases:
    - task_id: HumanEval/0
      passed: true
      completion: "<raw model text>"
      extracted_code: "<program passed to exec>"
      extraction_error: null
```

`completion` is raw. `extracted_code` is the auditable input to the executor.
`extraction_error` records a rejected completion without discarding the raw text.

Chat tokenization disables additional special tokens because the rendered chat
template already contains its control tokens. Speed generation retains its
original tokenization behavior. The HF evaluation path records the effective
tokenizer template and generation settings; `protocol_sha256` covers this full
metadata. Missing or unusable chat templates fail before the speed/eval run,
rather than becoming a misleading series of failed generation cases.

Scores from different `mode` values are not comparable and must not be averaged
or placed in the same table without labeling the protocol. The protocol id/hash
travels with every score for this reason.

## Slice size and G0 calibration

`common.humaneval_n` defaults to 50 for the normal benchmark rows. The full
HumanEval set is 164 problems; set `humaneval_n: 164` for the G0 harness
calibration run. The slice size is a configuration choice and is never changed
based on a model's observed score.

G0 is a measurement of the harness, not a model-quality claim. The spec's G0
gate runs `HuggingFaceTB/SmolLM2-1.7B` base on HumanEval-164 with a frozen,
prewritten prompt template, and compares the score to the paper's 22.6 within
+/- 6 points. Do not tune prompts, temperature, commit mode, thresholds, or
parsing on HumanEval results. The score is recorded as harness calibration only;
HumanEval remains sealed for D1 model decisions.

A command sketch for a G0 calibration config:

```bash
python -m bench --config bench/config-g0.yaml --out results_g0.yaml
```

`bench/config-g0.yaml` would copy the common block with
`humaneval_n: 164`, select the SmolLM2-1.7B row, and set an explicit
`eval_prompt` mode.

## Evidence status

The historical `results_t4*.yaml` rows are completion-protocol, 50-problem
results from the earlier harness. They are not valid quality evidence for the
chat protocol, and no fixed accuracy claim should be made from them. A new
calibrated run under a recorded protocol is required before quoting a pass@1
number as a quality result.
