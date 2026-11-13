"""HumanEval pass@1 guardrail (n=1, exec-based).

HumanEval is held out of all training data and touched only at eval. We run a
held-out slice (default 50 problems, full 164 for calibration) at n=1 and
report the fraction whose completion passes the function's test suite.
Diffusion models opt out here (they need a code-task decode loop of their own);
the AR baselines are the quality guardrail.

Evaluation protocol v1 (see docs/EVALUATION_PROTOCOL.md)
--------------------------------------------------------
Every eval run declares an explicit ``mode``:

* ``completion`` -- the raw HumanEval prompt is the model prompt; the model
  continues it. This is the historical base-model protocol. No repo-name
  guessing: the mode is per-model config (``eval_prompt``), never inferred
  from a model id.
* ``chat`` -- the prompt is wrapped with the tokenizer's
  ``apply_chat_template`` and a fixed, prewritten instruction asking for one
  complete program in a single ```python fence. The raw generated text is
  preserved in the results row; extraction is a separate, auditable step.

Extraction policy v1 is deliberately conservative:

* completion mode: only an exact echoed prompt is removed from the
  continuation; nothing else is rewritten. The prompt is then prepended to
  the continuation, exactly as before, so base-model scores are comparable
  with the historical harness.
* chat mode: fenced output is authoritative. Exactly one Python-tagged fence
  is used verbatim (other-language fences are ignored); multiple Python
  fences are rejected as ambiguous. Untagged fences are accepted only when
  exactly one exists and its body parses as a Python module. Text outside the
  fence, including trailing prose, is ignored.
* chat mode without a fence: the whole response must parse as a module with
  at least one top-level function. If it does not, an exact echoed prompt
  prefix is removed and the remainder is retried. Prose-only or unparseable
  output is rejected rather than guessed at.
* Extraction never prepends ``def``, cuts at the next ``def``, strips
  imports, or rewrites signatures. Chat programs are standalone: the HumanEval
  prompt is not prepended a second time.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import signal
from typing import Any

from datasets import load_dataset

# Canonical HumanEval. The raw file on openai/human-eval is GZIP'd (.jsonl.gz);
# the plain .jsonl path was deleted and 404s.
_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

MAX_HUMANEVAL = 164
DEFAULT_HUMANEVAL_N = 50

PROTOCOL_VERSION = "humaneval-eval-v1"
CHAT_TEMPLATE_ID = "humaneval-chat-instruction-v1"
COMPLETION_TEMPLATE_ID = "humaneval-raw-completion-v1"
EXTRACTION_POLICY_VERSION = "humaneval-extract-v1"
EVAL_MODES = ("completion", "chat")

# Fixed, prewritten instruction. Do not tune this against HumanEval scores;
# any change is a new template id and invalidates protocol comparisons.
CHAT_SYSTEM = "You are an expert Python programmer."
CHAT_INSTRUCTION = (
    "Write one complete, self-contained Python program that solves the problem. "
    "Put the entire program inside a single ```python fenced code block. "
    "Include every import the program needs and the required function signature. "
    "Do not write any explanation, prose, or text outside the code block."
)
_CHAT_TEMPLATE_TEXT = CHAT_SYSTEM + "\n\n" + CHAT_INSTRUCTION

_PYTHON_LANGS = {"python", "py", "python3"}
_FENCE_OPEN = re.compile(r"^\s*(`{3,}|~{3,})\s*([A-Za-z0-9_+-]*)\s*$")
_FENCE_CLOSE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")


class ExtractionError(ValueError):
    """An eval completion cannot be turned into a runnable HumanEval program."""


def _load(prompts: int) -> list[dict[str, Any]]:
    rows = []
    try:
        # The dataset moved out of the top-level namespace; datasets does not
        # reliably follow the 307, so query the new location directly.
        ds = load_dataset("openai/openai_humaneval", split="test")
        rows = list(ds)
    except Exception:
        # Fallback: fetch the raw gzipped jsonl directly.
        import gzip
        import urllib.request
        with urllib.request.urlopen(_URL) as r:
            rows = [json.loads(line) for line in gzip.decompress(r.read())
                    .decode("utf-8").splitlines() if line.strip()]
    return [r for r in rows[:prompts]]


def chat_messages(prompt: str) -> list[dict[str, str]]:
    """The exact chat messages used for a chat-mode eval prompt."""
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": prompt + "\n\n" + CHAT_INSTRUCTION},
    ]


def build_protocol(mode: str) -> dict[str, Any]:
    """Stable, auditable description of the prompt/extraction protocol."""
    if mode == "chat":
        template_text = _CHAT_TEMPLATE_TEXT
        template_id = CHAT_TEMPLATE_ID
    elif mode == "completion":
        # No wrapper text: the raw HumanEval prompt is the prompt.
        template_text = ""
        template_id = COMPLETION_TEMPLATE_ID
    else:
        raise ValueError(
            f"eval mode must be one of {EVAL_MODES}, got {mode!r}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_id": (f"{PROTOCOL_VERSION}:{mode}:{template_id}:"
                        f"{EXTRACTION_POLICY_VERSION}"),
        "mode": mode,
        "prompt_template_id": template_id,
        "prompt_template_sha256": hashlib.sha256(
            template_text.encode("utf-8")).hexdigest(),
        "prompt_template": template_text,
        "extraction_policy_version": EXTRACTION_POLICY_VERSION,
    }


def _strip_echo(completion: str, prompt: str) -> str:
    """Remove an exact echoed prompt. This is the only completion-mode surgery:
    prepending 'def ' or cutting at the next def corrupted valid bodies."""
    stripped = prompt.strip()
    if stripped and stripped in completion:
        return completion.split(stripped, 1)[1]
    return completion


def _fenced_blocks(text: str) -> list[tuple[str, str, bool]]:
    """Return (language, body, closed) for every Markdown fence in text.

    A fence may be backtick- or tilde-delimited and at least three characters
    long. An unclosed fence runs to the end of the text; chat models often
    omit the closing fence, and rejecting it would discard a valid program.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, str, bool]] = []
    i = 0
    while i < len(lines):
        opener = _FENCE_OPEN.match(lines[i])
        if not opener:
            i += 1
            continue
        fence = opener.group(1)
        language = opener.group(2).lower()
        body: list[str] = []
        i += 1
        closed = False
        while i < len(lines):
            closer = _FENCE_CLOSE.match(lines[i])
            if (closer and closer.group(1)[0] == fence[0]
                    and len(closer.group(1)) >= len(fence)):
                closed = True
                i += 1
                break
            body.append(lines[i])
            i += 1
        blocks.append((language, "\n".join(body), closed))
    return blocks


def _validate_program(code: str) -> None:
    """Reject empty, unparseable, or functionless programs.

    A HumanEval entry point is a top-level function, so a module with no
    top-level ``def`` cannot be scored. ``ast.parse`` also rejects prose and
    truncated bodies instead of letting the executor fail obscurely.
    """
    if not code.strip():
        raise ExtractionError("empty program")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ExtractionError(
            f"unparseable program: {exc.msg} at line {exc.lineno}") from exc
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               for node in tree.body):
        raise ExtractionError("program has no top-level function definition")


def _select_fenced(blocks: list[tuple[str, str, bool]]) -> str:
    """Choose one fenced program under the documented fence policy."""
    python = [b for b in blocks if b[0] in _PYTHON_LANGS]
    if len(python) == 1:
        return python[0][1]
    if len(python) > 1:
        raise ExtractionError(
            "multiple Python fences are ambiguous; expected one")
    untagged = [b for b in blocks if b[0] == ""]
    if len(untagged) == 1:
        code = untagged[0][1]
        try:
            _validate_program(code)
        except ExtractionError:
            pass
        else:
            return code
    if untagged:
        raise ExtractionError(
            "no usable Python fence; untagged fences are ambiguous")
    raise ExtractionError("no fenced code block")


def _extract_plain_chat(text: str, prompt: str) -> str:
    """No-fence chat output: whole response, or response minus an echo."""
    candidate = text.strip()
    try:
        _validate_program(candidate)
        return candidate
    except ExtractionError:
        pass
    # The model may have echoed the prompt and then continued it. Keep the
    # echo when the combined text is already a valid program (this preserves
    # the function signature); otherwise drop the leading echo and retry.
    stripped_prompt = prompt.strip()
    if stripped_prompt and candidate.startswith(stripped_prompt):
        remainder = candidate[len(stripped_prompt):].strip()
        try:
            _validate_program(remainder)
        except ExtractionError:
            pass
        else:
            return remainder
    if stripped_prompt and stripped_prompt in candidate:
        remainder = candidate.split(stripped_prompt, 1)[1].strip()
        try:
            _validate_program(remainder)
        except ExtractionError:
            pass
        else:
            return remainder
    raise ExtractionError(
        "no fenced program and the response is not a complete Python module")


def _extract_chat(completion: str, prompt: str) -> str:
    """Extract a standalone program from a chat-mode completion."""
    text = completion.strip()
    if not text:
        raise ExtractionError("empty completion")
    blocks = _fenced_blocks(text)
    if blocks:
        code = _select_fenced(blocks)
    else:
        code = _extract_plain_chat(text, prompt)
    # Inside a fence the prompt is part of the model's program, not a wrapper:
    # stripping it would delete the function signature. Only surrounding
    # whitespace is normalized.
    code = code.strip("\n")
    _validate_program(code)
    return code


def _extract(completion: str, prompt: str, mode: str = "completion") -> str:
    """Extract the runnable portion of a completion for ``mode``."""
    if mode == "completion":
        return _strip_echo(completion, prompt)
    if mode == "chat":
        return _extract_chat(completion, prompt)
    raise ValueError(f"eval mode must be one of {EVAL_MODES}, got {mode!r}")


def _program(prompt: str, completion: str, mode: str) -> str:
    """The exact text handed to exec for a case."""
    extracted = _extract(completion, prompt, mode)
    if mode == "chat":
        # A chat response is a full program; prepending the prompt again
        # would duplicate the function or corrupt an echoed prompt.
        return extracted
    return prompt + "\n" + extracted


def _run_program(program: str, tests: str, entry_point: str,
                 timeout_seconds: float = 5.0) -> bool:
    """Exec a runnable program plus the dataset test suite."""
    def timed_out(_signum, _frame):
        raise TimeoutError("HumanEval execution timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        exec_globals: dict[str, Any] = {}
        code = (program + "\n" + tests + "\ncheck(" + entry_point + ")\n")
        exec(compile(code, "<eval>", "exec"), exec_globals)  # noqa: S102
        return True
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run(prompt: str, completion: str, tests: str, entry_point: str,
         timeout_seconds: float = 5.0, mode: str = "completion") -> bool:
    """Canonical HumanEval check for one completion.

    The prompt docstring's >>> examples are NOT tests -- judging those made
    every parseable completion pass. Extraction errors fail the case.
    """
    try:
        program = _program(prompt, completion, mode)
    except ExtractionError:
        return False
    return _run_program(program, tests, entry_point, timeout_seconds)


def run_humaneval(gen_fn, prompts: int = DEFAULT_HUMANEVAL_N,
                  mode: str = "completion", timeout_seconds: float = 5.0,
                  protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run HumanEval at n=1 and return pass@1 plus raw, auditable cases.

    ``gen_fn(prompt: str) -> str`` returns the raw model completion. The
    results row records that raw text, the extracted program, any extraction
    error, and a stable protocol description. Different protocols (for
    example completion vs chat) are never silently merged: the protocol id and
    hash travel with every score.
    """
    if mode not in EVAL_MODES:
        raise ValueError(f"eval mode must be one of {EVAL_MODES}, got {mode!r}")
    if type(prompts) is not int or not 1 <= prompts <= MAX_HUMANEVAL:
        raise ValueError(
            f"HumanEval slice must be 1..{MAX_HUMANEVAL}, got {prompts}")
    expected = build_protocol(mode)
    if protocol is not None and not isinstance(protocol, dict):
        raise ValueError("provided protocol must be a mapping")
    if protocol is not None and any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("provided protocol does not match the evaluation mode/template")
    protocol = protocol or expected
    rows = _load(prompts)
    passed = 0
    cases = []
    for r in rows:
        prompt = r["prompt"]
        completion: Any = ""
        generation_error = None
        try:
            completion = gen_fn(prompt)
        except Exception as exc:
            generation_error = repr(exc)
        if not isinstance(completion, str):
            completion = "" if completion is None else str(completion)
        case: dict[str, Any] = {"task_id": r.get("task_id"), "passed": False,
                                "completion": completion}
        if generation_error is not None:
            case["generation_error"] = generation_error
        try:
            program = _program(prompt, completion, mode)
        except ExtractionError as exc:
            case["extraction_error"] = str(exc)
            cases.append(case)
            continue
        ok = _run_program(program, r.get("test", ""),
                          r.get("entry_point", ""), timeout_seconds)
        case["passed"] = ok
        case["extracted_code"] = program
        if ok:
            passed += 1
        cases.append(case)
    n = len(rows)
    return {"n": n, "passed": passed, "pass@1": passed / n if n else 0.0,
            "protocol": protocol, "cases": cases}
