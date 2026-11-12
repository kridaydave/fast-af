"""HumanEval pass@1 guardrail (n=1, exec-based).

HumanEval is held out of all training data and touched only at eval. We run a
small held-out slice (default 50 problems) at n=1 and report the fraction whose
completion passes the function's test suite. Diffusion models opt out here
(they need a code-task decode loop of their own); the AR baselines are the
quality anchor.
"""
from __future__ import annotations

import json
import signal
from typing import Any

from datasets import load_dataset

# Canonical HumanEval. The raw file on openai/human-eval is GZIP'd (.jsonl.gz);
# the plain .jsonl path was deleted and 404s.
_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


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


def _extract(completion: str, prompt: str) -> str:
    """Completion is the continuation of prompt; if the model repeated the
    prompt verbatim, strip down to the part after it. No other surgery —
    prepending 'def ' or cutting at the next def corrupted valid bodies."""
    stripped = prompt.strip()
    if stripped and stripped in completion:
        return completion.split(stripped, 1)[1]
    return completion


def _run(prompt: str, completion: str, tests: str, entry_point: str,
         timeout_seconds: float = 5.0) -> bool:
    """Canonical HumanEval check: exec prompt+completion, then the dataset's
    test suite via check(entry_point). The prompt docstring's >>> examples
    are NOT tests — judging those made every parseable completion pass."""
    def timed_out(_signum, _frame):
        raise TimeoutError("HumanEval execution timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        exec_globals: dict[str, Any] = {}
        code = (prompt + "\n" + _extract(completion, prompt) + "\n"
                + tests + "\ncheck(" + entry_point + ")\n")
        exec(compile(code, "<eval>", "exec"), exec_globals)  # noqa: S102
        return True
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_humaneval(gen_fn, prompts: int = 50) -> dict[str, Any]:
    """gen_fn(prompt: str) -> str. Returns pass@1 stats."""
    rows = _load(prompts)
    passed = 0
    cases = []
    for r in rows:
        prompt = r["prompt"]
        try:
            completion = gen_fn(prompt)
        except Exception:
            completion = ""
        ok = _run(prompt, completion, r.get("test", ""),
                  r.get("entry_point", ""))
        cases.append({"task_id": r.get("task_id"), "passed": ok,
                      "completion": completion})
        if ok:
            passed += 1
    n = len(rows)
    return {"n": n, "passed": passed, "pass@1": passed / n if n else 0.0,
            "cases": cases}
