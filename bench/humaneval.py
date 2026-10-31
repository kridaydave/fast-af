"""HumanEval pass@1 guardrail (n=1, exec-based).

HumanEval is held out of all training data and touched only at eval. We run a
small held-out slice (default 50 problems) at n=1 and report the fraction whose
completion passes the function's test suite. Diffusion models opt out here
(they need a code-task decode loop of their own); the AR baselines are the
quality anchor.
"""
from __future__ import annotations

import json
import re
from typing import Any

from datasets import load_dataset

# Canonical HumanEval, deduped to the standard 164.
_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"


def _load(prompts: int) -> list[dict[str, Any]]:
    rows = []
    try:
        ds = load_dataset("openai_humaneval", split="test")
        rows = list(ds)
    except Exception:
        # Fallback: fetch raw jsonl directly.
        import urllib.request
        with urllib.request.urlopen(_URL) as r:
            rows = [json.loads(line) for line in r if line.strip()]
    return [r for r in rows[:prompts]]


def _extract(completion: str, prompt: str) -> str:
    """Strip to the function body the prompt defines, up to the next def/EOF."""
    body = completion
    # Keep only code after the prompt, before any trailing new top-level def.
    if prompt and prompt in body:
        body = body.split(prompt, 1)[1]
    body = "def " + body if not body.lstrip().startswith(("def", "class", "@")) else body
    # cut at the next top-level function/class definition
    cut = re.split(r"\n(?:def|class)\s", "\n" + body.lstrip("\n"))
    return cut[0].strip()


def _run(prompt: str, completion: str) -> bool:
    try:
        exec_globals: dict[str, Any] = {}
        code = prompt + "\n" + _extract(completion, prompt) + "\n"
        exec(compile(code, "<eval>", "exec"), exec_globals)  # noqa: S102
        # run the assertions embedded in the prompt's docstring test block
        test = _extract_tests(prompt)
        if test:
            exec(compile(test, "<test>", "exec"), exec_globals)
        return True
    except Exception:
        return False


def _extract_tests(prompt: str) -> str:
    # HumanEval prompts embed checks() in the docstring; run them.
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not m:
        return ""
    doc = m.group(1)
    mm = re.search(r"check\(.*?\)\n(.*?)(\ncheck\()", doc, re.DOTALL)
    return mm.group(1) if mm else ""


def run_humaneval(gen_fn, prompts: int = 50) -> dict[str, Any]:
    """gen_fn(prompt: str) -> str. Returns pass@1 stats."""
    rows = _load(prompts)
    passed = 0
    for r in rows:
        prompt = r["prompt"]
        try:
            completion = gen_fn(prompt)
        except Exception:
            completion = ""
        if _run(prompt, completion):
            passed += 1
    n = len(rows)
    return {"n": n, "passed": passed, "pass@1": passed / n if n else 0.0}
