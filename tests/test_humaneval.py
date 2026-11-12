"""Safety and parsing regressions for the HumanEval executor."""

import time

from bench.humaneval import _extract, _run


def test_extract_uses_the_same_stripped_prompt_it_matches():
    prompt = "\n  def answer():\n    "
    completion = "prefix def answer():\n    return 42"
    assert _extract(completion, prompt) == "\n    return 42"


def test_run_accepts_a_valid_completion():
    prompt = "def answer():\n"
    completion = "    return 42\n"
    tests = "def check(fn):\n    assert fn() == 42"
    assert _run(prompt, completion, tests, "answer", timeout_seconds=0.5)


def test_run_times_out_infinite_generated_code():
    started = time.perf_counter()
    assert not _run("", "while True:\n    pass", "", "",
                    timeout_seconds=0.05)
    assert time.perf_counter() - started < 1.0
