"""Safety and parsing regressions for the HumanEval executor and eval protocol."""
import time

import pytest

from bench.humaneval import (CHAT_INSTRUCTION, EXTRACTION_POLICY_VERSION,
                             ExtractionError, _extract, _fenced_blocks,
                             _program, _run, build_protocol, run_humaneval)

PROMPT = "def add(a, b):\n    '''Add two numbers.'''\n"


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


# ---- completion mode: preserve the historical base-model behavior ----

def test_completion_mode_keeps_nested_definitions_and_imports():
    prompt = "def outer():\n"
    completion = ("    import math\n"
                  "    def inner(x):\n"
                  "        return math.sqrt(x)\n"
                  "    return inner(4)\n")
    extracted = _extract(completion, prompt, mode="completion")
    assert extracted == completion
    # The prompt is prepended once; the body is not rewritten.
    assert _program(prompt, completion, "completion") == prompt + "\n" + completion


def test_completion_mode_does_not_treat_fences_as_code():
    prompt = "def add(a, b):\n"
    completion = "```python\n    return a + b\n```\n"
    # No fence surgery in completion mode: changing it would alter historical
    # scores, and a fenced base continuation is an honest failure.
    assert _extract(completion, prompt, mode="completion") == completion


# ---- chat mode: fenced and plain programs ----

def test_chat_plain_code_is_used_verbatim():
    code = "import math\n\ndef add(a, b):\n    return a + b\n"
    assert _extract(code, PROMPT, mode="chat") == code.strip()


def test_chat_fenced_code_ignores_prose_and_trailing_text():
    completion = ("Sure, here is the solution:\n\n"
                  "```python\n"
                  "import math\n\n"
                  "def add(a, b):\n"
                  "    return a + b\n"
                  "```\n\n"
                  "This function adds two numbers.\n")
    extracted = _extract(completion, PROMPT, mode="chat")
    assert extracted == "import math\n\ndef add(a, b):\n    return a + b"
    assert "Sure" not in extracted
    assert "This function" not in extracted


def test_chat_echoed_prompt_inside_a_fence_keeps_the_signature():
    completion = "```python\n" + PROMPT + "    return a + b\n```"
    extracted = _extract(completion, PROMPT, mode="chat")
    assert extracted.startswith("def add(a, b):")
    assert "'''Add two numbers.'''" in extracted
    assert extracted.endswith("return a + b")


def test_chat_echoed_prompt_without_a_fence_is_not_double_prefixed():
    # The echo already contains the signature; dropping it would leave a bare
    # body, so the combined text must be kept when it parses.
    completion = PROMPT + "    return a + b\n"
    assert _extract(completion, PROMPT, mode="chat") == completion.strip()
    assert _program(PROMPT, completion, "chat") == completion.strip()


def test_chat_nested_functions_are_preserved():
    completion = ("```python\n"
                  "def add(a, b):\n"
                  "    def helper(x):\n"
                  "        return x + 1\n"
                  "    return helper(a) + b\n"
                  "```")
    extracted = _extract(completion, PROMPT, mode="chat")
    assert "def helper(x):" in extracted
    assert extracted.count("def ") == 2


def test_chat_multiple_python_fences_are_ambiguous():
    completion = ("```python\ndef a():\n    return 1\n```\n"
                  "```python\ndef b():\n    return 2\n```")
    with pytest.raises(ExtractionError, match="multiple Python fences"):
        _extract(completion, PROMPT, mode="chat")


def test_chat_single_python_fence_wins_over_other_languages():
    completion = ("```bash\npip install x\n```\n"
                  "```python\ndef add(a, b):\n    return a + b\n```")
    assert _extract(completion, PROMPT, mode="chat") == (
        "def add(a, b):\n    return a + b")


def test_chat_untagged_fence_is_used_when_it_is_python():
    completion = "```\ndef add(a, b):\n    return a + b\n```"
    assert _extract(completion, PROMPT, mode="chat") == (
        "def add(a, b):\n    return a + b")


def test_chat_unclosed_fence_is_accepted():
    completion = "```python\ndef add(a, b):\n    return a + b\n"
    assert _extract(completion, PROMPT, mode="chat") == (
        "def add(a, b):\n    return a + b")


def test_chat_plain_prose_is_rejected():
    with pytest.raises(ExtractionError, match="not a complete Python module"):
        _extract("Here is the solution:\n" + PROMPT, PROMPT, mode="chat")


def test_chat_plain_code_with_trailing_prose_is_rejected():
    # Conservative policy: without a fence, the whole response must parse.
    # A fenced answer is the supported way to mix code and prose.
    completion = "def add(a, b):\n    return a + b\n\nThis adds two numbers."
    with pytest.raises(ExtractionError, match="not a complete Python module"):
        _extract(completion, PROMPT, mode="chat")


def test_chat_rejects_a_program_with_no_function():
    with pytest.raises(ExtractionError, match="no top-level function"):
        _extract("```python\nimport math\n```", PROMPT, mode="chat")


def test_chat_rejects_an_unparseable_fence():
    with pytest.raises(ExtractionError, match="unparseable"):
        _extract("```python\ndef broken(:\n```", PROMPT, mode="chat")


def test_chat_run_does_not_double_append_the_prompt():
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    tests = "def check(fn):\n    assert fn(2, 3) == 5"
    assert _run(PROMPT, completion, tests, "add", mode="chat")


def test_fenced_blocks_reports_closing_state():
    blocks = _fenced_blocks("```python\nx = 1\n```\n```\ny = 2\n")
    assert blocks == [("python", "x = 1", True), ("", "y = 2", False)]


# ---- protocol and results audit ----

def test_protocol_is_stable_and_distinguishes_modes():
    completion = build_protocol("completion")
    chat = build_protocol("chat")
    assert completion["mode"] == "completion"
    assert chat["mode"] == "chat"
    assert completion["prompt_template_id"] != chat["prompt_template_id"]
    assert completion["prompt_template_sha256"] != chat["prompt_template_sha256"]
    assert chat["prompt_template"] == (
        "You are an expert Python programmer.\n\n" + CHAT_INSTRUCTION)
    assert completion["extraction_policy_version"] == EXTRACTION_POLICY_VERSION
    assert chat["extraction_policy_version"] == EXTRACTION_POLICY_VERSION
    assert completion["protocol_id"] != chat["protocol_id"]
    assert chat["protocol_id"] == (
        "humaneval-eval-v1:chat:humaneval-chat-instruction-v1:"
        "humaneval-extract-v1")


def test_run_humaneval_records_protocol_raw_completion_and_errors(monkeypatch):
    rows = [
        {"task_id": "ok", "prompt": PROMPT,
         "test": "def check(fn):\n    assert fn(2, 3) == 5",
         "entry_point": "add"},
        {"task_id": "bad", "prompt": PROMPT,
         "test": "def check(fn):\n    assert fn(2, 3) == 5",
         "entry_point": "add"},
    ]

    def fake_load(prompts):
        assert prompts == 2
        return rows

    monkeypatch.setattr("bench.humaneval._load", fake_load)
    completions = iter([
        "```python\ndef add(a, b):\n    return a + b\n```",
        "I cannot solve this problem.",
    ])
    result = run_humaneval(lambda prompt: next(completions), prompts=2,
                           mode="chat")
    assert result["n"] == 2
    assert result["passed"] == 1
    assert result["pass@1"] == 0.5
    assert result["protocol"]["mode"] == "chat"
    assert result["protocol"]["prompt_template_id"]
    assert result["protocol"]["prompt_template_sha256"]
    ok, bad = result["cases"]
    assert ok["passed"] is True
    assert ok["completion"].startswith("```python")
    assert "def add" in ok["extracted_code"]
    assert bad["passed"] is False
    assert bad["completion"] == "I cannot solve this problem."
    assert "extraction_error" in bad


def test_run_humaneval_rejects_out_of_range_slice():
    with pytest.raises(ValueError, match="1..164"):
        run_humaneval(lambda prompt: "", prompts=0, mode="completion")
    with pytest.raises(ValueError, match="1..164"):
        run_humaneval(lambda prompt: "", prompts=165, mode="completion")


def test_run_humaneval_rejects_unknown_mode():
    with pytest.raises(ValueError, match="eval mode"):
        run_humaneval(lambda prompt: "", prompts=1, mode="chat-ish")

def test_run_humaneval_completion_mode_keeps_raw_continuation(monkeypatch):
    rows = [{"task_id": "add", "prompt": PROMPT,
             "test": "def check(fn):\n    assert fn(2, 3) == 5",
             "entry_point": "add"}]
    monkeypatch.setattr("bench.humaneval._load", lambda prompts: rows)
    result = run_humaneval(lambda prompt: "    return a + b\n", prompts=1,
                           mode="completion")
    assert result["passed"] == 1
    assert result["protocol"]["mode"] == "completion"
    assert result["cases"][0]["completion"] == "    return a + b\n"
    assert result["cases"][0]["extracted_code"] == PROMPT + "\n    return a + b\n"


def test_run_humaneval_accepts_the_full_164_slice(monkeypatch):
    seen = []
    monkeypatch.setattr("bench.humaneval._load",
                        lambda prompts: seen.append(prompts) or [])
    run_humaneval(lambda prompt: "", prompts=164, mode="completion")
    assert seen == [164]


def test_run_humaneval_rejects_mislabelled_protocol_before_loading(monkeypatch):
    from bench.humaneval import build_protocol
    with pytest.raises(ValueError, match='provided protocol'):
        run_humaneval(lambda p: '', prompts=1, mode='chat',
                      protocol=build_protocol('completion'))


@pytest.mark.parametrize('count', [True, 3.5, '164'])
def test_humaneval_rejects_noninteger_case_counts(count):
    with pytest.raises(ValueError, match='slice'):
        run_humaneval(lambda p: '', prompts=count)
