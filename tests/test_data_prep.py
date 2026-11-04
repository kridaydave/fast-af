"""Tests for the HumanEval decontamination in scripts/data_prep.py.

load_humaneval() fetches the real 164-problem set (HF mirror, gz fallback),
so the leak tests run against actual HumanEval text, not invented fixtures.
"""
import pytest

from scripts.data_prep import (NGRAM, contaminated, humaneval_ngrams,
                               load_humaneval, sample)


def test_real_humaneval_builds_ngrams():
    rows = load_humaneval()
    assert len(rows) == 164
    grams = humaneval_ngrams(rows)
    assert len(grams) > 1000
    # a gram from the first prompt must be in the set
    words = rows[0]["prompt"].lower().split()
    assert tuple(words[:NGRAM]) in grams
    # solutions use the canonical_solution key; missing it once silently
    # dropped every solution gram from the decontamination set
    assert "canonical_solution" in rows[0]
    solution = tuple(rows[0]["canonical_solution"].lower().split()[:NGRAM])
    assert solution in grams
    assert contaminated(rows[0]["canonical_solution"], grams)


def test_doc_leaking_humaneval_prompt_is_dropped():
    rows = load_humaneval()
    grams = humaneval_ngrams(rows)
    leaked = "intro words " * 10 + rows[0]["prompt"]
    assert contaminated(leaked, grams)


def test_leak_detection_is_case_insensitive():
    rows = load_humaneval()
    grams = humaneval_ngrams(rows)
    assert contaminated(rows[0]["prompt"].upper(), grams)


def test_clean_doc_passes():
    grams = humaneval_ngrams(load_humaneval())
    clean = ("the quick brown fox jumped over the lazy dog while the "
             "compiler优化 the loop without asking anyone for permission ")
    assert not contaminated(clean, grams)


def test_doc_shorter_than_ngram_passes():
    grams = humaneval_ngrams(load_humaneval())
    assert not contaminated("just a few short words", grams)


def test_sample_writes_clean_docs_and_counts_dropped(tmp_path, capsys):
    # fake bucket: first doc contains the 13-gram, second is clean
    gram = " ".join(str(i) for i in range(NGRAM))
    docs = [{"text": f"{gram} tail text here"},
            {"text": "completely ordinary documentation about a function"}]
    grams = {tuple(gram.split())}
    out = tmp_path / "bucket.jsonl"
    sample(docs, "text", str(out), grams)
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    assert "ordinary documentation" in lines[0]
    assert "dropped 1 contaminated" in capsys.readouterr().out
    # success must not leave the temp file behind
    assert not (tmp_path / "bucket.jsonl.part").exists()


def test_sample_failure_preserves_previous_output(tmp_path):
    # a stream dying mid-write must not clobber the last good file:
    # output lands in .part first and only replaces on success
    out = tmp_path / "bucket.jsonl"
    out.write_text('{"text": "previous good run"}\n')

    def dying_stream():
        yield {"text": "doc one"}
        raise OSError("network died mid-iteration")

    with pytest.raises(OSError, match="network died"):
        sample(dying_stream(), "text", str(out), set())
    assert out.read_text() == '{"text": "previous good run"}\n'
    assert not (tmp_path / "bucket.jsonl.part").exists()


def _run_main(monkeypatch, run_result=None, run_error=None):
    """Drive main() with _run stubbed, capturing the os._exit code."""
    import scripts.data_prep as dp

    codes = []
    if run_error is None:
        monkeypatch.setattr(dp, "_run", lambda: run_result)
    else:
        def boom():
            raise run_error
        monkeypatch.setattr(dp, "_run", boom)
    # _exit never returns in production; recording it lets main() finish
    monkeypatch.setattr(dp.os, "_exit", codes.append)
    dp.main()
    assert len(codes) == 1, "main() must exit exactly once"
    return codes[0]


def test_main_exits_zero_when_every_bucket_passes(monkeypatch):
    assert _run_main(monkeypatch, run_result=True) == 0


def test_main_exits_one_when_a_bucket_fails(monkeypatch):
    # a failed bucket makes _run() return False; the script must still
    # exit non-zero (regression: this mapping was once inverted)
    assert _run_main(monkeypatch, run_result=False) == 1


def test_main_exits_one_when_run_raises(monkeypatch):
    # even an unexpected crash must hard-exit 1 instead of hanging on
    # datasets' atexit join
    assert _run_main(monkeypatch, run_error=RuntimeError("boom")) == 1
