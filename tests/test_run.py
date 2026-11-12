"""Tests for speed-harness prompt fitting (the spec's 1k-in timing contract).

fit_prompt must hit the token target exactly with both tokenizer APIs the
harness meets: HF (encode returns list[int]) and tokenizers.Tokenizer
(encode returns an Encoding with .ids).
"""
from bench.run import _SPEED_PROMPTS, fit_prompt

TARGET = 1024


class ListTokenizer:
    """HF-style: encode -> list[int], exact round-trip."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class EncodingTokenizer:
    """tokenizers.Tokenizer-style: encode -> object with .ids."""

    class Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        return self.Enc([ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class DriftingTokenizer:
    """decode loses one char per round-trip, the failure fit_prompt guards:
    decode -> encode drifts at the cut, so the loop must converge or stop."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)[:-1]


def test_fit_prompt_hits_target_exactly():
    text, n = fit_prompt(_SPEED_PROMPTS, ListTokenizer(), TARGET)
    assert n == TARGET
    # content must come from the speed prompts, not invented filler
    assert text.startswith(_SPEED_PROMPTS[0][:50])


def test_fit_prompt_accepts_encoding_style_tokenizer():
    text, n = fit_prompt(_SPEED_PROMPTS, EncodingTokenizer(), 500)
    assert n == 500
    assert isinstance(text, str)


def test_fit_prompt_cycles_short_prompts_to_reach_target():
    # three prompts are far shorter than 1024 chars, so cycling must happen
    text, n = fit_prompt(["ab", "cd"], ListTokenizer(), 64)
    assert n == 64
    assert "ab" in text and "cd" in text


def test_fit_prompt_terminates_on_roundtrip_drift():
    # drift can prevent an exact hit; the loop is capped at 3 tries and
    # must still return rather than spin forever
    text, n = fit_prompt(_SPEED_PROMPTS, DriftingTokenizer(), TARGET)
    assert isinstance(text, str)
    assert abs(n - TARGET) <= 3


class StubBackend:
    """Returns prebuilt Generations so _timed_gen's accounting is testable."""

    def __init__(self, generations):
        self._gens = list(generations)

    def generate(self, prompt, out_len):
        return self._gens.pop(0)


def test_timed_gen_records_pass_accounting():
    # the spec requires tokens_emitted/passes/blocks/tokens_per_pass in the
    # results row for diffusion backends
    from bench.backends import Generation
    from bench.run import _timed_gen

    gens = [Generation(text="t", num_tokens=8, decode_seconds=0.5,
                       passes=4, blocks=1) for _ in range(3)]
    out = _timed_gen(StubBackend(gens), "p", out_len=8, runs=3)
    assert out["tokens_emitted"] == 24
    assert out["passes"] == 12
    assert out["passes_per_run"] == [4, 4, 4]
    assert out["blocks"] == 3
    assert out["tokens_per_pass"] == 2.0  # 24 tokens / 12 passes


def test_timed_gen_records_zero_accounting_for_untracked_backends():
    # The result schema is stable across backends; AR/dllm report explicit 0s.
    from bench.backends import Generation
    from bench.run import _timed_gen

    gens = [Generation(text="t", num_tokens=8, decode_seconds=0.5)
            for _ in range(2)]
    out = _timed_gen(StubBackend(gens), "p", out_len=8, runs=2)
    assert out["tokens_emitted"] == 16
    assert out["passes"] == 0
    assert "tokens_per_pass" not in out
    assert out["median_tps"] > 0


def test_timed_gen_uses_one_aggregation_basis():
    from bench.backends import Generation
    from bench.run import _timed_gen

    gens = [Generation(text="a", num_tokens=8, decode_seconds=1, passes=4),
            Generation(text="b", num_tokens=8, decode_seconds=1, passes=1)]
    out = _timed_gen(StubBackend(gens), "p", out_len=8, runs=2)
    assert out["tokens_emitted"] == 16
    assert out["passes"] == 5
    assert out["tokens_per_pass"] == 3.2


def test_timed_gen_rejects_short_speed_rows():
    from bench.backends import Generation
    from bench.run import _timed_gen

    gens = [Generation(text="short", num_tokens=4, decode_seconds=1,
                       stop_reason="eos")]
    import pytest
    with pytest.raises(RuntimeError, match="exactly 8 output tokens"):
        _timed_gen(StubBackend(gens), "p", out_len=8, runs=1)


def test_comparisons_report_token_agreement_and_speed_ratio():
    from bench.run import _comparisons

    results = [
        {"id": "base", "config": {},
         "speed": {"sample_token_ids": [1, 2, 3], "median_tps": 10.0}},
        {"id": "cached", "config": {"compare_to": "base",
                                      "semantic_equivalence_claimed": False},
         "speed": {"sample_token_ids": [1, 9, 3], "median_tps": 25.0}},
    ]
    comparison = _comparisons(results)[0]
    assert comparison["exact_token_match"] is False
    assert comparison["token_agreement"] == 0.666667
    assert comparison["median_tps_ratio"] == 2.5
    assert comparison["semantic_equivalence_claimed"] is False


def test_run_persists_atomically_after_each_model(tmp_path, monkeypatch):
    import importlib
    import yaml
    run_module = importlib.import_module("bench.run")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "common": {"prompt_len": 8, "out_len": 8, "runs": 1,
                   "batch": 1, "warmup": 0, "seed": 0,
                   "dtype": "float32"},
        "models": [{"id": "ok", "backend": "hf"},
                   {"id": "bad", "backend": "hf"}],
    }))
    calls = []

    def fake_run_model(model, common):
        calls.append(model["id"])
        if model["id"] == "bad":
            raise RuntimeError("boom")
        return {"id": "ok", "speed": {}}

    monkeypatch.setattr(run_module, "run_model", fake_run_model)
    monkeypatch.setattr(run_module, "_provenance", lambda path: {
        "git_sha": "abc", "config_sha256": "hash"})
    out_path = tmp_path / "results.yaml"
    rows = run_module.run(str(cfg), str(out_path))
    saved = yaml.safe_load(out_path.read_text())
    assert calls == ["ok", "bad"]
    assert rows[1]["error"] == "RuntimeError('boom')"
    assert len(saved["results"]) == 2
    assert not (tmp_path / "results.yaml.part").exists()
