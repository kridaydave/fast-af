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


# ---- evaluation protocol wiring ----

def test_humaneval_spec_requires_an_explicit_mode():
    from bench.run import _humaneval_spec

    common = {"humaneval": True, "humaneval_n": 50}
    assert _humaneval_spec({"humaneval": False, "eval_prompt": "chat"},
                           common) is None
    assert _humaneval_spec({"humaneval": True, "eval_prompt": "completion"},
                           common) == {"mode": "completion", "n": 50}
    assert _humaneval_spec({"humaneval": True, "eval_prompt": "chat",
                            "humaneval_n": 164}, common) == {
        "mode": "chat", "n": 164}

    import pytest
    with pytest.raises(ValueError, match="explicit eval_prompt"):
        _humaneval_spec({"id": "m", "humaneval": True}, common)
    with pytest.raises(ValueError, match="explicit eval_prompt"):
        _humaneval_spec({"id": "m", "humaneval": True,
                         "eval_prompt": "instruct"}, common)
    with pytest.raises(ValueError, match="1..164"):
        _humaneval_spec({"id": "m", "humaneval": True,
                         "eval_prompt": "completion",
                         "humaneval_n": 165}, common)


def test_run_model_uses_the_configured_chat_protocol(monkeypatch):
    import importlib
    run_module = importlib.import_module("bench.run")
    from bench.backends import Generation, TransformersBackend
    from bench.humaneval import CHAT_INSTRUCTION, build_protocol

    class ChatTokenizer:
        chat_template = "fake"

        def encode(self, text):
            return [ord(c) for c in text]

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            return "<chat>" + messages[-1]["content"]

    class FakeHF(TransformersBackend):
        def __init__(self):
            self.tokenizer = ChatTokenizer()
            self.prompts = []

        def warmup(self):
            pass

        def generate(self, prompt, out_len, force_length=True, add_special_tokens=True):
            self.special_tokens = add_special_tokens
            self.prompts.append((prompt, out_len, force_length))
            return Generation(text="```python\ndef f():\n    return 1\n```",
                              num_tokens=out_len, decode_seconds=0.1)

    captured = {}

    def fake_run_humaneval(gen_fn, prompts, mode, **kwargs):
        captured["prompts"] = prompts
        captured["mode"] = mode
        assert kwargs["protocol"]["tokenizer_chat_template_sha256"]
        captured["generated"] = gen_fn("PROBLEM")
        return {"n": prompts, "passed": 0, "pass@1": 0.0,
                "protocol": build_protocol(mode), "cases": []}

    backend = FakeHF()
    monkeypatch.setattr(run_module, "_build_backend", lambda m, common: backend)
    monkeypatch.setattr(run_module, "run_humaneval", fake_run_humaneval)
    monkeypatch.setattr(run_module, "_git", lambda *args: "sha")

    model = {"id": "chat-model", "backend": "hf", "repo": "x",
             "humaneval": True, "eval_prompt": "chat", "humaneval_n": 7}
    common = {"prompt_len": 4, "out_len": 4, "runs": 1, "batch": 1,
              "warmup": 0, "seed": 0, "dtype": "float32",
              "humaneval": True, "humaneval_n": 50}

    result = run_module.run_model(model, common)
    assert captured == {"prompts": 7, "mode": "chat",
                        "generated": "```python\ndef f():\n    return 1\n```"}
    assert backend.special_tokens is False
    assert CHAT_INSTRUCTION in backend.prompts[-1][0]
    # eval generation is not forced to the speed contract's exact length
    assert backend.prompts[-1][1:] == (256, False)
    assert result["humaneval"]["protocol"]["mode"] == "chat"


def test_build_backend_forwards_warm_batch(monkeypatch):
    import importlib
    run_module = importlib.import_module("bench.run")
    captured = {}

    def fake_load_smoldlm(repo, **kwargs):
        captured.update(kwargs)
        return "fake-backend"

    monkeypatch.setattr(run_module, "load_smoldlm", fake_load_smoldlm)
    m = {"backend": "custom", "repo": "x", "block_size": 32,
         "kv_cache": True, "warm_batch": True}
    assert run_module._build_backend(m, {"dtype": "float16"}) == "fake-backend"
    assert captured["warm_batch"] is True and captured["kv_cache"] is True
    # roster entries that don't mention it stay on the validated default
    del m["warm_batch"]
    run_module._build_backend(m, {"dtype": "float16"})
    assert captured["warm_batch"] is False


def test_validate_common_checks_humaneval_n():
    import pytest
    from bench.run import _validate_common

    _validate_common({"batch": 1, "prompt_len": 1, "out_len": 1, "runs": 1,
                      "humaneval_n": 164})
    with pytest.raises(ValueError, match="humaneval_n"):
        _validate_common({"batch": 1, "prompt_len": 1, "out_len": 1, "runs": 1,
                          "humaneval_n": 165})
