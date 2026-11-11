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
    assert out["passes"] == 4
    assert out["blocks"] == 1
    assert out["tokens_per_pass"] == 2.0  # 24 tokens / 12 passes


def test_timed_gen_omits_accounting_for_untracked_backends():
    # AR/dllm backends leave passes=0; the row must not carry fake zeros
    from bench.backends import Generation
    from bench.run import _timed_gen

    gens = [Generation(text="t", num_tokens=8, decode_seconds=0.5)
            for _ in range(2)]
    out = _timed_gen(StubBackend(gens), "p", out_len=8, runs=2)
    assert "passes" not in out
    assert "tokens_per_pass" not in out
    assert out["median_tps"] > 0
