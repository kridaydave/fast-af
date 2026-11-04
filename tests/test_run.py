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
