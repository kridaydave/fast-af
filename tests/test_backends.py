"""Unit tests for the CustomBackend block-diffusion decode loop.

score_fn and tokenizer are fakes so the loop runs on CPU with no model
download. The fakes follow the score_fn contract: they never predict the
mask id (0) or pad id (2), the same guarantee load_smoldlm()'s pinning
provides. Real-model wiring is covered in test_smoldlm_wiring.py.
"""
from types import SimpleNamespace

import pytest
import torch

from bench.backends import CustomBackend

VOCAB = 64
MASK = CustomBackend.MASK_ID
EOS = 60
PREFIX = (10, 11)


class RecordingTokenizer:
    """Tokenizes to a fixed prefix and records exactly what decode receives."""

    def __init__(self):
        self.decoded = None

    def encode(self, text):
        return SimpleNamespace(ids=list(PREFIX))

    def decode(self, ids, skip_special_tokens=True):
        self.decoded = list(ids)
        return "decoded"


def position_scorer(conf_by_pos, calls):
    """Logits keyed by absolute sequence position.

    conf_by_pos(p) -> "high" pins token p at logit 20 (confidence ~1.0);
    "low" splits probability between tokens p and 32+p (confidence ~0.5,
    below the 0.9 threshold), with argmax tie-breaking to p.
    """
    def score_fn(seq):
        calls.append(seq.shape[1])
        logits = torch.full((1, seq.shape[1], VOCAB), -20.0)
        for p in range(seq.shape[1]):
            if conf_by_pos(p) == "high":
                logits[0, p, p] = 20.0
            else:
                logits[0, p, p] = 0.0
                logits[0, p, 32 + p] = 0.0
        return logits
    return score_fn


def make_backend(conf_by_pos, steps=4, confidence=0.9):
    calls = []
    tok = RecordingTokenizer()
    backend = CustomBackend(block_size=8, steps=steps, confidence=confidence,
                            score_fn=position_scorer(conf_by_pos, calls),
                            tokenizer=tok, eos_id=EOS)
    return backend, tok, calls


def test_confident_block_fills_in_one_pass():
    # every slot clears 0.9 on the first pass, so the block commits and the
    # loop must not spend another forward pass on it
    backend, tok, calls = make_backend(lambda p: "high")
    g = backend.generate("prompt", out_len=8)
    assert tok.decoded == [2, 3, 4, 5, 6, 7, 8, 9]  # block slots, no masks
    assert calls == [10]  # one score_fn call: prefix 2 + block 8
    assert g.num_tokens == 8
    assert g.decode_seconds > 0


def test_low_confidence_force_fills_on_last_pass():
    # nothing clears 0.9: each pass commits one top-1 slot so the loop
    # still makes progress, and the last pass force-fills the rest rather
    # than committing mask tokens
    backend, tok, calls = make_backend(lambda p: "low", steps=3)
    g = backend.generate("prompt", out_len=8)
    assert tok.decoded == [2, 3, 4, 5, 6, 7, 8, 9]
    assert len(calls) == 3  # all three passes consumed
    assert MASK not in tok.decoded
    assert g.num_tokens == 8


def test_threshold_miss_commits_one_slot_per_pass():
    # a pass whose threshold clears nothing must not re-score an identical
    # block: it commits exactly the single most confident open slot, so the
    # recorded block loses one mask per call until the last pass fills up
    blocks = []

    def score_fn(seq):
        blocks.append(seq[0, -8:].tolist())
        logits = torch.full((1, seq.shape[1], VOCAB), -20.0)
        for p in range(seq.shape[1]):
            logits[0, p, p] = 0.0          # conf ~0.5, below 0.9
            logits[0, p, 32 + p] = 0.0     # tie breaks argmax to p
        return logits

    tok = RecordingTokenizer()
    backend = CustomBackend(block_size=8, steps=4, confidence=0.9,
                            score_fn=score_fn, tokenizer=tok, eos_id=EOS)
    backend.generate("prompt", out_len=8)
    assert [b.count(MASK) for b in blocks] == [8, 7, 6, 5]
    assert MASK not in tok.decoded


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="steps"):
        make_backend(lambda p: "high", steps=0)


def test_unpinned_scorer_raises_instead_of_committing_masks():
    # score_fn that lets the mask id win argmax would silently write mask
    # tokens into the output; the loop must fail loudly instead
    def score_fn(seq):
        logits = torch.full((1, seq.shape[1], VOCAB), -20.0)
        logits[:, :, MASK] = 20.0  # mask always wins
        return logits

    tok = RecordingTokenizer()
    backend = CustomBackend(block_size=8, steps=4, confidence=0.9,
                            score_fn=score_fn, tokenizer=tok, eos_id=EOS)
    with pytest.raises(RuntimeError, match="mask id"):
        backend.generate("prompt", out_len=8)


def test_threshold_unmasks_partial_block_then_force_fills():
    # first block slot stays below threshold, the rest clear it: round one
    # fills seven slots, the last round fills the leftover
    backend, tok, calls = make_backend(
        lambda p: "low" if p == 2 else "high", steps=2)
    backend.generate("prompt", out_len=8)
    assert tok.decoded == [2, 3, 4, 5, 6, 7, 8, 9]
    assert len(calls) == 2
    assert MASK not in tok.decoded


def test_eos_truncates_block_and_stops():
    # slot for abs position 3 predicts EOS: everything after it is cut and
    # no second block runs
    def conf_by_pos(p):
        return "high"
    calls = []
    tok = RecordingTokenizer()

    def score_fn(seq):
        calls.append(seq.shape[1])
        logits = torch.full((1, seq.shape[1], VOCAB), -20.0)
        for p in range(seq.shape[1]):
            token = EOS if p == 3 else p
            logits[0, p, token] = 20.0
        return logits

    backend = CustomBackend(block_size=8, steps=4, confidence=0.9,
                            score_fn=score_fn, tokenizer=tok, eos_id=EOS)
    g = backend.generate("prompt", out_len=8)
    assert tok.decoded == [2]  # cut right after the first slot
    assert g.num_tokens == 1
    assert len(calls) == 1  # stopped before any further pass or block


def test_multiple_blocks_until_out_len():
    # out_len 12 needs two full blocks of 8 (the loop counts committed
    # tokens, so it overshoots to 16 like any block-aligned decode)
    backend, tok, calls = make_backend(lambda p: "high")
    g = backend.generate("prompt", out_len=12)
    assert tok.decoded == list(range(2, 18))
    assert g.num_tokens == 16


def test_warmup_runs():
    backend, tok, _ = make_backend(lambda p: "high")
    backend.warmup()
    assert tok.decoded is not None
