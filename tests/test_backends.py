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
    g = backend.generate("prompt", out_len=8)
    assert [b.count(MASK) for b in blocks] == [8, 7, 6, 5]
    assert MASK not in tok.decoded
    assert g.fill_counts == [1, 1, 1, 5]
    assert g.fallback_commits == 3


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="steps"):
        make_backend(lambda p: "high", steps=0)


def test_block_size_must_be_positive():
    with pytest.raises(ValueError, match="block_size"):
        CustomBackend(block_size=0, steps=4, confidence=0.9,
                      score_fn=lambda seq: None,
                      tokenizer=RecordingTokenizer())


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


def test_topk_fills_stock_schedule_counts():
    # commit="topk" ignores the threshold and fills k0 = block_size // steps
    # = 2 slots per pass (the modeling_smoldlm schedule), so one block
    # burns all 4 passes: masks left per pass go 8, 6, 4, 2
    blocks_seen = []

    def score_fn(seq):
        blocks_seen.append(seq[0, -8:].tolist())
        logits = torch.full((1, seq.shape[1], VOCAB), -20.0)
        for p in range(seq.shape[1]):
            logits[0, p, p] = 0.0          # conf ~0.5, below any threshold
            logits[0, p, 32 + p] = 0.0
        return logits

    tok = RecordingTokenizer()
    backend = CustomBackend(block_size=8, steps=4, confidence=0.9,
                            score_fn=score_fn, tokenizer=tok, eos_id=EOS,
                            commit="topk")
    g = backend.generate("prompt", out_len=8)
    assert [b.count(MASK) for b in blocks_seen] == [8, 6, 4, 2]
    assert MASK not in tok.decoded
    assert (g.passes, g.blocks) == (4, 1)


def test_pass_and_block_counters():
    # the spec wants passes/blocks in every results row; they must reflect
    # the loop: one confident block = 1 pass, threshold thrash = steps
    # passes, out_len over one block = two blocks
    backend, _, _ = make_backend(lambda p: "high")
    g = backend.generate("prompt", out_len=8)
    assert (g.passes, g.blocks) == (1, 1)

    backend, _, _ = make_backend(lambda p: "low", steps=3)
    g = backend.generate("prompt", out_len=8)
    assert (g.passes, g.blocks) == (3, 1)

    backend, _, _ = make_backend(lambda p: "high")
    g = backend.generate("prompt", out_len=12)
    assert (g.passes, g.blocks) == (2, 2)


def test_kv_cache_routes_through_hooks_not_score_fn():
    # kv_cache=True must never touch score_fn: the prefix is warmed into
    # the cache once, every denoising pass scores only the block at its
    # absolute position, and each committed block is appended to the cache
    warm, score, reset = [], [], []

    def score_block(block, pos_offset):
        score.append((tuple(block[0].tolist()), pos_offset))
        logits = torch.full((1, block.shape[1], VOCAB), -20.0)
        for i in range(block.shape[1]):
            logits[0, i, pos_offset + i] = 20.0  # absolute-position token
        return logits

    def score_fn(seq):
        raise AssertionError("score_fn called although kv_cache=True")

    tok = RecordingTokenizer()
    backend = CustomBackend(block_size=8, steps=4, confidence=0.9,
                            score_fn=score_fn, tokenizer=tok, eos_id=EOS,
                            kv_cache=True, cache_reset=lambda: reset.append(1),
                            cache_warm=lambda ids, pos: warm.append((tuple(ids), pos)),
                            score_block=score_block)
    g = backend.generate("prompt", out_len=8)
    assert reset == [1]                      # one cache clear per generate
    assert warm == [(PREFIX, 0),             # prefix warmed inside total timing
                    ((2, 3, 4, 5, 6, 7, 8, 9), 2)]  # committed block
    assert score == [((MASK,) * 8, 2)]       # one pass, block at pos 2
    assert tok.decoded == [2, 3, 4, 5, 6, 7, 8, 9]
    assert (g.passes, g.blocks) == (1, 1)
    assert g.warm_passes == 2
    assert g.stop_reason == "max_new_tokens"


def test_kv_cache_positions_advance_across_blocks():
    # out_len 12 needs two blocks; the second must score at pos 10 (prefix
    # 2 + first block 8) and warm at the same offset, never re-read pos 0
    warm, score = [], []

    def score_block(block, pos_offset):
        score.append(pos_offset)
        logits = torch.full((1, block.shape[1], VOCAB), -20.0)
        for i in range(block.shape[1]):
            logits[0, i, pos_offset + i] = 20.0
        return logits

    tok = RecordingTokenizer()
    backend = CustomBackend(block_size=8, steps=4, confidence=0.9,
                            score_fn=lambda seq: (_ for _ in ()).throw(
                                AssertionError("score_fn called")),
                            tokenizer=tok, eos_id=EOS,
                            kv_cache=True, cache_reset=lambda: None,
                            cache_warm=lambda ids, pos: warm.append((tuple(ids), pos)),
                            score_block=score_block)
    g = backend.generate("prompt", out_len=12)
    assert score == [2, 10]
    assert [pos for _, pos in warm] == [0, 2, 10]
    assert tok.decoded == list(range(2, 18))
    assert (g.passes, g.blocks) == (2, 2)


def test_kv_cache_requires_all_three_hooks():
    with pytest.raises(ValueError, match="cache"):
        CustomBackend(block_size=8, steps=4, confidence=0.9,
                      score_fn=lambda seq: None, tokenizer=RecordingTokenizer(),
                      kv_cache=True)


def test_unknown_commit_rejected():
    with pytest.raises(ValueError, match="commit"):
        CustomBackend(block_size=8, steps=4, confidence=0.9,
                      score_fn=lambda seq: None, tokenizer=RecordingTokenizer(),
                      commit="sampling")


def test_negative_temperature_rejected():
    with pytest.raises(ValueError, match="temperature"):
        CustomBackend(block_size=8, steps=4, confidence=0.9,
                      score_fn=lambda seq: None, tokenizer=RecordingTokenizer(),
                      temperature=-0.1)


def test_temperature_with_confidence_warns():
    with pytest.warns(RuntimeWarning, match="commit='confidence'"):
        CustomBackend(block_size=8, steps=4, confidence=0.9,
                      score_fn=lambda seq: None,
                      tokenizer=RecordingTokenizer(), temperature=0.7)


def test_temperature_samples_and_still_commits():
    # temperature > 0 goes through the stock gumbel path with commit="topk"
    # (the confidence threshold is calibrated to greedy probs, defect D3);
    # output must still be mask-free with the same stock pass schedule
    def score_fn(seq):
        logits = torch.full((1, seq.shape[1], VOCAB), -20.0)
        for p in range(seq.shape[1]):
            logits[0, p, p] = 20.0
        return logits

    tok = RecordingTokenizer()
    backend = CustomBackend(block_size=8, steps=4, confidence=0.9,
                            score_fn=score_fn, tokenizer=tok, eos_id=EOS,
                            temperature=1.0, commit="topk")
    g = backend.generate("prompt", out_len=8)
    assert g.num_tokens == 8
    assert MASK not in tok.decoded
    assert (g.passes, g.blocks) == (4, 1)  # k0 = 2 per pass, 8 slots


# ---- evaluation-only TransformersBackend path (no model download) ----

def _fake_transformers_backend(tokenizer):
    """Build a TransformersBackend without __init__ (no model download)."""
    from bench.backends import TransformersBackend
    backend = object.__new__(TransformersBackend)
    backend.tokenizer = tokenizer
    return backend


class ChatTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "<rendered>" + messages[-1]["content"]


def test_build_eval_prompt_completion_is_the_raw_prompt():
    backend = _fake_transformers_backend(None)
    prompt = "def add(a, b):\n"
    assert backend.build_eval_prompt(prompt, "completion") == prompt


def test_build_eval_prompt_chat_uses_the_fixed_instruction():
    from bench.humaneval import CHAT_INSTRUCTION
    tokenizer = ChatTokenizer()
    backend = _fake_transformers_backend(tokenizer)
    prompt = "def add(a, b):\n"
    rendered = backend.build_eval_prompt(prompt, "chat")
    assert rendered == "<rendered>" + prompt + "\n\n" + CHAT_INSTRUCTION
    assert tokenizer.messages[0]["role"] == "system"
    assert tokenizer.messages[1]["role"] == "user"
    assert tokenizer.messages[1]["content"] == prompt + "\n\n" + CHAT_INSTRUCTION


def test_build_eval_prompt_rejects_unknown_mode():
    backend = _fake_transformers_backend(None)
    import pytest
    with pytest.raises(ValueError, match="eval prompt mode"):
        backend.build_eval_prompt("x", "chat-ish")


def test_build_eval_prompt_chat_requires_a_chat_template():
    class NoChatTemplate:
        pass

    backend = _fake_transformers_backend(NoChatTemplate())
    import pytest
    with pytest.raises(ValueError, match="chat template"):
        backend.build_eval_prompt("x", "chat")


def test_generate_eval_uses_unbounded_decode():
    backend = _fake_transformers_backend(None)
    seen = {}

    def fake_generate(prompt, out_len, force_length=True):
        from bench.backends import Generation
        seen.update(prompt=prompt, out_len=out_len,
                    force_length=force_length)
        return Generation(text="ok", num_tokens=1, decode_seconds=0.1)

    backend.generate = fake_generate
    g = backend.generate_eval("prompt", "completion", max_new_tokens=12)
    assert g.text == "ok"
    assert seen == {"prompt": "prompt", "out_len": 12, "force_length": False}


def test_chat_eval_disables_duplicate_special_tokens_and_records_template_hash():
    from bench.backends import Generation
    import hashlib
    tok = ChatTokenizer()
    backend = _fake_transformers_backend(tok)
    seen = {}

    def generate(prompt, out_len, **kwargs):
        seen.update(kwargs)
        return Generation(text='ok', num_tokens=1, decode_seconds=.1)

    backend.generate = generate
    backend.generate_eval('problem', 'chat')
    assert seen == {'force_length': False, 'add_special_tokens': False}
    first = backend.evaluation_protocol('chat')
    assert first['tokenizer_chat_template_sha256'] == hashlib.sha256(b'fake-template').hexdigest()
    tok.chat_template = 'different-template'
    second = backend.evaluation_protocol('chat')
    assert first['protocol_sha256'] != second['protocol_sha256']


def test_chat_generation_tokenizes_rendered_prompt_without_extra_bos():
    from bench.backends import TransformersBackend
    import torch

    class Encoding(dict):
        def to(self, device):
            return self

    class Tok(ChatTokenizer):
        pad_token_id = 1

        def __call__(self, text, **kwargs):
            self.tokenize_kwargs = kwargs
            ids = [1, 5] if kwargs.get('add_special_tokens') is False else [1, 1, 5]
            return Encoding(input_ids=torch.tensor([ids]))

        def decode(self, ids, **kwargs):
            return 'answer'

    class Model:
        def __call__(self, **kwargs):
            return None

        def generate(self, **kwargs):
            self.input_ids = kwargs['input_ids']
            return torch.cat([self.input_ids, torch.tensor([[7]])], dim=1)

    backend = _fake_transformers_backend(Tok())
    backend.model, backend.device = Model(), 'cpu'
    backend.generate_eval('task', 'chat', max_new_tokens=4)
    assert backend.model.input_ids.tolist() == [[1, 5]]
    assert backend.tokenizer.tokenize_kwargs['add_special_tokens'] is False
    backend.generate('speed task', 1)
    assert 'add_special_tokens' not in backend.tokenizer.tokenize_kwargs
