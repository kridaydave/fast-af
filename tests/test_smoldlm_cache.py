"""Focused cache tests for the smoldlm cached policy.

Two layers:

- Fake-hook tests run hermetically on CPU: they pin what CustomBackend
  does with the cache hooks (warm/score ordering, call counts, the
  cache_warm_all fast path) with no model download.
- Real-model tests load HoangHa/smoldlm-144m on CPU and pin the two
  correctness claims behind the batched warm (warm_batch=True): its cache
  matches the chunked schedule's, and greedy generations are token-for-token
  identical. They also pin no_grad on cache warm and reset/stale-mask
  hygiene across repeated generates.
"""
from types import SimpleNamespace

import pytest
import torch

from bench.backends import CustomBackend, load_smoldlm

REPO = "HoangHa/smoldlm-144m"
MASK = CustomBackend.MASK_ID
VOCAB = 64
PREFIX = tuple(range(10, 18))  # 8 fake prefix tokens, exactly one block


class RecordingTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=list(PREFIX))

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


def _make_backend(cache_warm_all=None, log=None):
    """kv_cache backend with recording hooks. Every hook appends a tag to
    `log` so ordering is observable; score_block pins token (offset + i)
    at slot i, so commits are deterministic."""
    log = log if log is not None else []

    def cache_warm(ids, pos):
        log.append(("warm", tuple(ids), pos))

    def score_block(block, pos_offset):
        log.append(("score", tuple(block[0].tolist()), pos_offset))
        logits = torch.full((1, block.shape[1], VOCAB), -20.0)
        for i in range(block.shape[1]):
            logits[0, i, (pos_offset + i) % VOCAB] = 20.0
        return logits

    backend = CustomBackend(
        block_size=8, steps=4, confidence=0.9,
        score_fn=lambda seq: (_ for _ in ()).throw(
            AssertionError("score_fn called although kv_cache=True")),
        tokenizer=RecordingTokenizer(), eos_id=60,
        kv_cache=True,
        cache_reset=lambda: log.append(("reset",)),
        cache_warm=cache_warm,
        score_block=score_block,
        cache_warm_all=cache_warm_all)
    return backend, log


def test_warm_all_hook_gets_whole_prefix_in_one_call():
    log = []
    backend, _ = _make_backend(
        cache_warm_all=lambda ids: log.append(("warm_all", tuple(ids))),
        log=log)
    g = backend.generate("prompt", out_len=8)
    # reset, then the whole prefix through one call, then one score pass,
    # then the commit warm at the right absolute offset
    assert log == [("reset",), ("warm_all", PREFIX),
                   ("score", (MASK,) * 8, 8),
                   ("warm", (8, 9, 10, 11, 12, 13, 14, 15), 8)]
    assert g.warm_passes == 1 + g.blocks == 2


def test_chunked_warm_counts_are_unchanged_without_hook():
    backend, log = _make_backend()
    g = backend.generate("prompt", out_len=8)
    # prefix (8 ids = one block) chunk-warmed, then one committed block
    assert log == [("reset",), ("warm", PREFIX, 0),
                   ("score", (MASK,) * 8, 8),
                   ("warm", (8, 9, 10, 11, 12, 13, 14, 15), 8)]
    assert g.warm_passes == 2


def test_warm_all_without_kv_cache_rejected():
    with pytest.raises(ValueError, match="kv_cache"):
        CustomBackend(
            block_size=8, steps=4, confidence=0.9,
            score_fn=lambda seq: None, tokenizer=RecordingTokenizer(),
            cache_warm_all=lambda ids: None)


# ---- real-model tests (CPU, fp32, weights from the local HF cache) ----

@pytest.fixture(scope="module")
def backend_chunked():
    return load_smoldlm(REPO, block_size=32, steps=4, confidence=0.9,
                        dtype="float32", kv_cache=True, warm_batch=False,
                        device="cpu")


@pytest.fixture(scope="module")
def backend_batched():
    return load_smoldlm(REPO, block_size=32, steps=4, confidence=0.9,
                        dtype="float32", kv_cache=True, warm_batch=True,
                        device="cpu")


PROMPT = "The meaning of life is " * 15


def test_batched_warm_cache_matches_chunked(backend_chunked, backend_batched):
    ids = backend_chunked.tokenizer.encode(PROMPT).ids
    assert len(ids) > 64  # exercise several complete blocks and a partial block
    backend_chunked.cache_reset()
    backend_chunked._warm(list(ids))
    backend_batched.cache_reset()
    backend_batched._warm(list(ids))
    for layer, (blk_a, blk_b) in enumerate(zip(backend_chunked.model.blocks,
                                               backend_batched.model.blocks)):
        ka, va = blk_a.attn.kv_cache
        kb, vb = blk_b.attn.kv_cache
        assert ka.shape == kb.shape and va.shape == vb.shape, layer
        # fp32 SDPA dispatches masked and unmasked inputs to different
        # backends, so exact equality is the wrong bar here; an fp64 probe
        # pins the exact same semantics (max abs diff 1.5e-14)
        torch.testing.assert_close(ka, kb, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(va, vb, rtol=1e-4, atol=1e-4)


def test_batched_warm_generate_tokens_equal_chunked(backend_chunked,
                                                    backend_batched):
    g1 = backend_chunked.generate(PROMPT, out_len=32)
    g2 = backend_batched.generate(PROMPT, out_len=32)
    assert g1.token_ids == g2.token_ids
    assert MASK not in g1.token_ids
    assert g1.blocks == g2.blocks


def test_warm_mask_cleared_and_repeat_generate_identical(backend_batched):
    g1 = backend_batched.generate(PROMPT, out_len=16)
    # no stale mask survives the warm: a later score pass must never read it
    for blk in backend_batched.model.blocks:
        assert blk.attn._warm_mask is None
    g2 = backend_batched.generate(PROMPT, out_len=16)
    # cache_reset runs at the top of every generate and greedy decode is
    # deterministic, so a polluted or unreset cache would show up here
    assert g1.token_ids == g2.token_ids


def test_cache_warm_leaves_no_autograd_graph(backend_chunked):
    with torch.enable_grad():
        g = backend_chunked.generate(PROMPT, out_len=16)
    # block-aligned decode overshoots to the full block
    assert g.num_tokens == 32
    for blk in backend_chunked.model.blocks:
        k, v = blk.attn.kv_cache
        assert k.grad_fn is None and not k.requires_grad
        assert v.grad_fn is None and not v.requires_grad


def test_warm_passes_accounting(backend_chunked, backend_batched):
    plen = len(backend_chunked.tokenizer.encode(PROMPT).ids)
    n_blocks = 1  # out_len 32 with block_size 32
    g = backend_chunked.generate(PROMPT, out_len=32)
    assert g.warm_passes == -(-plen // 32) + n_blocks
    g = backend_batched.generate(PROMPT, out_len=32)
    assert g.warm_passes == 1 + n_blocks


def test_warm_batch_without_kv_cache_rejected():
    with pytest.raises(ValueError, match="kv_cache"):
        load_smoldlm(REPO, block_size=32, steps=4, confidence=0.9,
                     dtype="float32", kv_cache=False, warm_batch=True)


def test_unknown_device_rejected():
    # rejected before any download, so the whole test is free
    with pytest.raises(ValueError, match="device must be cpu or cuda"):
        load_smoldlm(REPO, block_size=32, steps=4, confidence=0.9,
                     dtype="float32", device="tpu")


def test_cuda_device_rejected_when_unavailable(monkeypatch):
    # forced unavailable so the test means the same thing on a GPU box
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA was requested"):
        load_smoldlm(REPO, block_size=32, steps=4, confidence=0.9,
                     dtype="float32", device="cuda")


def test_warm_batch_requires_pinned_source(tmp_path, monkeypatch):
    # the mask-capable attention copy is a verbatim copy of one pinned
    # revision; if upstream rewrites modeling_smoldlm.py the copy (and with
    # it the block-causal equivalence) no longer holds, so warm_batch must
    # refuse to run instead of silently diverging
    fake = tmp_path / "modeling_smoldlm.py"
    fake.write_text("# rewritten upstream\n")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda *args, **kwargs: str(fake))
    with pytest.raises(ValueError, match="validated smoldlm source"):
        load_smoldlm(REPO, block_size=32, steps=4, confidence=0.9,
                     dtype="float32", kv_cache=True, warm_batch=True,
                     device="cpu")
