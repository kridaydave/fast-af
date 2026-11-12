"""Real-model smoke test for the smoldlm wiring (CPU, downloads weights
once into the HF cache). This is the test that proves score_fn, the
tokenizer, and generate() actually work together -- no fakes here.
"""
import torch

from bench.backends import load_smoldlm

REPO = "HoangHa/smoldlm-144m"


def _backend(dtype="float32", kv_cache=False):
    return load_smoldlm(REPO, block_size=32, steps=4, confidence=0.9,
                        dtype=dtype, kv_cache=kv_cache)


def test_score_fn_returns_pinned_logits():
    backend = _backend()
    # device-aware: on Colab the model lives on the GPU, so the probe
    # tensor must match or the forward pass itself fails
    seq = torch.tensor([[5, 6, 0, 0, 0]], device=backend.device)
    logits = backend.score_fn(seq)
    assert logits.shape == (1, 5, 49152)  # vocab per the repo's config.json
    # mask (0) and pad (2) can never win argmax -- the loop's no-mask
    # guarantee depends on this pinning
    assert torch.isneginf(logits[0, :, 0]).all()
    assert torch.isneginf(logits[0, :, 2]).all()
    assert torch.isfinite(logits[0, :, 5]).all()


def test_generate_produces_generation():
    # fp32 here; fp16 is the T4 tier and also runs on CPU, so both paths
    # get exercised locally
    for dtype in ("float32", "float16"):
        backend = _backend(dtype)
        g = backend.generate("The meaning of life is", 32)
        assert isinstance(g.text, str), dtype
        assert g.num_tokens >= 1, dtype
        assert g.decode_seconds > 0, dtype
        assert backend.device in ("cpu", "cuda")


def test_cached_generate_uses_real_model_hooks():
    backend = _backend(kv_cache=True)
    g = backend.generate("The meaning of life is", 8)
    assert isinstance(g.text, str)
    assert g.num_tokens >= 1
    assert g.passes >= 1
    assert g.warm_passes >= 1  # at least the cached prefix
    assert 0 not in (g.token_ids or [])


def test_roster_path_builds_and_generates():
    # the exact dict shape bench/config.yaml feeds _build_backend on Colab
    from bench.run import _build_backend
    m = {"backend": "custom", "repo": REPO, "block_size": 32, "steps": 4,
         "confidence": 0.9}
    backend = _build_backend(m, {"dtype": "float32"})
    g = backend.generate("The meaning of life is", 8)
    assert g.num_tokens >= 1
    assert isinstance(g.text, str)
