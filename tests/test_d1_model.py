from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from scripts.d1_model import (AttentionState, BlockAttention, D1Config, D1LM,
                              GatedDeltaNet, GDNState, gated_delta_reference)
from scripts.train_diffusion import block_diffusion_mask


@pytest.fixture
def tiny():
    torch.set_num_threads(1)
    return D1Config(d_model=16, n_layers=4, query_heads=2, kv_heads=1, head_dim=8,
                    ffn_hidden=32, gdn_heads=2, gdn_key_dim=4, gdn_value_dim=8,
                    block_size=4)


def test_delta_rule_matches_affine_state_transition_and_gradients():
    torch.manual_seed(4)
    q = torch.randn(1, 3, 2, 3, requires_grad=True)
    k = F.normalize(torch.randn_like(q), dim=-1).detach().requires_grad_()
    v = torch.randn(1, 3, 2, 4, requires_grad=True)
    g = torch.full((1, 3, 2), -0.2, requires_grad=True)
    beta = torch.full((1, 3, 2), 0.4, requires_grad=True)
    initial = torch.randn(1, 2, 3, 4, requires_grad=True)
    actual, state = gated_delta_reference(q, k, v, g, beta, initial)
    before = initial.detach().clone()
    # Independent affine matrix formulation of the transition.
    expected, h = [], initial
    for t in range(3):
        key = k[:, t].unsqueeze(-1)
        transition = g[:, t].exp()[..., None, None] * (
            torch.eye(3) - beta[:, t, :, None, None] * (key @ key.transpose(-1, -2)))
        h = transition @ h + beta[:, t, :, None, None] * (key @ v[:, t, :, None, :])
        expected.append((q[:, t, :, None, :] @ h).squeeze(-2) / 3 ** 0.5)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, h)
    args = (q, k, v, g, beta, initial)
    actual_grads = torch.autograd.grad(actual.square().sum() + state.square().sum(), args, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum() + h.square().sum(), args)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
    assert torch.equal(initial, before)


def clone_cache(cache):
    result = []
    for state in cache.layers:
        tensors = (state.keys, state.values) if isinstance(state, AttentionState) else (state.recurrent, *state.conv)
        result.extend(t.detach().clone() for t in tensors)
    return result


def test_full_and_cached_forward_match_without_mutating_prefix(tiny):
    torch.manual_seed(3)
    model = D1LM(64, 12, tiny).eval()
    tokens = torch.arange(5, 17).reshape(1, 12)
    with torch.no_grad():
        full = model(tokens)
        cache = model.prefill(tokens[:, :4])
        before = clone_cache(cache)
        # A discarded denoising candidate must not alter clean prefix state.
        model.forward_block(torch.zeros_like(tokens[:, 4:8]), cache)
        candidate, committed = model.forward_block(tokens[:, 4:8], cache)
        repeat, _ = model.forward_block(tokens[:, 4:8], cache)
        torch.testing.assert_close(candidate, full[:, 4:8], rtol=1e-5, atol=1e-6)
        assert torch.equal(candidate, repeat)
        for old, new in zip(before, clone_cache(cache)):
            assert torch.equal(old, new)
        last, _ = model.forward_block(tokens[:, 8:], committed)
        torch.testing.assert_close(last, full[:, 8:], rtol=1e-5, atol=1e-6)
        assert committed.tokens == 8
        for state in committed.layers:
            if isinstance(state, GDNState):
                assert state.recurrent.shape == (1, 2, 4, 8)
                assert all(t.shape[-1] == tiny.conv_width - 1 for t in state.conv)
            else:
                assert state.keys.shape == (1, 1, 8, 8)


def test_future_blocks_cannot_leak_but_active_block_is_bidirectional(tiny):
    torch.manual_seed(31)
    model = D1LM(64, 12, tiny)
    tokens = torch.arange(5, 17).reshape(1, 12)
    original = model(tokens)
    future = tokens.clone()
    future[:, 4:] += 16
    torch.testing.assert_close(model(future)[:, :4], original[:, :4], rtol=0, atol=0)
    right_context = tokens.clone()
    right_context[:, 3] = 40
    assert not torch.allclose(model(right_context)[:, 0], original[:, 0], atol=1e-6)


def test_cached_gradients_match_full_forward(tiny):
    torch.manual_seed(12)
    full = D1LM(64, 8, tiny)
    cached = D1LM(64, 8, tiny)
    cached.load_state_dict(full.state_dict())
    inputs = torch.arange(5, 13).reshape(1, 8)
    full(inputs)[:, 4:].square().mean().backward()
    cache = cached.prefill(inputs[:, :4])
    logits, _ = cached.forward_block(inputs[:, 4:], cache)
    logits.square().mean().backward()
    for (name, left), (_, right) in zip(full.named_parameters(), cached.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_frontier_masks_only_active_tokens_and_removes_future():
    targets = torch.arange(5, 29).reshape(2, 12)
    targets[0, 5] = 1
    original = targets.clone()
    inputs, mask = block_diffusion_mask(targets, 4, frontier=1,
                                        generator=torch.Generator().manual_seed(4))
    assert inputs.shape == mask.shape == (2, 8)
    assert torch.equal(inputs[:, :4], targets[:, :4])
    assert not mask[:, :4].any() and not mask[0, 5]
    assert mask[:, 4:].any(dim=1).all()
    assert (inputs[mask] == 0).all()
    assert torch.equal(targets, original)
    later = targets.clone()
    later[:, 8:] = 55
    other, other_mask = block_diffusion_mask(later, 4, frontier=1,
                                            generator=torch.Generator().manual_seed(4))
    assert torch.equal(inputs, other) and torch.equal(mask, other_mask)


def test_frontier_sampling_is_seeded_and_covers_eligible_blocks():
    targets = torch.full((3, 12), 7)
    targets[:, :4] = 1  # all-EOS block is ineligible
    def sample():
        generator = torch.Generator().manual_seed(5)
        return [block_diffusion_mask(targets, 4, generator=generator)[0].shape[1] for _ in range(30)]
    lengths = sample()
    assert lengths == sample() and set(lengths) == {8, 12}
    with pytest.raises(ValueError, match='no eligible'):
        block_diffusion_mask(torch.ones(2, 8, dtype=torch.long), 4)


@pytest.mark.parametrize('prompt_length,new_tokens', [(0, 7), (3, 6), (4, 4), (5, 0)])
def test_generation_preserves_prompt_and_commits_clean_blocks(tiny, prompt_length, new_tokens):
    torch.manual_seed(8)
    model = D1LM(64, 16, tiny)
    prefix = torch.arange(5, 5 + prompt_length).reshape(1, prompt_length)
    result = model.generate(prefix, new_tokens, steps=3)
    output = result['tokens']
    assert output.shape == (1, prompt_length + new_tokens)
    assert torch.equal(output[:, :prompt_length], prefix)
    assert not torch.isin(output[:, prompt_length:], torch.tensor([0, 2, 3, 4])).any()
    assert model.training  # restore caller's mode
    assert result['passes'] == result['prefill_passes'] + result['denoise_passes'] + result['commit_passes']
    if new_tokens:
        last_block_start = (output.shape[1] - 1) // tiny.block_size * tiny.block_size
        with torch.no_grad():
            expected_cache = model.prefill(output[:, :last_block_start])
            _, expected_cache = model.forward_block(output[:, last_block_start:], expected_cache)
        assert result['cache'].tokens == expected_cache.tokens
        for a, b in zip(clone_cache(result['cache']), clone_cache(expected_cache)):
            torch.testing.assert_close(a, b)


def test_frozen_architecture_contains_real_gdn_and_gqa_layers():
    with torch.device('meta'):
        model = D1LM(32000, 1024)
    assert len(model.blocks) == 16
    assert [i + 1 for i, block in enumerate(model.blocks) if isinstance(block.mixer, BlockAttention)] == [4, 8, 12, 16]
    assert sum(isinstance(block.mixer, GatedDeltaNet) for block in model.blocks) == 12
    assert model.tokens.weight.shape == (32000, 640)
    assert model.blocks[3].mixer.q_proj.weight.shape == (640, 640)
    assert model.blocks[3].mixer.k_proj.weight.shape == (128, 640)
    assert model.blocks[0].mixer.v_proj.weight.shape == (1024, 640)
    assert not any('positions' in name for name, _ in model.named_parameters())


@pytest.mark.parametrize('changes', [{'query_heads': 5, 'head_dim': 128}, {'conv_width': 0},
                                     {'n_layers': 15}, {'d_model': True}])
def test_invalid_architecture_fails_before_allocating(changes):
    with pytest.raises(ValueError):
        D1Config(**{**asdict(D1Config()), **changes})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/FLA validation needs a real GPU')
def test_fla_cuda_outputs_states_and_gradients_match_torch():
    pytest.importorskip('fla', reason='FLA is not installed on this GPU host')
    torch.manual_seed(2)
    cfg = D1Config()
    reference = GatedDeltaNet(cfg, 'torch').cuda()
    accelerated = GatedDeltaNet(cfg, 'fla').cuda()
    accelerated.load_state_dict(reference.state_dict())
    x = torch.randn(1, 64, 640, device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        left, ls = reference(x, use_cache=True)
        right, rs = accelerated(x, use_cache=True)
    torch.testing.assert_close(left, right, rtol=0.03, atol=0.003)
    torch.testing.assert_close(ls.recurrent, rs.recurrent, rtol=0.03, atol=0.003)
    for expected, actual in ((left, right), (ls.recurrent, rs.recurrent)):
        error = torch.linalg.vector_norm(expected.float() - actual.float())
        scale = torch.linalg.vector_norm(expected.float())
        assert error <= 0.03 * scale + 1e-6
    (left.float().square().sum() + ls.recurrent.square().sum()).backward()
    (right.float().square().sum() + rs.recurrent.square().sum()).backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), accelerated.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all(), name
        # Aggregate relative error rejects all-zero gradients even when each
        # individual entry is small. This runs only on actual CUDA + FLA.
        error = torch.linalg.vector_norm(a.grad.float() - b.grad.float())
        scale = torch.linalg.vector_norm(a.grad.float())
        assert error <= 0.05 * scale + 1e-7, name


@pytest.mark.skipif(torch.cuda.is_available(), reason='only checks the missing-CUDA error path')
def test_gpu_validator_reports_skip_without_substituting_cpu(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    report = tmp_path / 'gpu.json'
    result = subprocess.run([sys.executable, '-m', 'scripts.validate_d1', '--out', str(report)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    data = json.loads(report.read_text())
    assert data['status'] == 'SKIP' and data['device'] == 'cuda'
    assert not data['checks'] and 'params' not in data


def test_invalid_backend_does_not_fall_back_to_reference(tiny):
    from scripts.d1_model import backend_info

    for construct in (lambda: D1LM(64, 8, tiny, backend='typo'),
                      lambda: GatedDeltaNet(tiny, 'typo'), lambda: backend_info('typo')):
        with pytest.raises(ValueError, match='backend must be torch or fla'):
            construct()


def test_t4_policy_rejection_precedes_kernel_import_or_allocation(monkeypatch):
    # Simulates a hardware *error path* only; executes no CUDA model or kernel.
    import argparse
    from scripts.validate_d1 import validate

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (7, 5))
    report, status = validate(argparse.Namespace(device='cuda', config=None, vocab_size=32000,
                                               forward_only=False))
    assert status == 1 and report['status'] == 'FAIL'
    assert 'T4/SM75 is unsupported' in report['error']
    assert not report['checks'] and 'params' not in report
