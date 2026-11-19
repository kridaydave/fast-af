"""Dense training must reproduce independent clean-prefix frontier forwards.

The oracle deliberately uses the original full forward, not dense helpers or
training caches. No CUDA claim is made by these CPU checks.
"""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from scripts.d1_model import D1Config, D1LM
from scripts.distributed_training import DistributedTraining
from scripts.train_diffusion import (TrainConfig, all_blocks_diffusion_mask, config_hash_of,
                                     load_checkpoint, model_masked_loss, train, evaluate, bundle_vocab, load_manifest)
from tests.test_train_diffusion import make_bundle, tiny_d1


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    return D1Config(d_model=16, n_layers=4, query_heads=2, kv_heads=1, head_dim=8,
                    ffn_hidden=32, gdn_heads=2, gdn_key_dim=4, gdn_value_dim=8, block_size=4)


def independent_logits(model, clean, noisy):
    block = model.cfg.block_size
    return torch.cat([model(torch.cat((clean[:, :start], noisy[:, start:start + block]), dim=1))[:, -block:]
                      for start in range(0, clean.shape[1], block)], dim=1)


@pytest.mark.parametrize('backend', ['torch', 'chunk'])
@pytest.mark.parametrize('width', [1, 4, 7])
def test_dense_outputs_and_every_parameter_gradient_match_independent_frontiers(cfg, backend, width):
    torch.manual_seed(77)
    cfg = replace(cfg, conv_width=width)
    oracle = D1LM(64, 12, cfg, 'torch')
    dense = D1LM(64, 12, cfg, backend)
    dense.load_state_dict(oracle.state_dict())
    clean = torch.arange(5, 29).reshape(2, 12)
    noisy, mask = all_blocks_diffusion_mask(clean, 4, generator=torch.Generator().manual_seed(73))
    expected = independent_logits(oracle, clean, noisy)
    actual = dense(noisy, clean_inputs=clean)
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-6)
    expected_loss = F.cross_entropy(expected[mask], clean[mask])
    actual_loss = model_masked_loss(dense, noisy, clean, mask, clean_inputs=clean)
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_loss.backward(); actual_loss.backward()
    for (name, a), (_, b) in zip(oracle.named_parameters(), dense.named_parameters()):
        assert a.grad is not None and b.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=4e-4, atol=2e-6, msg=name)


def test_clean_current_future_and_other_noisy_blocks_cannot_leak(cfg):
    torch.manual_seed(6)
    model = D1LM(64, 12, cfg, 'chunk')
    clean = torch.arange(5, 17).reshape(1, 12)
    noisy = torch.zeros_like(clean)
    original = model(noisy, clean_inputs=clean)
    # Changing the clean answer and clean future cannot change that block.
    for start in (0, 4, 8):
        changed = clean.clone(); changed[:, start:] += 20
        torch.testing.assert_close(model(noisy, clean_inputs=changed)[:, start:start + 4],
                                   original[:, start:start + 4], rtol=0, atol=0)
    # Other noisy blocks, including earlier ones, never supply a prefix.
    changed_noisy = noisy.clone(); changed_noisy[:, :4] = 50; changed_noisy[:, 8:] = 51
    torch.testing.assert_close(model(changed_noisy, clean_inputs=clean)[:, 4:8], original[:, 4:8], rtol=0, atol=0)
    changed_past = clean.clone(); changed_past[:, :4] = 55
    assert not torch.allclose(model(noisy, clean_inputs=changed_past)[:, 4:8], original[:, 4:8])


def test_masks_cover_every_eligible_block_reproducibly_and_preserve_reserved_tokens():
    clean = torch.arange(5, 29).reshape(2, 12)
    clean[0, :4] = 1; clean[1, :4] = torch.tensor([1, 2, 3, 4])
    before = clean.clone()
    a, mask = all_blocks_diffusion_mask(clean, 4, generator=torch.Generator().manual_seed(9))
    b, other = all_blocks_diffusion_mask(clean, 4, generator=torch.Generator().manual_seed(9))
    assert torch.equal(a, b) and torch.equal(mask, other) and torch.equal(clean, before)
    assert not mask[:, :4].any()
    assert mask[:, 4:].reshape(2, 2, 4).any(-1).all()
    assert (a[mask] == 0).all() and torch.equal(a[~mask], clean[~mask])


def test_dense_local_ddp_sum_including_zero_scored_partition(cfg):
    torch.manual_seed(15)
    model = D1LM(64, 12, cfg, 'chunk')
    target = torch.arange(5, 29).reshape(2, 12)
    target[1] = 1
    noisy, mask = all_blocks_diffusion_mask(target, 4)
    d = DistributedTraining(1, 'cpu')
    d.world_size = 2  # Exercise local partition/loss math only, no distributed backend claimed.
    d.rank = 0
    left = d.loss(model, noisy, target, mask, 4, clean_inputs=target)
    d.rank = 1
    right = d.loss(model, noisy, target, mask, 4, clean_inputs=target)
    assert right.item() == 0 and right.requires_grad
    expected = F.cross_entropy(model(noisy, clean_inputs=target, logits_mask=mask), target[mask], reduction='sum')
    torch.testing.assert_close(left + right, expected)
    (left + right).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_dense_partial_batches_accumulation_and_exact_resume(tmp_path):
    root = make_bundle(tmp_path, shard_sequences=5)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'full'), profile='d1', mask_mode='blockwise',
                      d1=tiny_d1(), gdn_backend='chunk', d1_supervision='all-blocks', steps=8, warmup=1,
                      batch_size=3, grad_accum=2, lr=0.003, checkpoint_every=4, eval_every=4, eval_full=True)
    full = train(cfg, log=lambda *_: None)
    path = tmp_path / 'full' / 'checkpoint-000004.pt'
    resumed = train(replace(cfg, out=str(tmp_path / 'resumed')), resume=path, log=lambda *_: None)
    assert resumed['losses'] == full['losses'][4:]
    assert resumed['val_masked_ce'] == full['val_masked_ce'][1:]
    assert full['tokens_processed'] == 2 * full['tokens_seen']
    assert full['tokens_scored'] > full['tokens_seen'] / 3
    assert full['losses'][-1] < full['losses'][0]
    a = load_checkpoint(tmp_path / 'full' / 'checkpoint-000008.pt')
    b = load_checkpoint(tmp_path / 'resumed' / 'checkpoint-000008.pt')
    for key in ('tokens_seen', 'tokens_processed', 'tokens_scored', 'loader', 'scheduler'):
        assert a[key] == b[key]
    for key in a['model']:
        assert torch.equal(a['model'][key], b['model'][key]), key
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    with pytest.raises(ValueError, match='config hash mismatch'):
        train(replace(cfg, d1_supervision='frontier'), resume=path, log=lambda *_: None)
    assert config_hash_of(cfg) != config_hash_of(replace(cfg, d1_supervision='frontier'))
    # Keep the frozen validation distribution identical across training modes.
    vocab, length = bundle_vocab(root), load_manifest(root)['seq_len']
    model = D1LM(vocab, length, tiny_d1(), 'chunk')
    assert evaluate(cfg, model, vocab, length, batches=2) == evaluate(replace(cfg, d1_supervision='frontier'), model, vocab, length, batches=2)


def test_dense_input_guards(cfg):
    model = D1LM(64, 12, cfg, 'chunk')
    inputs = torch.ones(1, 12, dtype=torch.long)
    with pytest.raises(ValueError, match='whole-block'):
        model(inputs[:, :3], clean_inputs=inputs[:, :3])
    with pytest.raises(ValueError, match='whole-block'):
        model(inputs, clean_inputs=inputs, logits_last=4)
    with pytest.raises(ValueError, match='logits_mask'):
        model(inputs, clean_inputs=inputs, logits_mask=inputs)


def test_dense_execution_restores_determinism_settings_on_error(monkeypatch):
    # Configuration only; does not execute or simulate any CUDA work.
    from scripts.train_diffusion import d1_training_execution
    import os

    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    before = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    with pytest.raises(RuntimeError, match='abort'):
        with d1_training_execution('all-blocks', 'cuda'):
            assert torch.are_deterministic_algorithms_enabled()
            assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            raise RuntimeError('abort')
    assert torch.are_deterministic_algorithms_enabled() == before
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn


def test_interior_clean_states_and_input_gradients_match_recurrence_across_chunks():
    from scripts.d1_model import _gated_delta_chunked, gated_delta_reference

    torch.manual_seed(13)
    q, k = [F.normalize(torch.randn(2, 256, 2, 4), dim=-1).requires_grad_() for _ in range(2)]
    v = torch.randn(2, 256, 2, 5, requires_grad=True)
    # Strong decay also checks that future differences are masked before exp.
    decay = (-torch.rand(2, 256, 2) * 4).requires_grad_()
    beta = torch.rand(2, 256, 2, requires_grad=True)
    initial = torch.randn(2, 2, 4, 5, requires_grad=True)
    args = (q, k, v, decay, beta, initial)
    actual, final, entries = _gated_delta_chunked(*args, collect_states=True, state_stride=32)
    expected, states, state = [], [], initial
    for start in range(0, 256, 32):
        states.append(state)
        output, state = gated_delta_reference(q[:, start:start + 32], k[:, start:start + 32],
                                              v[:, start:start + 32], decay[:, start:start + 32],
                                              beta[:, start:start + 32], state)
        expected.append(output)
    expected = torch.cat(expected, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(final, state, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(torch.stack(entries), torch.stack(states), rtol=2e-4, atol=2e-6)
    objective = lambda output, end, starts: output.square().sum() + end.square().sum() + torch.stack(starts).square().sum()
    left = torch.autograd.grad(objective(actual, final, entries), args, retain_graph=True)
    right = torch.autograd.grad(objective(expected, state, states), args)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=2e-5)
