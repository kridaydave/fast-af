from dataclasses import asdict, replace

import pytest
import torch
from torch.nn import functional as F

from scripts.d1_model import D1Config, D1LM, backend_info, tensorcore_mm
from scripts.distributed_training import DistributedTraining, partition_batch
from scripts.prepare_corpus import canonical, digest
from scripts.train_diffusion import TrainConfig, config_hash_of, _validate, CONFIG_HASH_EXCLUDE


def tiny_model():
    return D1LM(64, 8, D1Config(d_model=16, n_layers=4, query_heads=2, kv_heads=1, head_dim=8,
        ffn_hidden=32, gdn_heads=2, gdn_key_dim=4, gdn_value_dim=8, block_size=4), backend='chunk')


@pytest.mark.parametrize('empty_rank', [False, True])
def test_partitioned_loss_gradients_match_global_token_weighting(empty_rank):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    full = tiny_model()
    targets = torch.randint(5, 64, (4, 8))
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[0, -1] = not empty_rank
    mask[2, -2:] = True
    mask[3, -4:] = True
    inputs = targets.masked_fill(mask, 0)
    expected = F.cross_entropy(full(inputs)[mask], targets[mask])
    expected.backward()
    local_gradients = []
    local_losses = []
    for rank in (0, 1):
        local = tiny_model()
        local.load_state_dict(full.state_dict())
        context = DistributedTraining.__new__(DistributedTraining)
        context.world_size, context.rank = 2, rank
        summed = context.loss(local, inputs, targets, mask, 4)
        (summed * 2 / mask.sum()).backward()
        local_losses.append(summed.detach())
        local_gradients.append({name: p.grad.clone() for name, p in local.named_parameters()})
    torch.testing.assert_close(sum(local_losses) / mask.sum(), expected)
    for name, p in full.named_parameters():
        actual = (local_gradients[0][name] + local_gradients[1][name]) / 2
        torch.testing.assert_close(actual, p.grad, atol=1e-6, rtol=2e-5)
    # Prefix computation must remain connected despite projecting only the suffix.
    assert full.tokens.weight.grad[targets[2, 0]].abs().sum() > 0


def test_suffix_head_logits_and_gradients_match_dense_head():
    torch.manual_seed(13)
    m = tiny_model()
    x = torch.randint(5, 64, (2, 8))
    dense = m(x)[:, -4:]
    suffix = m(x, logits_last=4)
    torch.testing.assert_close(dense, suffix)
    a = torch.autograd.grad(dense.square().sum(), tuple(m.parameters()))
    b = torch.autograd.grad(suffix.square().sum(), tuple(m.parameters()))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
    for value in (0, -1, 9, True):
        with pytest.raises(ValueError, match='logits_last'):
            m(x, logits_last=value)
    with pytest.raises(ValueError, match='logits_last'):
        m(x, logits_last=4, logits_mask=torch.ones_like(x, dtype=torch.bool))


def test_backend_and_distributed_guards(monkeypatch):
    with pytest.raises(ValueError, match='CUDA operands'):
        tensorcore_mm(torch.randn(2, 3, 4), torch.randn(2, 4, 5))
    assert backend_info('chunk-tc')['state_solve'] == 'fp32'
    m = tiny_model()
    m.blocks[0].mixer.backend = 'chunk-tc'
    with pytest.raises(ValueError, match='CUDA fp16'):
        m(torch.ones(1, 8, dtype=torch.long))
    cfg = TrainConfig(bundle='unused', out='unused')
    for change in ({'world_size': 3}, {'world_size': 2}, {'fused_optimizer': True}, {'gdn_backend': 'chunk-tc'}):
        with pytest.raises(ValueError):
            _validate(replace(cfg, **change))
    monkeypatch.setenv('WORLD_SIZE', '2')
    with pytest.raises(ValueError, match='--world-size'):
        DistributedTraining(1, 'cpu')
    with pytest.raises(ValueError, match='two visible CUDA'):
        DistributedTraining(2, 'cpu')
    with pytest.raises(ValueError, match='evenly'):
        partition_batch(*(torch.ones(3, 8) for _ in range(3)), 0, 2)


def test_default_config_hash_preserves_existing_checkpoints():
    cfg = TrainConfig(bundle='fixture', out='ignored')
    legacy = {k: v for k, v in asdict(cfg).items()
              if k not in set(CONFIG_HASH_EXCLUDE) | {'world_size', 'fused_optimizer'}}
    assert config_hash_of(cfg) == digest(canonical(legacy).encode())
    assert config_hash_of(cfg) != config_hash_of(replace(cfg, world_size=2))
    assert config_hash_of(cfg) != config_hash_of(replace(cfg, fused_optimizer=True))
