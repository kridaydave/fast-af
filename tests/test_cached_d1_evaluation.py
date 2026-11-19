import json

import numpy as np
import pytest
import torch

from scripts.d1_model import D1Config, D1LM
from scripts.evaluate_d1_checkpoint import evaluate_cached
from scripts.prepare_corpus import filehash, write_manifest
from scripts.train_diffusion import SPECIAL_IDS, TrainConfig, evaluate


@pytest.mark.parametrize('backend', ['torch', 'chunk'])
def test_cached_full_frontier_loss_matches_recomputation_with_eos_and_partial_batch(tmp_path, backend):
    torch.set_num_threads(1); torch.manual_seed(101)
    cfg_model = D1Config(d_model=16, n_layers=4, query_heads=2, kv_heads=1, head_dim=8,
                        ffn_hidden=32, gdn_heads=2, gdn_key_dim=4, gdn_value_dim=8, block_size=4)
    data = np.arange(5, 41, dtype='<u2').reshape(3, 12)
    data[:, 4:8] = 1  # An unscored frontier must still enter the clean cache.
    path = tmp_path / 'validation.bin'; data.tofile(path)
    info = {'tokens': 36, 'sequences': 3,
            'shards': [{'path': path.name, 'tokens': 36, 'sequences': 3, 'sha256': filehash(path)}]}
    write_manifest(tmp_path / 'manifest.json', {'schema': 'training-mixture/v1',
                   'dtype': '<u2', 'seq_len': 12, 'special_ids': SPECIAL_IDS,
                   'splits': {'train': info, 'validation': info}})
    cfg = TrainConfig(bundle=str(tmp_path), out='unused', profile='d1', mask_mode='blockwise',
                      d1=cfg_model, gdn_backend=backend, batch_size=2, seed=9)
    model = D1LM(64, 12, cfg_model, backend).train()
    rng = torch.get_rng_state().clone()
    legacy = evaluate(cfg, model, 64, 12, batches=None)
    reference = evaluate_cached(cfg, model, 64, recompute_prefix=True)
    cached = evaluate_cached(cfg, model, 64)
    assert cached['val_masked_ce'] == pytest.approx(legacy, abs=1e-6)
    assert cached['val_masked_ce'] == pytest.approx(reference['val_masked_ce'], abs=1e-6)
    assert cached['scored_tokens'] == reference['scored_tokens']
    assert cached['contexts'] == 3 and cached['batches'] == 2
    assert cached['commit_passes'] == 4 and cached['score_passes'] == 4
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(TimeoutError):
        evaluate_cached(cfg, model, 64, wall_cap=-1)
    assert model.training


@pytest.mark.skipif(torch.cuda.is_available(), reason='CPU failure-mode check')
def test_checkpoint_evaluator_refuses_missing_cuda(tmp_path, monkeypatch):
    from scripts.evaluate_d1_checkpoint import main

    out = tmp_path / 'result.json'
    monkeypatch.setattr('sys.argv', ['evaluate_d1_checkpoint.py', '--checkpoint', str(tmp_path / 'x.pt'),
                                    '--bundle', str(tmp_path), '--out', str(out)])
    assert main() == 1
    report = json.loads(out.read_text())
    assert report['status'] == 'FAIL' and report['error'] == 'CUDA unavailable'
