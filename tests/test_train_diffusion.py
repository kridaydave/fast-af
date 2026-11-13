import json
import math
from pathlib import Path

import pytest
import torch

from scripts.build_training_mix import build
from scripts.prepare_corpus import filehash, write_manifest
from scripts.train_diffusion import (CHECKPOINT_VERSION, DiffusionLM, MASK_ID, ShardLoader,
                                     TrainConfig, build_model, build_optimizer,
                                     bundle_provenance, diffusion_mask, evaluate,
                                     load_checkpoint, load_manifest, main,
                                     masked_cross_entropy, save_checkpoint, train)
from scripts.train_tokenizer import train as train_tokenizer

SEQ = 8
VOCAB = 261
FORBIDDEN = (0, 2, 3, 4)


def make_bundle(tmp_path, seq=SEQ, train_tokens=640, validation_tokens=320, shard_sequences=16):
    data = tmp_path / 'prepared'
    data.mkdir()
    for split in ('train', 'validation'):
        rows = [{'text': f'{split} {bucket} document {i} ' + ('some ordinary Unicode λ text ' * 4),
                 'bucket': bucket, 'source': {'fixture': True}}
                for bucket in ('code', 'web', 'math') for i in range(24)]
        (data / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write_manifest(data / 'manifest.json',
                   {'schema': 'prepared-corpus/v1', 'smoke_only': False,
                    'benchmarks': {'human_eval': {'records': 164}, 'mbpp': {'records': 974}},
                    'outputs': {s: {'sha256': filehash(data / f'{s}.jsonl')}
                                for s in ('train', 'validation')}})
    tokens = tmp_path / 'tokenizer'
    train_tokenizer([data / 'train.jsonl'], tokens, vocab_size=VOCAB, prepared=data / 'manifest.json')
    out = tmp_path / 'bundle'
    build(data, tokens, out, train_tokens=train_tokens, validation_tokens=validation_tokens,
          seq_len=seq, shard_sequences=shard_sequences)
    return out


def loader(root, **kwargs):
    kwargs.setdefault('batch_size', 8)
    kwargs.setdefault('seed', 7)
    kwargs.setdefault('vocab', VOCAB)
    return ShardLoader(root, **kwargs)


def test_shard_loader_covers_epoch_in_seeded_order_and_resumes(tmp_path):
    root = make_bundle(tmp_path)
    total = sum(s['sequences'] for s in json.loads((root / 'manifest.json').read_text())
                ['splits']['train']['shards'])
    batches = loader(root, verify_hashes=True)
    seen, rows = 0, []
    while seen < total:
        batch = next(batches)
        assert batch.dtype == torch.int64 and batch.ndim == 2 and batch.shape[1] == SEQ
        assert 1 <= int(batch.min()) and int(batch.max()) < VOCAB
        assert not torch.isin(batch, torch.tensor(FORBIDDEN)).any()
        seen += batch.shape[0]
        rows.append(int(batch[0, 0]))
    assert seen == total
    repeat_loader = loader(root)
    repeat = [int(next(repeat_loader)[0, 0]) for _ in range(len(rows))]
    assert repeat == rows
    other_loader = loader(root, seed=8)
    other = [int(next(other_loader)[0, 0]) for _ in range(len(rows))]
    assert sorted(other) == sorted(rows) and other != rows

    probe = loader(root)
    for _ in range(3):
        next(probe)
    state = probe.state_dict()
    expected = next(probe)
    resumed = loader(root)
    resumed.load_state_dict(state)
    assert torch.equal(next(resumed), expected)


def test_diffusion_mask_is_bernoulli_and_excludes_eos_and_reserved():
    targets = torch.arange(1, 256 * 64 + 1).reshape(256, 64) % 240 + 5
    targets[0, :8] = 1  # EOS
    targets[1, :8] = torch.tensor([0, 2, 3, 4, 0, 2, 3, 4])  # reserved
    inputs, loss_mask = diffusion_mask(targets, min_ratio=0.1, max_ratio=0.9,
                                       generator=torch.Generator().manual_seed(11))
    assert torch.equal(inputs[~loss_mask], targets[~loss_mask])
    assert (inputs[loss_mask] == MASK_ID).all()
    assert not loss_mask[targets == 1].any()
    assert not loss_mask[torch.isin(targets, torch.tensor([0, 2, 3, 4]))].any()
    assert (loss_mask.sum(1) >= 1).all()
    ratios = loss_mask.float().mean(1)
    assert 0.40 < ratios.mean() < 0.60  # uniform [0.1, 0.9] has mean 0.5
    assert ratios.std() > 0.05  # per-sequence rates, not one global rate
    assert len(set(map(int, loss_mask.sum(1)))) > 1  # per-token Bernoulli counts vary
    again_inputs, again_mask = diffusion_mask(targets, min_ratio=0.1, max_ratio=0.9,
                                              generator=torch.Generator().manual_seed(11))
    assert torch.equal(loss_mask, again_mask) and torch.equal(inputs, again_inputs)
    with pytest.raises(ValueError, match='mask rates'):
        diffusion_mask(targets, min_ratio=0.5, max_ratio=0.5)


def test_diffusion_mask_rate_bounds_are_configurable():
    targets = torch.full((64, 256), 7, dtype=torch.long)
    inputs, loss_mask = diffusion_mask(targets, min_ratio=0.2, max_ratio=0.4,
                                       generator=torch.Generator().manual_seed(3))
    assert (inputs[loss_mask] == MASK_ID).all()
    assert 0.15 < loss_mask.float().mean() < 0.45
    assert loss_mask.sum(1).min() >= 1


def test_loss_scores_masked_positions_only():
    targets = torch.tensor([[5, 6, 7, 8]])
    loss_mask = torch.tensor([[True, False, False, False]])
    logits = torch.zeros(1, 4, 10)
    logits[0, 0, 5] = 10.0
    logits[0, 1:, 0] = 10.0
    assert masked_cross_entropy(logits, targets, loss_mask) < 1e-3
    logits[0, 0, 5] = 0.0
    logits[0, 0, 0] = 10.0
    assert masked_cross_entropy(logits, targets, loss_mask) > 1.0


def test_train_loop_reduces_loss_and_writes_checkpoints(tmp_path):
    root = make_bundle(tmp_path)
    out = tmp_path / 'run'
    cfg = TrainConfig(bundle=str(root), out=str(out), steps=200, warmup=10, checkpoint_every=100,
                      eval_every=100, batch_size=8, lr=1e-3, seed=5)
    summary = train(cfg, log=lambda *a: None)
    assert summary['step'] == 200 and summary['tokens_seen'] == 200 * 8 * SEQ
    assert summary['params'] > 0
    losses = summary['losses']
    assert all(math.isfinite(x) for x in losses)
    assert losses[-1] < losses[0] * 0.6
    assert math.isfinite(summary['val_losses'][-1])
    assert (out / 'checkpoint-000200.pt').exists()
    assert (out / 'checkpoint-last.txt').read_text() == 'checkpoint-000200.pt'
    assert not list(out.glob('*.tmp'))
    assert json.loads((out / 'config.json').read_text())['config_hash'] == summary['config_hash']


def test_resume_reproduces_uninterrupted_run(tmp_path):
    root = make_bundle(tmp_path)
    common = dict(bundle=str(root), steps=30, warmup=5, checkpoint_every=15, eval_every=15,
                  batch_size=8, lr=1e-3, seed=3)
    full = train(TrainConfig(out=str(tmp_path / 'a'), **common), log=lambda *a: None)
    ckpt = tmp_path / 'a' / 'checkpoint-000015.pt'
    assert ckpt.exists()
    resumed = train(TrainConfig(out=str(tmp_path / 'b'), **common), resume=ckpt, log=lambda *a: None)
    assert resumed['step'] == full['step'] == 30
    assert resumed['tokens_seen'] == full['tokens_seen']
    assert resumed['losses'] == full['losses'][15:]
    a = load_checkpoint(tmp_path / 'a' / 'checkpoint-000030.pt')
    b = load_checkpoint(tmp_path / 'b' / 'checkpoint-000030.pt')
    assert a['loader'] == b['loader'] and a['scheduler'] == b['scheduler']
    assert a['model'].keys() == b['model'].keys()
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert not list((tmp_path / 'b').glob('*.tmp'))


def test_resume_rejects_config_mismatch_and_finished_checkpoint(tmp_path):
    root = make_bundle(tmp_path)
    base = dict(bundle=str(root), steps=5, warmup=1, checkpoint_every=5, batch_size=8, lr=1e-3, seed=1)
    train(TrainConfig(out=str(tmp_path / 'a'), **base), log=lambda *a: None)
    ckpt = tmp_path / 'a' / 'checkpoint-000005.pt'
    changed = TrainConfig(out=str(tmp_path / 'b'), **{**base, 'lr': 2e-3})
    with pytest.raises(ValueError, match='config hash mismatch'):
        train(changed, resume=ckpt, log=lambda *a: None)
    with pytest.raises(ValueError, match='already at step'):
        train(TrainConfig(out=str(tmp_path / 'c'), **base), resume=ckpt, log=lambda *a: None)


def test_cli_smoke(tmp_path, capsys):
    root = make_bundle(tmp_path)
    main(['--bundle', str(root), '--out', str(tmp_path / 'cli'), '--steps', '5', '--warmup', '1',
          '--batch-size', '4', '--checkpoint-every', '5'])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary['step'] == 5 and len(summary['config_hash']) == 64
    assert (tmp_path / 'cli' / 'checkpoint-000005.pt').exists()


def test_cli_p1_fails_clearly(tmp_path, capsys):
    root = make_bundle(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(['--bundle', str(root), '--out', str(tmp_path / 'cli'), '--p1'])
    assert exit_info.value.code == 2
    assert 'P1 requested without prerequisites' in capsys.readouterr().err


def test_optimizer_groups_follow_recipe(tmp_path):
    root = make_bundle(tmp_path)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'run'))
    model = build_model(cfg, VOCAB, SEQ)
    optimizer = build_optimizer(model, cfg)
    assert len(optimizer.param_groups) == 2
    decay, no_decay = optimizer.param_groups
    assert decay['weight_decay'] == cfg.weight_decay
    assert no_decay['weight_decay'] == 0.0
    total = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    assert total == sum(p.numel() for p in model.parameters())
    assert all(p.ndim > 1 for p in decay['params'])
    assert all(p.ndim <= 1 or id(p) in {id(model.tokens.weight), id(model.positions)}
               for p in no_decay['params'])
    assert id(model.tokens.weight) in {id(p) for p in no_decay['params']}
    assert id(model.positions) in {id(p) for p in no_decay['params']}


def test_gradient_accumulation_consumes_global_batch(tmp_path):
    root = make_bundle(tmp_path)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'run'), steps=3, warmup=1,
                      batch_size=4, grad_accum=2, checkpoint_every=3, eval_every=0)
    summary = train(cfg, log=lambda *a: None)
    assert summary['step'] == 3
    assert summary['tokens_seen'] == 3 * 4 * 2 * SEQ
    assert all(math.isfinite(x) for x in summary['losses'])


def test_full_validation_consumes_whole_split(tmp_path):
    root = make_bundle(tmp_path)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'run'), batch_size=8, seed=5)
    model = build_model(cfg, VOCAB, SEQ)
    full = evaluate(cfg, model, VOCAB, SEQ, batches=None)
    meta = load_manifest(root)
    expected_batches = sum((s['sequences'] + cfg.batch_size - 1) // cfg.batch_size
                           for s in meta['splits']['validation']['shards'])
    one_epoch = evaluate(cfg, model, VOCAB, SEQ, batches=expected_batches)
    assert full == one_epoch
    assert math.isfinite(full)


def test_checkpoint_binds_provenance_and_detects_tamper(tmp_path):
    root = make_bundle(tmp_path)
    out = tmp_path / 'run'
    cfg = TrainConfig(bundle=str(root), out=str(out), steps=5, warmup=1,
                      checkpoint_every=5, eval_every=0)
    summary = train(cfg, log=lambda *a: None)
    ckpt = out / 'checkpoint-000005.pt'
    payload = load_checkpoint(ckpt)
    meta = load_manifest(root)
    expected = bundle_provenance(root, meta)
    assert payload['provenance'] == expected
    assert payload['format_version'] == CHECKPOINT_VERSION
    assert payload['library_versions']['torch'] == torch.__version__
    assert len(payload['provenance']['shards']) >= 2  # train + validation shards
    assert {s['split'] for s in payload['provenance']['shards']} == {'train', 'validation'}
    assert (out / 'checkpoint-000005.pt.sha256').exists()
    assert summary['config_hash'] == payload['config_hash']

    raw = ckpt.read_bytes()
    ckpt.write_bytes(b'x' + raw[1:])
    with pytest.raises(ValueError, match='checksum mismatch'):
        load_checkpoint(ckpt)
    ckpt.write_bytes(raw)
    (out / 'checkpoint-000005.pt.sha256').unlink()
    with pytest.raises(ValueError, match='missing checkpoint checksum sidecar'):
        load_checkpoint(ckpt)


def test_resume_rejects_provenance_and_cursor_tampering(tmp_path):
    root = make_bundle(tmp_path)
    out = tmp_path / 'run'
    cfg = TrainConfig(bundle=str(root), out=str(out), steps=5, warmup=1,
                      checkpoint_every=5, eval_every=0)
    train(cfg, log=lambda *a: None)
    ckpt = out / 'checkpoint-000005.pt'
    payload = load_checkpoint(ckpt)
    payload['provenance']['fingerprint'] = '0' * 64
    save_checkpoint(tmp_path / 'tampered.pt', payload)
    with pytest.raises(ValueError, match='provenance mismatch'):
        train(cfg, resume=tmp_path / 'tampered.pt', log=lambda *a: None)

    payload = load_checkpoint(ckpt)
    payload['loader']['position'] = 999
    save_checkpoint(tmp_path / 'bad-cursor.pt', payload)
    with pytest.raises(ValueError, match='position out of range'):
        train(cfg, resume=tmp_path / 'bad-cursor.pt', log=lambda *a: None)

    payload = load_checkpoint(ckpt)
    payload['tokens_seen'] = 10**9
    save_checkpoint(tmp_path / 'bad-tokens.pt', payload)
    with pytest.raises(ValueError, match='tokens_seen inconsistent'):
        train(cfg, resume=tmp_path / 'bad-tokens.pt', log=lambda *a: None)


def test_p1_refuses_without_prerequisites(tmp_path):
    root = make_bundle(tmp_path)
    base = dict(bundle=str(root), out=str(tmp_path / 'run'))
    with pytest.raises(ValueError, match='P1 requested without prerequisites'):
        train(TrainConfig(**base, p1=True), log=lambda *a: None)
    with pytest.raises(NotImplementedError, match='D1 architecture'):
        train(TrainConfig(**base, p1=True, profile='d1', device='cpu'), log=lambda *a: None)
    with pytest.raises(ValueError, match='amp requires device=cuda'):
        train(TrainConfig(**base, amp=True), log=lambda *a: None)


REAL_BUNDLE = Path('/home/kriday/Code/AI/fast-af-boi/artifacts/training-v1/final')


@pytest.mark.skipif(not REAL_BUNDLE.exists(), reason='real training-v1 bundle not on this machine')
def test_real_training_v1_bundle_loads_and_scores():
    from scripts.train_diffusion import bundle_vocab, load_manifest
    meta = load_manifest(REAL_BUNDLE)
    assert meta['schema'] == 'training-mixture/v1' and meta['seq_len'] == 1024
    vocab = bundle_vocab(REAL_BUNDLE)
    assert vocab == 32000
    shard = meta['splits']['train']['shards'][0]
    assert filehash(REAL_BUNDLE / shard['path']) == shard['sha256']
    provenance = bundle_provenance(REAL_BUNDLE, meta)
    assert provenance['fingerprint'] == meta['fingerprint']
    assert len(provenance['shards']) == 11
    batches = ShardLoader(REAL_BUNDLE, 'train', batch_size=4, seed=1, vocab=vocab)
    targets = next(batches)
    assert targets.shape == (4, 1024) and 1 <= int(targets.min()) and int(targets.max()) < vocab
    inputs, loss_mask = diffusion_mask(targets, min_ratio=0.1)
    assert not loss_mask[targets == 1].any()
    assert loss_mask.any()
    loss = masked_cross_entropy(DiffusionLM(vocab, 1024)(inputs), targets, loss_mask)
    assert torch.isfinite(loss) and loss > 0
