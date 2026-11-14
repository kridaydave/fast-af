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
    assert math.isfinite(summary['val_masked_ce'][-1])
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
    with pytest.raises(ValueError, match='GPU cache/gradient/resume validation'):
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


@pytest.mark.parametrize('tokens', [0, 1, -8])
def test_rejects_invalid_shard_dimensions(tmp_path, tokens):
    root = make_bundle(tmp_path)
    path = root / 'manifest.json'
    meta = json.loads(path.read_text())
    meta.pop('fingerprint')
    meta['splits']['train']['shards'][0]['tokens'] = tokens
    write_manifest(path, meta)
    with pytest.raises(ValueError, match='whole sequences'):
        load_manifest(root)


@pytest.mark.parametrize('seq_len', [0, -8, True, 1.5])
def test_rejects_invalid_sequence_length(tmp_path, seq_len):
    root = make_bundle(tmp_path)
    path = root / 'manifest.json'
    meta = json.loads(path.read_text())
    meta.pop('fingerprint')
    meta['seq_len'] = seq_len
    write_manifest(path, meta)
    with pytest.raises(ValueError, match='seq_len must be a positive integer'):
        load_manifest(root)


def test_manifest_checks_counts_and_missing_tokenizer(tmp_path):
    root = make_bundle(tmp_path)
    path = root / 'manifest.json'
    original = json.loads(path.read_text())
    for field in ('sequences', 'tokens'):
        meta = json.loads(json.dumps(original))
        meta.pop('fingerprint')
        meta['splits']['train'][field] += 1
        write_manifest(path, meta)
        with pytest.raises(ValueError, match='totals mismatch'):
            load_manifest(root)
    path.write_text(json.dumps(original))
    (root / 'tokenizer.json').unlink()
    with pytest.raises(ValueError, match='missing tokenizer model'):
        bundle_provenance(root, load_manifest(root))


def test_resume_with_partial_batches_and_accumulation(tmp_path):
    from dataclasses import replace

    root = make_bundle(tmp_path)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'full'), steps=15,
                      warmup=2, batch_size=6, grad_accum=3, checkpoint_every=5,
                      eval_every=5, eval_full=True)
    full = train(cfg, log=lambda *a: None)
    ckpt = tmp_path / 'full' / 'checkpoint-000005.pt'
    resumed = train(replace(cfg, out=str(tmp_path / 'resumed')), resume=ckpt, log=lambda *a: None)
    assert resumed['losses'] == full['losses'][5:]
    a = load_checkpoint(tmp_path / 'full' / 'checkpoint-000015.pt')
    b = load_checkpoint(tmp_path / 'resumed' / 'checkpoint-000015.pt')
    assert a['loader'] == b['loader'] and a['scheduler'] == b['scheduler']
    assert a['tokens_seen'] == b['tokens_seen'] == 3 * 640
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])

    for field, value, message in [('tokens_seen', a['tokens_seen'] - SEQ, 'tokens_seen inconsistent'),
                                  ('step', 14, 'cursor inconsistent')]:
        damaged = dict(a, **{field: value})
        save_checkpoint(tmp_path / 'damaged.pt', damaged)
        with pytest.raises(ValueError, match=message):
            train(cfg, resume=tmp_path / 'damaged.pt', log=lambda *a: None)
    damaged = load_checkpoint(ckpt)
    damaged['loader']['sequence'] = 1
    damaged['loader']['position'] = 0
    save_checkpoint(tmp_path / 'damaged.pt', damaged)
    with pytest.raises(ValueError, match='batch boundary'):
        train(cfg, resume=tmp_path / 'damaged.pt', log=lambda *a: None)


@pytest.mark.parametrize('failure', ['loss', 'gradient'])
def test_nonfinite_training_aborts_before_updating_weights(tmp_path, monkeypatch, failure):
    import scripts.train_diffusion as training

    root = make_bundle(tmp_path)
    model = DiffusionLM(VOCAB, SEQ)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    if failure == 'loss':
        model.register_forward_hook(lambda module, args, output: output * float('nan'))
    else:
        model.tokens.weight.register_hook(lambda grad: grad * float('inf'))
    monkeypatch.setattr(training, 'build_model', lambda *args: model)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'run'), steps=2, warmup=0)
    with pytest.raises((FloatingPointError, RuntimeError), match='non-finite'):
        train(cfg, log=lambda *a: None)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in initial.items())
    assert not list((tmp_path / 'run').glob('checkpoint-*'))


def test_cpu_grad_scaler_overflow_skips_update_and_recovers():
    # Exercises the real scaler on CPU; this is not CUDA/fp16 validation.
    from scripts.train_diffusion import optimizer_step

    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=8.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    initial = model.weight.detach().clone()
    for value in (float('inf'), 1e30):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.ones(1, 2)).sum()).backward()
        model.weight.grad.fill_(value)
        old_scale = scaler.get_scale()
        norm, skipped = optimizer_step(model, optimizer, scheduler, scaler, 1.0)
        assert skipped and not math.isfinite(norm)
        assert scaler.get_scale() == old_scale * scaler.get_backoff_factor()
        assert torch.equal(model.weight, initial)
        assert not optimizer.state
        assert scheduler.last_epoch == 0
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    norm, skipped = optimizer_step(model, optimizer, scheduler, scaler, 1.0)
    assert not skipped and math.isfinite(norm)
    assert scheduler.last_epoch == 1
    assert not torch.equal(model.weight, initial)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_schedule_logs_lr_used_and_reaches_minimum(tmp_path):
    root = make_bundle(tmp_path)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'run'), steps=4, warmup=1,
                      eval_every=0, checkpoint_every=4)
    records = []
    train(cfg, log=lambda line: records.append(json.loads(line)))
    records = [row for row in records if 'step' in row]
    assert records[0]['lr'] == cfg.lr
    assert records[-1]['lr'] == pytest.approx(cfg.lr * 0.1)
    assert not any(r['optimizer_step_skipped'] for r in records)


@pytest.mark.parametrize('name', ['lr', 'weight_decay', 'eps', 'grad_clip'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_rejects_nonfinite_hyperparameters(tmp_path, name, value):
    cfg = TrainConfig(bundle='unused', out=str(tmp_path), **{name: value})
    with pytest.raises(ValueError, match=f'{name} must be finite'):
        train(cfg)


@pytest.mark.parametrize('manifest_name', ['manifest.json', 'tokenizer_manifest.json'])
def test_rejects_incompatible_special_token_ids(tmp_path, manifest_name):
    from scripts.train_diffusion import bundle_vocab

    root = make_bundle(tmp_path)
    path = root / manifest_name
    meta = json.loads(path.read_text())
    meta.pop('fingerprint')
    meta['special_ids']['[EOS]'] = 7
    write_manifest(path, meta)
    with pytest.raises(ValueError, match='special token IDs'):
        (load_manifest if manifest_name == 'manifest.json' else bundle_vocab)(root)


def test_accumulation_matches_one_token_weighted_partial_global_batch(tmp_path):
    root = make_bundle(tmp_path, shard_sequences=4)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'run'), steps=1, warmup=0,
                      batch_size=3, grad_accum=2, d_model=16, n_layers=1, n_heads=2,
                      eval_every=0, seed=23)
    torch.manual_seed(cfg.seed)
    reference = build_model(cfg, VOCAB, SEQ)
    optimizer = build_optimizer(reference, cfg)
    batches = ShardLoader(root, batch_size=cfg.batch_size, seed=cfg.seed, vocab=VOCAB)
    inputs, targets, masks = [], [], []
    for _ in range(cfg.grad_accum):
        target = next(batches)
        masked_input, mask = diffusion_mask(target)
        inputs.append(masked_input)
        targets.append(target)
        masks.append(mask)
    assert [len(t) for t in targets] == [3, 1]
    loss = masked_cross_entropy(reference(torch.cat(inputs)), torch.cat(targets), torch.cat(masks))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    train(cfg, log=lambda *a: None)
    actual = load_checkpoint(tmp_path / 'run' / 'checkpoint-000001.pt')
    assert actual['tokens_seen'] == 4 * SEQ
    for name, param in reference.state_dict().items():
        torch.testing.assert_close(actual['model'][name], param, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('steps,warmup', [(1, 0), (4, 0), (4, 3)])
def test_short_schedule_endpoints(steps, warmup):
    from scripts.train_diffusion import lr_at

    cfg = TrainConfig(bundle='unused', out='unused', steps=steps, warmup=warmup)
    assert lr_at(steps - 1, cfg) == pytest.approx(cfg.lr * (1.0 if steps == 1 else 0.1))


def tiny_d1():
    from scripts.d1_model import D1Config
    return D1Config(d_model=16, n_layers=4, query_heads=2, kv_heads=1, head_dim=8,
                    ffn_hidden=32, gdn_heads=2, gdn_key_dim=4, gdn_value_dim=8, block_size=4)


def test_d1_training_resume_restores_frontier_rng_and_token_counters(tmp_path):
    from dataclasses import replace

    root = make_bundle(tmp_path, shard_sequences=5)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'full'), profile='d1',
                      mask_mode='blockwise', d1=tiny_d1(), steps=20, warmup=2,
                      batch_size=3, grad_accum=3, lr=0.003, checkpoint_every=10,
                      eval_every=10, eval_full=True)
    full = train(cfg, log=lambda *args: None)
    ckpt = tmp_path / 'full' / 'checkpoint-000010.pt'
    resumed = train(replace(cfg, out=str(tmp_path / 'resumed')), resume=ckpt, log=lambda *a: None)
    assert full['metric'] == 'masked_ce'
    assert resumed['losses'] == full['losses'][10:]
    assert resumed['val_masked_ce'] == full['val_masked_ce'][1:]
    assert all(math.isfinite(value) for value in full['losses'])
    assert full['losses'][-1] < full['losses'][0]
    for name in ('step', 'tokens_seen', 'tokens_processed', 'tokens_scored'):
        assert resumed[name] == full[name]
    assert 0 < full['tokens_scored'] < full['tokens_processed'] < full['tokens_seen']
    a = load_checkpoint(tmp_path / 'full' / 'checkpoint-000020.pt')
    b = load_checkpoint(tmp_path / 'resumed' / 'checkpoint-000020.pt')
    for name in a['model']:
        assert torch.equal(a['model'][name], b['model'][name]), name
    assert a['loader'] == b['loader'] and a['scheduler'] == b['scheduler']
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    payload = load_checkpoint(ckpt)
    payload['tokens_processed'] = payload['tokens_seen'] + 1
    save_checkpoint(tmp_path / 'bad-counts.pt', payload)
    with pytest.raises(ValueError, match='processed/scored'):
        train(cfg, resume=tmp_path / 'bad-counts.pt', log=lambda *a: None)
    with pytest.raises(ValueError, match='config hash mismatch'):
        train(replace(cfg, d1=replace(cfg.d1, conv_width=3)), resume=ckpt, log=lambda *a: None)


def test_d1_full_validation_covers_frontiers_and_keeps_training_rng(tmp_path, monkeypatch):
    import scripts.train_diffusion as training

    root = make_bundle(tmp_path)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'run'), profile='d1',
                      mask_mode='blockwise', d1=tiny_d1(), batch_size=8)
    model = build_model(cfg, VOCAB, SEQ).eval()
    original = training.block_diffusion_mask
    frontiers = []
    def track(*args, **kwargs):
        frontiers.append(kwargs.get('frontier'))
        return original(*args, **kwargs)
    monkeypatch.setattr(training, 'block_diffusion_mask', track)
    before = torch.get_rng_state().clone()
    first = evaluate(cfg, model, VOCAB, SEQ, batches=None)
    assert frontiers == [0, 1] * 5
    assert torch.equal(torch.get_rng_state(), before)
    assert not model.training
    assert first == evaluate(cfg, model, VOCAB, SEQ, batches=None)


def test_d1_cli_loads_explicit_architecture_config(tmp_path, capsys):
    from dataclasses import asdict

    root = make_bundle(tmp_path)
    config = tmp_path / 'd1.json'
    config.write_text(json.dumps(asdict(tiny_d1())))
    capsys.readouterr()
    main(['--bundle', str(root), '--out', str(tmp_path / 'cli'), '--profile', 'd1',
          '--d1-config', str(config), '--steps', '2', '--warmup', '0', '--eval-every', '1'])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]['model'] == 'd1' and records[0]['params'] > 0
    assert records[-1]['metric'] == 'masked_ce'
    assert records[-1]['step'] == 2
    assert all('val_masked_ce' in row for row in records[1:-1])
    saved = json.loads((tmp_path / 'cli' / 'config.json').read_text())
    assert saved['d1'] == asdict(tiny_d1())
    assert saved['mask_mode'] == 'blockwise'


def test_mismatched_model_and_mask_mode_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='D1 requires blockwise'):
        train(TrainConfig(bundle='unused', out=str(tmp_path), profile='d1'))
    with pytest.raises(ValueError, match='smoke requires classical'):
        train(TrainConfig(bundle='unused', out=str(tmp_path), mask_mode='blockwise'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='D1 CUDA AMP resume needs real GPU hardware')
def test_d1_cuda_amp_checkpoint_resume(tmp_path):
    from dataclasses import replace
    from scripts.d1_model import D1Config

    pytest.importorskip('fla', reason='install pinned requirements-d1.txt on the GPU host')
    root = make_bundle(tmp_path, seq=64, train_tokens=1280, validation_tokens=1280)
    architecture = D1Config(d_model=32, n_layers=4, query_heads=4, kv_heads=2, head_dim=8,
                            ffn_hidden=64, gdn_heads=2, gdn_key_dim=16, gdn_value_dim=32)
    cfg = TrainConfig(bundle=str(root), out=str(tmp_path / 'full'), profile='d1',
                      mask_mode='blockwise', d1=architecture, device='cuda', amp=True,
                      gdn_backend='fla', steps=6, warmup=1, checkpoint_every=3,
                      eval_every=3, eval_batches=1, batch_size=2, grad_accum=2)
    full = train(cfg, log=lambda *a: None)
    ckpt = tmp_path / 'full' / 'checkpoint-000003.pt'
    resumed = train(replace(cfg, out=str(tmp_path / 'resumed')), resume=ckpt, log=lambda *a: None)
    assert resumed['losses'] == pytest.approx(full['losses'][3:], rel=1e-3, abs=1e-3)
    for key in ('tokens_seen', 'tokens_processed', 'tokens_scored'):
        assert resumed[key] == full[key]
    a = load_checkpoint(tmp_path / 'full' / 'checkpoint-000006.pt')
    b = load_checkpoint(tmp_path / 'resumed' / 'checkpoint-000006.pt')
    assert a['loader'] == b['loader'] and a['scheduler'] == b['scheduler']
    assert a['rng']['cuda'] is not None and b['rng']['cuda'] is not None
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=1e-3, atol=5e-4, msg=name)
