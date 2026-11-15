#!/usr/bin/env python3
"""Bounded D1 P0 timing on a verified production corpus, never a P1 launch."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_model import D1LM
from scripts.train_diffusion import (TrainConfig, load_checkpoint,
                                    load_manifest, train)


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('d1_model.py', 'train_diffusion.py', 'check_gdn_t4.py', 'check_d1_p0.py')}


def config(root, micro):
    return TrainConfig(bundle=str(root / 'full-bundle'), out=str(root / 'p0-run'),
                       profile='d1', mask_mode='blockwise', gdn_backend='chunk',
                       device='cuda', amp=True, amp_init_scale=256, batch_size=micro,
                       grad_accum=64 // micro, steps=200, warmup=4, lr=1e-3,
                       checkpoint_every=200, eval_every=0, verify_hashes=True, seed=1001)


def p0(root, result):
    probe = json.loads((root / 'probe.json').read_text())['checks']['probe']
    cfg = config(root, probe['selected_micro_batch'])
    result['config'] = asdict(cfg)
    result['warmup_updates'] = 50
    result['attempted_update_cap'] = 200
    result['wall_cap_seconds'] = 1800
    result['windows'] = []
    start = time.monotonic()
    previous = {}
    last_time = None
    skipped = 0

    def log(line):
        nonlocal previous, last_time, skipped
        print(line, flush=True)
        row = json.loads(line)
        if 'masked_ce' not in row:
            return
        if time.monotonic() - start > 1800:
            raise TimeoutError('P0 exceeded its 30-minute cap')
        if row['step'] > 50:
            skipped += int(row['optimizer_step_skipped'])
        if row['step'] in (50, 100, 150, 200):
            torch.cuda.synchronize()
            now = time.monotonic()
            if last_time is not None:
                seconds = now - last_time
                totals = {k: row[k] - previous[k] for k in
                          ('tokens_seen', 'tokens_processed', 'tokens_scored')}
                window = {'end_step': row['step'], 'attempts': 50,
                          'applied_updates': 50 - skipped, 'skipped_updates': skipped,
                          'seconds': seconds, **totals,
                          'consumed_tokens_per_second': totals['tokens_seen'] / seconds,
                          'processed_tokens_per_second': totals['tokens_processed'] / seconds,
                          'scored_tokens_per_second': totals['tokens_scored'] / seconds,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                result['windows'].append(window)
                (root / 'p0-progress.json').write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps({'p0_window': window}), flush=True)
            previous, last_time, skipped = row, now, 0
            torch.cuda.reset_peak_memory_stats()

    summary = train(cfg, log=log)
    result['summary'] = summary
    result['wall_seconds'] = time.monotonic() - start
    assert last_time is not None
    result['final_checkpoint_and_summary_seconds'] = time.monotonic() - last_time
    assert len(result['windows']) == 3, 'missing P0 timing windows'
    assert all(w['skipped_updates'] == 0 for w in result['windows']), 'AMP rejected timed updates'
    result['speed_gate'] = ('PASS' if min(w['processed_tokens_per_second']
                                          for w in result['windows']) >= 6000 else 'FAIL')
    assert result['speed_gate'] == 'PASS', 'processed-token throughput below 6000/s'


def overhead(root, result):
    p0_report = json.loads((root / 'p0.json').read_text())
    assert p0_report['status'] == 'PASS', 'P0 did not pass'
    cfg = config(root, p0_report['config']['batch_size'])
    meta = load_manifest(cfg.bundle, verify_hashes=True)
    payload = load_checkpoint(root / 'p0-run/checkpoint-000200.pt')
    model = D1LM(32000, 1024, backend='chunk').cuda().eval()
    model.load_state_dict(payload['model'])
    del payload
    gc.collect(); torch.cuda.empty_cache()
    from scripts.evaluate_d1_checkpoint import evaluate_cached

    torch.cuda.reset_peak_memory_stats()
    measured = evaluate_cached(cfg, model, 32000, log=print)
    result.update(seconds=measured['seconds'], val_masked_ce=measured['val_masked_ce'],
                  validation_sequences=meta['splits']['validation']['sequences'],
                  validation_unique_tokens=meta['splits']['validation']['tokens'],
                  frontiers_per_context=measured['frontiers_per_context'],
                  forwards=measured['score_passes'] + measured['commit_passes'],
                  processed_tokens=measured['processed_tokens'],
                  scored_tokens=measured['scored_tokens'],
                  cached_evaluator_sha256=hashlib.sha256(
                      Path(__file__).with_name('evaluate_d1_checkpoint.py').read_bytes()).hexdigest(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  scope='one cached full-frontier validation on the production split; P0 checkpoint, no quality gate claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('p0', 'overhead'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {'phase': args.phase, 'status': 'RUNNING', 'source_sha256': source_hashes()}
    code = 1
    try:
        assert (torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5)
                and torch.cuda.get_device_name() == 'Tesla T4'), 'requires a real T4'
        report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
        meta = load_manifest(args.root / 'full-bundle', verify_hashes=True)
        assert not meta.get('smoke_only') and meta['seq_len'] == 1024, 'requires the production corpus'
        report['corpus'] = {'fingerprint': meta['fingerprint'],
                            'splits': {name: {'tokens': info['tokens'], 'sequences': info['sequences']}
                                       for name, info in meta['splits'].items()}}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for phase in ('parity', 'resume', 'probe'):
            evidence = json.loads((args.root / f'{phase}.json').read_text())
            assert evidence['status'] == 'PASS', phase
            for name in ('d1_model.py', 'train_diffusion.py', 'check_gdn_t4.py'):
                assert evidence['source_sha256'][name] == report['source_sha256'][name], f'stale {phase} evidence'
        (p0 if args.phase == 'p0' else overhead)(args.root, report)
        report['status'] = 'PASS'
        code = 0
    except Exception as error:
        import traceback
        report.update(status='FAIL', error=str(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    (args.root / f'{args.phase}.json').write_text(json.dumps(report, indent=2) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
