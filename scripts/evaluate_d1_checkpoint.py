#!/usr/bin/env python3
"""Full-frontier D1 checkpoint evaluation with an immutable clean-prefix cache."""
import argparse
from dataclasses import fields, replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_model import D1Config, D1LM
from scripts.train_diffusion import (TrainConfig, ShardLoader, block_diffusion_mask,
    bundle_provenance, bundle_vocab, config_hash_of, load_checkpoint, load_manifest,
    masked_cross_entropy, model_masked_loss)


@torch.no_grad()
def evaluate_cached(cfg, model, vocab, *, batches=None, recompute_prefix=False,
                    wall_cap=None, log=None):
    """Score the same masks as full recomputation, committing only clean blocks.

    This is evaluation only: cached prefix states must never replace training
    prefix computation or its gradients. Counts include clean commit passes.
    A wall_cap aborts with TimeoutError; it is opt-in so that a bounded job can
    ask for one without killing a training run that evaluates mid-schedule.
    """
    if cfg.profile != 'd1' or cfg.d1 != model.cfg:
        raise ValueError('requires matching D1 configuration')
    if batches is not None and (type(batches) is not int or batches < 1):
        raise ValueError('batches must be positive or None')
    device = next(model.parameters()).device
    meta = load_manifest(cfg.bundle, cfg.verify_hashes)
    length, block = meta['seq_len'], cfg.d1.block_size
    if length % block or length > model.seq_len:
        raise ValueError('invalid evaluation context')
    loader = ShardLoader(cfg.bundle, 'validation', cfg.batch_size, cfg.seed + 1,
                         cfg.verify_hashes, vocab=vocab, finite=True)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    counts = {'contexts': 0, 'batches': 0, 'scored_tokens': 0, 'processed_tokens': 0,
              'score_passes': 0, 'commit_passes': 0}
    loss_sum = 0.0
    was_training = model.training
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.monotonic()
    try:
        while batches is None or counts['batches'] < batches:
            try:
                targets = next(loader).to(device)
            except StopIteration:
                break
            cache = None
            for frontier in range(length // block):
                if wall_cap is not None and time.monotonic() - start > wall_cap:
                    raise TimeoutError('evaluation wall cap reached')
                inputs, mask = block_diffusion_mask(
                    targets, block, min_ratio=cfg.mask_min, max_ratio=cfg.mask_max,
                    generator=generator, frontier=frontier)
                scored = int(mask.sum())
                begin, end = frontier * block, (frontier + 1) * block
                with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp):
                    if scored:
                        if recompute_prefix:
                            loss = model_masked_loss(model, inputs, targets[:, :end], mask)
                            counts['processed_tokens'] += inputs.numel()
                        else:
                            logits, candidate = model.forward_block(inputs[:, begin:end], cache)
                            loss = masked_cross_entropy(logits, targets[:, begin:end], mask[:, begin:end])
                            del candidate
                            counts['processed_tokens'] += targets[:, begin:end].numel()
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError('non-finite validation loss')
                        loss_sum += float(loss) * scored
                        counts['scored_tokens'] += scored
                        counts['score_passes'] += 1
                    # Even an all-EOS block must advance the clean entry cache.
                    # Never commit the candidate produced from the noisy block.
                    if not recompute_prefix and end < length:
                        _, cache = model.forward_block(targets[:, begin:end], cache)
                        counts['processed_tokens'] += targets[:, begin:end].numel()
                        counts['commit_passes'] += 1
            counts['contexts'] += targets.shape[0]
            counts['batches'] += 1
            if log is not None and counts['batches'] % 20 == 0:
                log(json.dumps({'validation_progress': counts}))
    finally:
        model.train(was_training)
    if not counts['scored_tokens']:
        raise ValueError('empty validation loss mask')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return {**counts, 'val_masked_ce': loss_sum / counts['scored_tokens'],
            'seconds': time.monotonic() - start,
            'mode': 'recompute-prefix' if recompute_prefix else 'clean-prefix-cache',
            'frontiers_per_context': length // block}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--wall-cap', type=float, default=600.0,
                        help='seconds before this bounded job aborts; 0 disables')
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    report = {'status': 'RUNNING', 'device': args.device, 'source_sha256': {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('d1_model.py', 'train_diffusion.py', 'evaluate_d1_checkpoint.py')}}
    code = 1
    try:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        if args.amp and args.device != 'cuda':
            raise ValueError('fp16 AMP evaluation requires CUDA')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        payload = load_checkpoint(args.checkpoint)
        raw = json.loads(args.checkpoint.with_name('config.json').read_text())
        values = {f.name: raw[f.name] for f in fields(TrainConfig)}
        values['d1'] = D1Config(**values['d1'])
        original = TrainConfig(**values)
        if config_hash_of(original) != payload['config_hash']:
            raise ValueError('checkpoint config hash mismatch')
        cfg = replace(original, bundle=str(args.bundle), batch_size=args.batch_size,
                      device=args.device, amp=args.amp, verify_hashes=True)
        meta = load_manifest(args.bundle, verify_hashes=True)
        if bundle_provenance(args.bundle, meta) != payload['provenance']:
            raise ValueError('checkpoint corpus provenance mismatch')
        vocab = bundle_vocab(args.bundle)
        model = D1LM(vocab, meta['seq_len'], cfg.d1, backend=cfg.gdn_backend).to(args.device)
        model.load_state_dict(payload['model'])
        del payload
        gc.collect()
        report.update(torch=torch.__version__, cuda=torch.version.cuda, amp=args.amp,
                      corpus_fingerprint=meta['fingerprint'], checkpoint=str(args.checkpoint))
        if args.device == 'cuda':
            report.update(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        report['evaluation'] = evaluate_cached(cfg, model, vocab, batches=args.max_batches,
                                               recompute_prefix=args.reference,
                                               wall_cap=args.wall_cap or None, log=print)
        if args.device == 'cuda':
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['status'] = 'PASS'; code = 0
    except Exception as error:
        import traceback
        report.update(status='FAIL', error=str(error), traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
