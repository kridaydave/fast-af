#!/usr/bin/env python3
"""Bounded 109M D1 throughput probe, on one or two T4s. Never pretraining.

Use torchrun --standalone --nproc-per-node=2 ... --world-size 2. Batch size is
GLOBAL per micro-batch, so --micro 8 compares identical work across 1/2 GPUs.
No checkpoints or validation in timing windows; report both token accounting and
GPU-seconds. Maximum 32 updates, 600 seconds, with mandatory real T4 hardware.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.distributed_training import DistributedTraining
from scripts.train_diffusion import (TrainConfig, ShardLoader, block_diffusion_mask, all_blocks_diffusion_mask,
    build_model, build_optimizer, model_masked_loss, optimizer_step, load_manifest, bundle_vocab, bundle_provenance,
    d1_training_execution)


def run(bundle, output, world_size=1, micro=8, backend='chunk', fused=False, mode='uniform',
        warmup=4, window_updates=4, windows=3, wall_cap=600, supervision='frontier'):
    if (world_size not in (1, 2) or micro not in (1, 2, 4, 8, 16) or 64 % micro or micro % world_size
            or not 1 <= warmup <= 4 or not 1 <= windows <= 3 or not 1 <= window_updates <= 8
            or warmup + windows * window_updates > 32 or not 0 < wall_cap <= 600
            or mode not in ('uniform', 'full') or supervision not in ('frontier', 'all-blocks')
            or supervision == 'all-blocks' and mode != 'uniform'):
        raise ValueError('invalid bounded probe settings')
    with d1_training_execution(supervision, 'cuda'):
        d = DistributedTraining(world_size, 'cuda')
        try:
            return measure(d, bundle, output, micro, backend, fused, mode, warmup,
                           window_updates, windows, wall_cap, supervision)
        finally:
            d.close()


def measure(d, bundle, output, micro, backend, fused, mode, warmup, window_updates, windows, wall_cap, supervision):
    device = d.device
    manifest = load_manifest(bundle, verify_hashes=True)
    if manifest['seq_len'] != 1024 or bundle_vocab(bundle) != 32000:
        raise ValueError('probe requires context 1024 and the 32000-token vocabulary')
    if torch.cuda.get_device_name(device) != 'Tesla T4' or torch.cuda.get_device_capability(device) != (7, 5):
        raise ValueError('probe requires actual Tesla T4 / SM75 hardware on every rank')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(73)
    torch.cuda.manual_seed_all(73)
    cfg = TrainConfig(bundle=str(bundle), out='unused', profile='d1', mask_mode='blockwise',
                      gdn_backend=backend, fused_optimizer=fused, world_size=d.world_size,
                      d1_supervision=supervision,
                      device='cuda', amp=True, amp_init_scale=8, batch_size=micro, grad_accum=64 // micro)
    model = build_model(cfg, 32000, 1024).to(device).train()
    parallel = d.wrap(model)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler('cuda', init_scale=8)
    loader = ShardLoader(bundle, batch_size=micro, seed=17, vocab=32000)
    generator = torch.Generator(device=device).manual_seed(1001)
    result = {'status': 'RUNNING', 'gpu': torch.cuda.get_device_name(device), 'devices': d.world_size,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'params': sum(p.numel() for p in model.parameters()),
              'context': 1024, 'global_batch': 64, 'global_micro': micro, 'per_gpu_micro': micro // d.world_size,
              'backend': backend, 'fused_optimizer': fused, 'mode': mode, 'supervision': supervision,
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'precision': 'fp16 autocast; fp32 state/solve/optimizer',
              'validation_included': False, 'checkpoint_included': False,
              'provenance': bundle_provenance(bundle, manifest),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob('*.py')}, 'windows': []}
    started = time.monotonic()
    def update():
        if time.monotonic() - started >= wall_cap:
            raise TimeoutError('bounded probe wall cap reached')
        staged = []
        consumed = processed = scored = 0
        for _ in range(cfg.grad_accum):
            target = next(loader).to(device)
            if supervision == 'all-blocks':
                inputs, mask = all_blocks_diffusion_mask(target, 32, generator=generator)
            else:
                inputs, mask = block_diffusion_mask(target, 32, generator=generator,
                                                     frontier=31 if mode == 'full' else None)
            count = int(mask.sum())
            if not count:
                raise ValueError('empty global loss mask')
            staged.append((inputs, target[:, :inputs.shape[1]], mask, count))
            consumed += target.numel(); processed += inputs.numel() * (2 if supervision == 'all-blocks' else 1); scored += count
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for index, (inputs, target, mask, count) in enumerate(staged):
            with d.sync_context(parallel, index == len(staged) - 1):
                with torch.autocast('cuda', dtype=torch.float16):
                    if d.world_size > 1:
                        summed = d.loss(parallel, inputs, target, mask, 32,
                                        clean_inputs=target if supervision == 'all-blocks' else None)
                        weighted = summed.float() * (d.world_size / scored)
                    else:
                        loss = model_masked_loss(model, inputs, target, mask,
                                                 clean_inputs=target if supervision == 'all-blocks' else None)
                        summed = loss.float() * count
                        weighted = summed / scored
                if not d.finite(summed):
                    raise FloatingPointError('nonfinite probe loss')
                scaler.scale(weighted).backward()
                loss_sum += summed.detach()
        norm, skipped = optimizer_step(model, optimizer, scheduler, scaler, 1.0)
        loss = d.loss_sum(loss_sum) / scored
        return {'consumed': consumed, 'processed': processed, 'scored': scored,
                'skipped': int(skipped), 'masked_ce': loss, 'grad_norm': norm}
    warm = [update() for _ in range(warmup)]
    result['warmup'] = warm
    for window in range(windows):
        d.barrier(); torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        start = time.monotonic()
        rows = [update() for _ in range(window_updates)]
        torch.cuda.synchronize(device); d.barrier()
        seconds = time.monotonic() - start
        totals = {key: sum(row[key] for row in rows) for key in ('consumed', 'processed', 'scored', 'skipped')}
        window_row = {'window': window, 'seconds': seconds, 'gpu_seconds': seconds * d.world_size,
                      **totals, 'updates': rows, 'peak_allocated_per_rank': torch.cuda.max_memory_allocated(device),
                      'peak_reserved_per_rank': torch.cuda.max_memory_reserved(device),
                      **{key + '_tokens_per_second': totals[key] / seconds for key in ('consumed', 'processed', 'scored')}}
        if d.world_size > 1:
            import torch.distributed as dist
            memory = torch.tensor([window_row['peak_allocated_per_rank'], window_row['peak_reserved_per_rank']], device=device)
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            window_row['peak_allocated_per_rank'], window_row['peak_reserved_per_rank'] = memory.tolist()
        result['windows'].append(window_row)
        if d.rank == 0:
            Path(output).write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(window_row), flush=True)
    if any(w['skipped'] for w in result['windows']):
        raise AssertionError('timed AMP skips; throughput is not qualified')
    result.update(status='PASS', wall_seconds=time.monotonic() - started, final_scale=scaler.get_scale())
    if d.rank == 0:
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--micro', type=int, default=8)
    p.add_argument('--backend', choices=('chunk', 'chunk-tc'), default='chunk')
    p.add_argument('--fused', action='store_true')
    p.add_argument('--mode', choices=('uniform', 'full'), default='uniform')
    p.add_argument('--supervision', choices=('frontier', 'all-blocks'), default='frontier')
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--window-updates', type=int, default=4)
    p.add_argument('--windows', type=int, default=3)
    p.add_argument('--wall-cap', type=float, default=600)
    a = p.parse_args()
    run(a.bundle, a.out, a.world_size, a.micro, a.backend, a.fused, a.mode,
        a.warmup, a.window_updates, a.windows, a.wall_cap, a.supervision)


if __name__ == '__main__':
    main()
