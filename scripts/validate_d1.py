#!/usr/bin/env python3
"""Run real D1 cache/gradient checks on the selected device.

Defaults to CUDA + pinned FLA. CPU reference checks must be requested explicitly.
This is a synthetic-token correctness check, not a P0 throughput or P1 result.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import torch
import torch.nn.functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_model import D1Config, D1LM, backend_info, require_fla_device


def validate(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    checks: list[str] = []
    report: dict[str, object] = {'status': 'RUNNING', 'device': args.device, 'torch': torch.__version__,
              'python': platform.python_version(), 'cuda_available': torch.cuda.is_available(),
              'timestamp': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'inputs': 'synthetic clean vocabulary tokens', 'p0_measured': False,
              'resume_checked': False}
    if args.device == 'cuda' and not torch.cuda.is_available():
        report.update(status='SKIP', reason='CUDA unavailable; no GPU checks ran')
        return report, 2
    try:
        cfg = D1Config(**json.loads(args.config.read_text())) if args.config else D1Config()
        if args.vocab_size < 6:
            raise ValueError('vocab_size must be at least 6')
        torch.set_num_threads(1)
        torch.manual_seed(17)
        device = torch.device(args.device)
        backend = 'fla' if args.device == 'cuda' else 'torch'
        if backend == 'fla':
            require_fla_device(device)
        report.update(config=asdict(cfg), backend=backend_info(backend),
                      gpu=torch.cuda.get_device_name(device) if args.device == 'cuda' else None)
        model = D1LM(args.vocab_size, max(math.ceil(1024 / cfg.block_size) * cfg.block_size, 3 * cfg.block_size), cfg, backend).to(device)
        model.eval()
        report['params'] = sum(p.numel() for p in model.parameters())
        tokens = torch.randint(5, args.vocab_size, (1, 3 * cfg.block_size), device=device)
        def autocast():
            return torch.autocast('cuda', dtype=torch.float16) if args.device == 'cuda' else nullcontext()
        rtol, atol = (0.02, 0.003) if args.device == 'cuda' else (1e-5, 1e-6)
        with torch.no_grad(), autocast():
            full = model(tokens)
            if not bool(torch.isfinite(full).all()):
                raise FloatingPointError('non-finite full-model logits')
            prefix = model.prefill(tokens[:, :cfg.block_size])
            active = tokens[:, cfg.block_size:2 * cfg.block_size]
            first, _ = model.forward_block(active, prefix)
            model.forward_block(torch.zeros_like(active), prefix)
            repeated, committed = model.forward_block(active, prefix)
            torch.testing.assert_close(first, repeated, rtol=0, atol=0)
            torch.testing.assert_close(first, full[:, cfg.block_size:2 * cfg.block_size], rtol=rtol, atol=atol)
            last, _ = model.forward_block(tokens[:, 2 * cfg.block_size:], committed)
            torch.testing.assert_close(last, full[:, 2 * cfg.block_size:], rtol=rtol, atol=atol)
            altered = tokens.clone()
            altered[:, cfg.block_size:] = 5
            torch.testing.assert_close(model(altered)[:, :cfg.block_size], full[:, :cfg.block_size], rtol=rtol, atol=atol)
            report['cache_max_abs_error'] = float((first.float() - full[:, cfg.block_size:2 * cfg.block_size].float()).abs().max())
        checks.extend(['finite full-model forward', 'cached/full equivalence',
                                 'immutable prefix across denoising rounds', 'no future-block leakage'])
        if not args.forward_only:
            model.train()
            targets = tokens[:, :cfg.block_size]
            noisy = targets.clone()
            noisy[:, ::2] = 0
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            with autocast():
                logits = model(noisy)
                loss = F.cross_entropy(logits[:, ::2].flatten(0, 1), targets[:, ::2].flatten())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('non-finite masked CE')
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f'missing or non-finite gradient: {name}')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('non-finite parameter after AdamW update')
            report.update(masked_ce=float(loss.detach()), grad_norm=float(norm))
            checks.extend(['finite backward through every parameter', 'finite AdamW update'])
        source = Path(__file__).with_name('d1_model.py')
        report['model_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        report['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=source.parent, text=True).strip()
        report['status'] = 'PASS'
        return report, 0
    except (ValueError, RuntimeError, FloatingPointError, OSError, AssertionError) as error:
        report.update(status='FAIL', error=str(error))
        return report, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--forward-only', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report, status = validate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
