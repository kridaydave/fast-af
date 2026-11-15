#!/usr/bin/env python3
"""Production-context T4/CPU output and cached-state parity. No training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_model import D1LM, GDNState, backend_info
from scripts.check_gdn_t4 import comparison


def check(backend='chunk-tc'):
    if torch.cuda.get_device_name() != 'Tesla T4' or torch.cuda.get_device_capability() != (7, 5):
        raise ValueError('requires a real T4')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(23)
    cpu = D1LM(32000, 1024).eval()
    gpu = D1LM(32000, 1024, backend=backend).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    tokens = torch.randint(5, 32000, (1, 1024))
    start = time.monotonic()
    with torch.no_grad():
        expected, cpu_cache = cpu._run(tokens, use_cache=True)
        with torch.autocast('cuda', dtype=torch.float16):
            full, full_cache = gpu._run(tokens.cuda(), use_cache=True)
            pieces, cache = [], None
            for offset in range(0, 1024, 32):
                logits, cache = gpu.forward_block(tokens[:, offset:offset+32].cuda(), cache)
                pieces.append(logits)
            cached = torch.cat(pieces, dim=1)
    assert cache is not None and cpu_cache is not None and full_cache is not None
    assert cache.tokens == full_cache.tokens == cpu_cache.tokens == 1024
    result = {'status': 'PASS', 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'backend': backend_info(backend), 'context': 1024,
              'params': sum(p.numel() for p in gpu.parameters()),
              'mode': 'fp16 autocast; CPU reference FP32',
              'cpu_full_output': comparison(expected, full, 0.03, 1e-5, 'CPU full logits'),
              'cpu_cached_output': comparison(expected, cached, 0.03, 1e-5, 'CPU cached logits'),
              'full_cached_output': comparison(full, cached, 0.003, 1e-5, 'full versus cached logits'),
              'states': []}
    for i, (a, b, c) in enumerate(zip(cpu_cache.layers, full_cache.layers, cache.layers)):
        cpu_parts = (a.recurrent, *a.conv) if isinstance(a, GDNState) else (a.keys, a.values)
        full_parts = (b.recurrent, *b.conv) if isinstance(b, GDNState) else (b.keys, b.values)
        cache_parts = (c.recurrent, *c.conv) if isinstance(c, GDNState) else (c.keys, c.values)
        result['states'].append({'layer': i,
            'cpu': [comparison(x, z, 0.05, 1e-5, f'CPU state {i}') for x, z in zip(cpu_parts, cache_parts)],
            'full': [comparison(y, z, 0.005, 1e-5, f'full state {i}') for y, z in zip(full_parts, cache_parts)]})
    result['wall_seconds'] = time.monotonic() - start
    result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('d1_model.py', 'check_d1_cache_long.py')}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('chunk', 'chunk-tc'), default='chunk-tc')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = check(args.backend)
    except Exception as error:
        import traceback
        report = {'status': 'FAIL', 'error': str(error), 'traceback': traceback.format_exc()}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('states', 'source_sha256')}), flush=True)
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
