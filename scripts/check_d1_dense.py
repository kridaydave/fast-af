#!/usr/bin/env python3
"""Real-T4 dense supervision parity and process-resume checks. No pretraining.

Run each phase under an external wall-clock timeout. CPU tests live separately;
this command exits SKIP/2 when actual Tesla T4 hardware is unavailable.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_model import D1Config, D1LM
from scripts.train_diffusion import all_blocks_diffusion_mask, d1_training_execution
from scripts import check_gdn_t4


def independent_logits(model, clean, noisy):
    size = model.cfg.block_size
    return torch.cat([model(torch.cat((clean[:, :start], noisy[:, start:start + size]), dim=1))[:, -size:]
                      for start in range(0, clean.shape[1], size)], dim=1)


def parity(backend):
    torch.set_num_threads(1)
    torch.manual_seed(77)
    cfg = replace(D1Config(), d_model=64, n_layers=4, query_heads=4, kv_heads=1,
                  head_dim=16, ffn_hidden=128, gdn_heads=2, gdn_key_dim=16, gdn_value_dim=32)
    cpu = D1LM(512, 128, cfg, 'torch')
    gpu = D1LM(512, 128, cfg, backend).cuda()
    gpu.load_state_dict(cpu.state_dict())
    clean = torch.randint(5, 512, (2, 128))
    clean[0, 10] = 1
    noisy, mask = all_blocks_diffusion_mask(clean, 32, generator=torch.Generator().manual_seed(99))
    expected = independent_logits(cpu, clean, noisy)
    loss = F.cross_entropy(expected[mask], clean[mask]); loss.backward()
    with torch.autocast('cuda', dtype=torch.float16):
        actual = gpu(noisy.cuda(), clean_inputs=clean.cuda())
        actual_loss = F.cross_entropy(actual[mask.cuda()], clean.cuda()[mask.cuda()])
    actual_loss.backward()
    compare = check_gdn_t4.comparison
    metrics = {'cpu_reference_logits': compare(expected, actual, 0.03, 1e-5, 'CPU logits'),
               'cpu_reference_loss': compare(loss, actual_loss, 0.01, 1e-5, 'CPU loss'),
               'parameter_gradients': {}}
    for (name, left), (_, right) in zip(cpu.named_parameters(), gpu.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        metrics['parameter_gradients'][name] = compare(left.grad, right.grad, 0.05, 1e-6, name)
    del cpu, gpu, expected, actual, loss, actual_loss
    gc.collect(); torch.cuda.empty_cache()
    # Full delivered model, whole context, fixed masks, unchanged cache tolerance.
    # This is a forward/cache check; it does not claim CPU full-model gradients.
    model = D1LM(32000, 1024, backend=backend).cuda().eval()
    clean = torch.randint(5, 32000, (1, 1024), device='cuda')
    noisy, mask = all_blocks_diffusion_mask(clean, 32)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        dense = model(noisy, clean_inputs=clean, logits_mask=mask)
        cache, outputs = None, []
        for start in range(0, 1024, 32):
            logits, _ = model.forward_block(noisy[:, start:start + 32], cache)
            outputs.append(logits[mask[:, start:start + 32]])
            _, cache = model.forward_block(clean[:, start:start + 32], cache)
        metrics['full_model_cached_logits'] = compare(torch.cat(outputs), dense, 0.003, 1e-6, 'full cached logits')
        metrics['full_model_params'] = sum(p.numel() for p in model.parameters())
        metrics['full_model_context'] = 1024
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('parity', 'resume'), required=True)
    p.add_argument('--backend', choices=('chunk', 'chunk-tc'), default='chunk-tc')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = {'status': 'RUNNING', 'backend': a.backend, 'supervision': 'all-blocks', 'phase': a.phase,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'source_sha256': {s.name: hashlib.sha256(s.read_bytes()).hexdigest()
                                for s in Path(__file__).parent.glob('*.py')}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        report.update(status='SKIP', error='real Tesla T4 hardware unavailable')
        a.out.write_text(json.dumps(report, indent=2) + '\n'); sys.exit(2)
    report['gpu'] = torch.cuda.get_device_name(0)
    try:
        assert report['gpu'] == 'Tesla T4' and torch.cuda.get_device_capability(0) == (7, 5)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if a.phase == 'parity':
            with d1_training_execution('all-blocks', 'cuda'):
                report['checks'] = parity(a.backend)
        else:
            check_gdn_t4.BACKEND = a.backend
            check_gdn_t4.FUSED_OPTIMIZER = True
            check_gdn_t4.AMP_INIT_SCALE = 8
            report['checks'] = check_gdn_t4.resume_check(a.root, supervision='all-blocks')
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error=str(error))
        raise
    finally:
        a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'phase': a.phase, 'out': str(a.out)}), flush=True)


if __name__ == '__main__':
    main()
