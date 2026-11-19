#!/usr/bin/env python3
"""Real two-T4 gradient and process-resume checks, bounded by the caller."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.d1_model import D1LM
from scripts.distributed_training import DistributedTraining
from scripts.check_gdn_t4 import comparison
from scripts.train_diffusion import load_checkpoint


def gradient_check(out, backend, supervision='frontier'):
    d = DistributedTraining(2, 'cuda')
    try:
        assert torch.cuda.get_device_name(d.device) == 'Tesla T4'
        assert torch.cuda.get_device_capability(d.device) == (7, 5)
        torch.set_num_threads(1)
        torch.manual_seed(19)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = D1LM(32000, 1024, backend=backend).to(d.device)
        reference = D1LM(32000, 1024, backend=backend).to(d.device)
        reference.load_state_dict(model.state_dict())
        parallel = d.wrap(model)
        generator = torch.Generator().manual_seed(31)
        targets = torch.randint(5, 32000, (4, 64), generator=generator).to(d.device)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[2, -2:] = True
        mask[3, -16:] = True
        inputs = targets.masked_fill(mask, 0)
        clean = targets if supervision == 'all-blocks' else None
        # Rank 0 has no scored tokens. Both must still backpropagate/reduce.
        with torch.autocast('cuda', dtype=torch.float16):
            expected = F.cross_entropy(reference(inputs, logits_mask=mask, clean_inputs=clean), targets[mask])
            summed = d.loss(parallel, inputs, targets, mask, 32, clean_inputs=clean)
        expected.backward()
        (summed * (2 / int(mask.sum()))).backward()
        metrics = {}
        for (name, a), (_, b) in zip(reference.named_parameters(), model.named_parameters()):
            assert a.grad is not None and b.grad is not None, name
            metrics[name] = comparison(a.grad, b.grad, 0.05, 1e-7, name)
        actual_loss = d.loss_sum(summed.detach()) / int(mask.sum())
        assert abs(float(expected.detach()) - actual_loss) < 0.003
        if d.rank == 0:
            Path(out).write_text(json.dumps({'status': 'PASS', 'backend': backend, 'supervision': supervision,
                'params': sum(p.numel() for p in model.parameters()), 'zero_scored_rank': True,
                'masked_ce_reference': float(expected.detach()), 'masked_ce_distributed': actual_loss,
                'parameter_gradients': metrics}, indent=2) + '\n')
    finally:
        d.close()


def training_rows(chunks):
    """communicate() can return several newline-delimited records in one chunk."""
    return [row for line in ''.join(chunks).splitlines()
            if line.startswith('{') and 'masked_ce' in (row := json.loads(line))]


def kill_process_tree(pid):
    """torchrun workers have their own sessions; killpg(launcher) misses them."""
    pids = [pid]
    for parent in pids:
        try:
            children = Path(f'/proc/{parent}/task/{parent}/children').read_text()
            pids.extend(int(child) for child in children.split())
        except FileNotFoundError:
            pass
    # Capture descendants first, then kill the launcher before its workers so
    # elastic's error handling cannot win the race and turn SIGKILL into exit 1.
    for child in pids:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def interruption_worker(path):
    """Pause rank zero at the logged update until the controller kills the job.

    A stdout reader alone can lag behind GPU work. The rank-zero pause also
    prevents the other rank from applying another collective optimizer update.
    Only logging is wrapped; the real trainer and update arithmetic are intact.
    """
    source = Path(__file__).resolve().parent.parent
    Path(path).write_text(
        'import json,sys,time\n'
        f'sys.path.insert(0, {str(source)!r})\n'
        'from scripts import train_diffusion as trainer\n'
        'original_train = trainer.train\n'
        'def train(cfg, resume=None, log=print):\n'
        '    def observed(line):\n'
        '        print(line, flush=True)\n'
        '        if isinstance(line, str) and line.startswith("{") and json.loads(line).get("step") == 4:\n'
        '            while True: time.sleep(1)\n'
        '    return original_train(cfg, resume=resume, log=observed)\n'
        'trainer.train = train\n'
        'trainer.main()\n')


def resume_check(root, backend, supervision='frontier'):
    root = Path(root)
    script = Path(__file__).with_name('train_diffusion.py')
    base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
            str(script), '--bundle', str(root / 'src/probe-bundle'), '--profile', 'd1',
            '--gdn-backend', backend, '--device', 'cuda', '--amp', '--amp-init-scale', '8',
            '--world-size', '2', '--batch-size', '2', '--grad-accum', '1', '--steps', '6',
            '--warmup', '2', '--lr', '0.0001', '--seed', '1001', '--checkpoint-every', '3',
            '--eval-every', '3', '--eval-batches', '1', '--fused-optimizer']
    if supervision != 'frontier':
        base += ['--d1-supervision', supervision]
    worker = root / 'interrupt_worker.py'
    interruption_worker(worker)
    reports = []
    started = time.monotonic()
    for name, resume, kill in [('full', None, False), ('interrupted', None, True),
                               ('restarted', root/'interrupted/checkpoint-000003.pt', False)]:
        argv = base + ['--out', str(root/name)]
        if kill:
            argv[argv.index(str(script))] = str(worker)
        if resume:
            argv += ['--resume', str(resume)]
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)
        lines = []
        try:
            assert p.stdout is not None
            for line in p.stdout:
                lines.append(line)
                if kill and line.startswith('{') and json.loads(line).get('step') == 4:
                    kill_process_tree(p.pid)
                    break
            rest, _ = p.communicate(timeout=30)
            lines.append(rest)
        finally:
            if p.poll() is None:
                kill_process_tree(p.pid)
                p.wait()
            (root/f'{name}.log').write_text(''.join(lines))
        assert p.returncode == (-9 if kill else 0), (name, p.returncode, ''.join(lines)[-3000:])
        rows = training_rows(lines)
        assert [row['step'] for row in rows] == (list(range(1, 5)) if kill
                                                 else list(range(4, 7)) if resume else list(range(1, 7)))
        if kill:
            assert not (root / name / 'checkpoint-000006.pt').exists(), 'job finished before interruption'
        assert all(not row['optimizer_step_skipped'] for row in rows)
        reports.append(rows)
    left = load_checkpoint(root/'full/checkpoint-000006.pt')
    right = load_checkpoint(root/'restarted/checkpoint-000006.pt')
    for key in ('step','tokens_seen','tokens_processed','tokens_scored','scaler','scheduler','loader'):
        assert left[key] == right[key], key
    assert left['scheduler']['last_epoch'] == 6
    assert torch.equal(left['corruption_rng'], right['corruption_rng'])
    for a, b in zip(left['distributed']['ranks'], right['distributed']['ranks']):
        assert all(torch.equal(a[k], b[k]) for k in ('cpu', 'cuda'))
    loss_error = max(abs(a['masked_ce'] - b['masked_ce'])
                     for a, b in zip(reports[0][3:], reports[2]))
    assert len(reports[2]) == 3 and loss_error < 0.003
    weights = {name: comparison(x, right['model'][name], 0.003, 1e-5, name)
               for name, x in left['model'].items()}
    optimizer_max = 0.0
    for index, state in left['optimizer']['state'].items():
        for key, value in state.items():
            metric = comparison(value, right['optimizer']['state'][index][key], 0.003, 1e-5,
                                f'optimizer {index} {key}')
            optimizer_max = max(optimizer_max, metric['max_abs'])
    result = {'status': 'PASS', 'backend': backend, 'supervision': supervision, 'world_size': 2, 'params': sum(v.numel() for v in left['model'].values()),
              'steps': 6, 'kill_after': 4, 'resume_from': 3, 'successful_updates': 6,
              'loader_scheduler_scaler_rng_equal': True, 'sampled_eval_on_rank0': True,
              'loss_max_abs_error': loss_error, 'weight_metrics': weights,
              'optimizer_max_abs_error': optimizer_max, 'wall_seconds': time.monotonic()-started,
              'checkpoint_sha256': {name: hashlib.sha256((root/name/'checkpoint-000006.pt').read_bytes()).hexdigest()
                                    for name in ('full','restarted')}}
    del left, right
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('gradients', 'resume'), required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--backend', choices=('chunk', 'chunk-tc'), default='chunk')
    p.add_argument('--supervision', choices=('frontier', 'all-blocks'), default='frontier')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.phase == 'gradients':
        gradient_check(a.out, a.backend, a.supervision)
    else:
        a.out.write_text(json.dumps(resume_check(a.root, a.backend, a.supervision), indent=2) + '\n')


if __name__ == '__main__':
    main()
