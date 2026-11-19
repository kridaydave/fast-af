import json
import subprocess
import sys

from scripts.check_d1_parallel import interruption_worker, training_rows, kill_process_tree


def test_resume_log_parser_accepts_buffered_multiline_tail():
    # Reproduce communicate() returning several lines as one chunk after kill.
    chunks = ['{"step": 1, "masked_ce": 4}\n',
              '{"step": 2, "masked_ce": 3}\n{"step": 3, "masked_ce": 2}\nwarning\n',
              '{"params": 109137632}\n']
    assert [row['step'] for row in training_rows(chunks)] == [1, 2, 3]


def test_interruption_worker_really_stops_cpu_training_after_four_updates(tmp_path):
    # Actual small CPU training/process kill, not simulated GPU validation.
    from tests.test_train_diffusion import make_bundle, tiny_d1
    from dataclasses import asdict

    bundle = make_bundle(tmp_path, shard_sequences=5)
    config = tmp_path / 'd1.json'
    config.write_text(json.dumps(asdict(tiny_d1())))
    worker = tmp_path / 'interrupt_worker.py'
    interruption_worker(worker)
    out = tmp_path / 'run'
    command = [sys.executable, '-u', str(worker), '--bundle', str(bundle), '--out', str(out),
               '--profile', 'd1', '--d1-config', str(config), '--gdn-backend', 'chunk',
               '--d1-supervision', 'all-blocks', '--steps', '6', '--warmup', '1',
               '--batch-size', '1', '--checkpoint-every', '3', '--eval-every', '0']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    lines = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            if line.startswith('{') and json.loads(line).get('step') == 4:
                kill_process_tree(process.pid)
                break
        tail, _ = process.communicate(timeout=30)
        lines.append(tail)
    finally:
        if process.poll() is None:
            kill_process_tree(process.pid)
            process.wait()
    assert process.returncode == -9
    assert [row['step'] for row in training_rows(lines)] == [1, 2, 3, 4]
    assert (out / 'checkpoint-000003.pt').is_file()
    assert not (out / 'checkpoint-000006.pt').exists()


def test_kill_reaches_workers_that_created_separate_sessions(tmp_path):
    import select
    import time

    heartbeat = tmp_path / 'heartbeat'
    worker = ('from pathlib import Path\nimport time\np=Path(' + repr(str(heartbeat)) + ')\n'
              'p.write_text("ready")\nprint("worker-ready",flush=True)\n'
              'while True:\n p.write_text(str(time.time_ns()))\n time.sleep(.01)\n')
    parent = ('import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-u","-c",'
              + repr(worker) + '],start_new_session=True)\ntime.sleep(30)\n')
    process = subprocess.Popen([sys.executable, '-u', '-c', parent], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready and process.stdout.readline().strip() == 'worker-ready'
        kill_process_tree(process.pid)
        process.communicate(timeout=5)  # A surviving worker holds stdout open.
        assert process.returncode == -9
        before = heartbeat.read_text(); time.sleep(.1)
        assert heartbeat.read_text() == before
    finally:
        if process.poll() is None:
            kill_process_tree(process.pid)
            process.wait()
