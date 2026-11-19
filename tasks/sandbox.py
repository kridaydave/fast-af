"""Fail-closed Linux bubblewrap execution; the parent owns grading."""
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import time

from scripts.prepare_corpus import canonical

WORKER = Path(__file__).with_name("worker.py").resolve()
MAX_OUTPUT = 65536
MAX_CODE = 32768


def command():
    bwrap = shutil.which("bwrap")
    if not bwrap or not Path("/usr/bin/python3").is_file():
        raise RuntimeError("Linux bubblewrap and /usr/bin/python3 are required; no unsandboxed fallback")
    return [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--clearenv", "--setenv", "PYTHONHASHSEED", "0", "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
            "--ro-bind", str(WORKER), "/worker.py", "/usr/bin/python3", "-I", "-S", "/worker.py"]


def execute(code, inputs, timeout=3.0):
    if not isinstance(code, str) or len(code.encode()) > MAX_CODE:
        return {"status": "invalid_code"}
    payload = canonical({"code": code, "inputs": inputs}).encode()
    if len(payload) > MAX_OUTPUT:
        return {"status": "input_limit"}
    started = time.monotonic()
    with subprocess.Popen(command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, start_new_session=True) as proc:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
            output = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        return {"status": "timeout"}
                    for key, _ in selector.select(remaining):
                        chunk = os.read(key.fd, 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        output.extend(chunk)
                        if len(output) > MAX_OUTPUT:
                            return {"status": "output_limit"}
            try:
                proc.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                return {"status": "timeout"}
            if proc.returncode:
                return {"status": "execution_error"}
            try:
                values = json.loads(output)
                if not isinstance(values, list) or len(values) != len(inputs):
                    raise ValueError("wrong result count")
                if not all(finite_json(x) for x in values):
                    raise ValueError("nonfinite result")
                return {"status": "completed", "values": values}
            except (ValueError, UnicodeError, RecursionError):
                return {"status": "invalid_output"}
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def finite_json(value):
    import math
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_json(x) for x in value)
    if isinstance(value, dict):
        return all(finite_json(x) for x in value.values())
    return True


def same(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same(actual[k], expected[k]) for k in expected)
    return actual == expected


def verify(code, task):
    result = execute(code, [c["args"] for c in task["cases"]])
    if result["status"] == "completed":
        result = {"status": "passed" if all(same(v, c["expected"]) for v, c in zip(result["values"], task["cases"])) else "wrong_answer"}
    return result


def probe():
    result = execute("def solve(x):\n    return x + 1\n", [[1]])
    if result != {"status": "completed", "values": [2]}:
        raise RuntimeError(f"bubblewrap execution unavailable: {result['status']}; refusing to grade")
