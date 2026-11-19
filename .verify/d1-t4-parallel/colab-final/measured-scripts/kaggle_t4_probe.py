#!/usr/bin/env python3
"""Bounded two-T4 notebook job. Package with package_kaggle_probe.py. No pretraining."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import torch


def main():
    r = Path("/kaggle/working/d1-opt")
    r.mkdir(exist_ok=True)
    sys.path.insert(0, str(r / "src"))
    started = time.monotonic()
    hardware = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        "nvidia_smi": subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True
        ).stdout,
        "topology": subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
        ).stdout,
    }
    (r / "hardware.json").write_text(json.dumps(hardware, indent=2))
    print(json.dumps(hardware), flush=True)
    assert len(hardware["devices"]) == 2 and all(
        x["name"] == "Tesla T4" and x["capability"] == (7, 5)
        for x in hardware["devices"]
    )
    phases = []

    def phase(name, args, cap, required=True):
        remaining = 1100 - (time.monotonic() - started)
        if remaining < cap:
            raise TimeoutError("whole job budget cannot cover " + name)
        t = time.monotonic()
        with (r / (name + ".log")).open("w") as log:
            child = subprocess.Popen(
                [sys.executable, *args],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=r / "src",
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                code = 124
        row = {"phase": name, "exit_code": code, "seconds": time.monotonic() - t}
        phases.append(row)
        (r / "phases.json").write_text(json.dumps(phases, indent=2))
        print(json.dumps(row), flush=True)
        if code and required:
            raise RuntimeError(
                f"{name} failed: " + (r / (name + ".log")).read_text()[-2500:]
            )
        return code

    try:
        # The CPU reference checker itself enforces T4 and tests output/gradient/cache.
        phase(
            "native-parity",
            [
                "-u",
                "scripts/check_gdn_t4.py",
                "--phase",
                "parity",
                "--backend",
                "chunk",
                "--root",
                str(r),
                "--out",
                str(r / "native-parity.json"),
            ],
            120,
        )
        tc_supported = True
        try:
            a = torch.randn(1, 8, 8, device="cuda", dtype=torch.float16)
            torch.bmm(a, a, out_dtype=torch.float32)
            del a
            torch.cuda.empty_cache()
        except (TypeError, RuntimeError):
            tc_supported = False
        backend = "chunk-tc" if tc_supported else "chunk"
        (r / "selection.json").write_text(
            json.dumps(
                {"tensorcore_bmm_available": tc_supported, "chosen_backend": backend}
            )
        )
        if tc_supported:
            phase(
                "tc-parity",
                [
                    "-u",
                    "scripts/check_gdn_t4.py",
                    "--phase",
                    "parity",
                    "--backend",
                    backend,
                    "--fused-optimizer",
                    "--root",
                    str(r),
                    "--out",
                    str(r / "tc-parity.json"),
                ],
                120,
            )
        phase(
            "parallel-gradients",
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "scripts/check_d1_parallel.py",
                "--phase",
                "gradients",
                "--backend",
                backend,
                "--root",
                str(r),
                "--out",
                str(r / "parallel-gradients.json"),
            ],
            90,
        )
        for name, devices, micro, b, fused in [
            ("baseline-micro4", 1, 4, "chunk", False),
            ("optimized-micro4", 1, 4, backend, True),
            ("optimized-micro8", 1, 8, backend, True),
            ("paired-dual", 2, 8, backend, True),
            ("dual-micro16", 2, 16, backend, True),
        ]:
            args = [
                "-u",
                "scripts/benchmark_d1_parallel.py",
                "--bundle",
                str(r / "src/probe-bundle"),
                "--out",
                str(r / (name + ".json")),
                "--world-size",
                str(devices),
                "--micro",
                str(micro),
                "--backend",
                b,
                "--wall-cap",
                "180",
            ]
            if fused:
                args += ["--fused"]
            if devices == 2:
                args = [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    *args[1:],
                ]
            phase(name, args, 210)
        phase(
            "parallel-resume",
            [
                "-u",
                "scripts/check_d1_parallel.py",
                "--phase",
                "resume",
                "--backend",
                backend,
                "--root",
                str(r),
                "--out",
                str(r / "parallel-resume.json"),
            ],
            240,
        )
        (r / "done.json").write_text(
            json.dumps({"status": "PASS", "wall_seconds": time.monotonic() - started})
        )
    except Exception:
        (r / "failure.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        # Reports/logs are retained, temporary multi-GB checkpoints are not notebook outputs.
        for name in ("full", "interrupted", "restarted"):
            if (r / name).exists():
                shutil.rmtree(r / name)
        (r / "allocation-work.json").write_text(
            json.dumps(
                {
                    "device_count": 2,
                    "job_wall_seconds": time.monotonic() - started,
                    "gpu_seconds_excluding_startup_teardown": 2
                    * (time.monotonic() - started),
                }
            )
        )


if __name__ == "__main__":
    main()
