"""Benchmark week orchestration: load roster, run each model, write JSON."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from .backends import TransformersBackend, DllmBackend, CustomBackend, Generation
from .humaneval import run_humaneval

# Prompts for the speed harness: coding-style, batch 1, out_len tokens each.
_SPEED_PROMPTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return",
    "class MaxHeap:\n    def __init__(self):\n        self.data = []\n    def push(self, x):\n        self.data.append(x)\n        self._sift_up(len(self.data) - 1)\n    def pop(self):\n        top = self.data[0]\n        self.data[0] = self.data[-1]\n        self.data.pop()\n        self._sift_down(0)\n        return top\n    def _sift_up(self, i):\n        while i > 0:\n            p = (i - 1) // 2\n            if self.data[i] > self.data[p]:\n                self.data[i], self.data[p] = self.data[p], self.data[i]\n                i = p\n            else:\n                break\n    def _sift_down(self, i):\n        n = len(self.data)\n        while True:\n            l, r, big = 2*i+1, 2*i+2, i\n            if l < n and self.data[l] > self.data[big]:\n                big = l\n            if r < n and self.data[r] > self.data[big]:\n                big = r\n            if big != i:\n                self.data[i], self.data[big] = self.data[big], self.data[i]\n                i = big\n            else:\n                break\n    def peek(self):\n        return",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    out, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    out.extend(left[i:]); out.extend(right[j:])\n    return",
]


def _build_backend(m: dict, common: dict):
    b = m["backend"]
    if b in ("vllm", "hf"):
        return TransformersBackend(m["repo"], m.get("quant"),
                                   common["dtype"])
    if b == "dllm":
        return DllmBackend(m["repo"], m.get("mode", "default"),
                           m.get("steps", 8), common["dtype"])
    if b == "custom":
        raise NotImplementedError(
            "custom backend (smoldlm-144m): wire score_fn + tokenizer from "
            "the smoldlm repo in bench/backends.py before running.")
    raise ValueError(f"unknown backend: {b}")


def _timed_gen(backend, prompt: str, out_len: int, runs: int) -> dict:
    tps = []
    texts = []
    toks = []
    for _ in range(runs):
        g: Generation = backend.generate(prompt, out_len)
        tps.append(g.num_tokens / g.decode_seconds)
        texts.append(g.text)
        toks.append(g.num_tokens)
    return {
        "median_tps": round(statistics.median(tps), 3),
        "min_tps": round(min(tps), 3),
        "max_tps": round(max(tps), 3),
        "runs": runs,
        "sample_text": texts[0][:200],
        "tokens_per_run": toks,
    }


def run_model(m: dict, common: dict) -> dict:
    print(f"[run] {m['id']} ({m['backend']})", flush=True)
    backend = _build_backend(m, common)
    backend.warmup()
    prompt = _SPEED_PROMPTS[0]
    speed = _timed_gen(backend, prompt, common["out_len"], common["runs"])
    result: dict = {
        "id": m["id"],
        "backend": m["backend"],
        "repo": m.get("repo"),
        "note": m.get("note", ""),
        "speed": speed,
    }
    if m.get("humaneval", common.get("humaneval", False)):
        print(f"[eval] HumanEval pass@1 for {m['id']}", flush=True)
        try:
            result["humaneval"] = run_humaneval(
                lambda p: backend.generate(p, 256).text,
                prompts=50)
        except Exception as e:  # speed result already in hand; don't discard it
            result["humaneval"] = {"error": repr(e)}
            print(f"[skip] humaneval for {m['id']}: {e}", flush=True)
    return result


def run(config_path: str, output_path: str | None = None,
        models: list[dict] | None = None) -> list[dict]:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    common = cfg["common"]
    if models is None:
        models = cfg["models"]
    results = []
    t_start = time.perf_counter()

    def _save():
        # Write after EVERY model: a hard crash late in the roster (e.g. the
        # 8B diffusion runs OOMing the process) must not lose earlier results.
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out = {"elapsed_seconds": round(time.perf_counter() - t_start, 1),
                   "common": common, "results": results}
            Path(output_path).write_text(yaml.safe_dump(out, sort_keys=False))

    for m in models:
        try:
            results.append(run_model(m, common))
        except Exception as e:  # keep going; record the failure
            results.append({"id": m["id"], "backend": m["backend"],
                            "error": repr(e)})
            print(f"[skip] {m['id']}: {e}", flush=True)
        _save()
    print(f"[done] wrote {output_path}", flush=True)
    return results
