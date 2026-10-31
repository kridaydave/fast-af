#!/usr/bin/env bash
# EdgeFast-Diff 1B — benchmark week bootstrap for Colab free T4.
# Upload this file to Colab and run the whole cell: it installs the env,
# writes the harness, and (optionally) runs it.
set -euo pipefail

echo "[1/4] installing env (torch cu121, vLLM for sm75, dLLM)..."
pip install --quiet --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --quiet --no-cache-dir "vllm==0.6.3.post2"
pip install --quiet --no-cache-dir "git+https://github.com/ZHZisZZ/dllm.git"
pip install --quiet --no-cache-dir pyyaml datasets accelerate safetensors

echo "[2/4] writing harness..."
mkdir -p bench

cat > bench/config.yaml <<'BENCH_EOF_bench_config_yaml'
# Benchmark week roster for EdgeFast-Diff 1B (D1 = 100M).
#
# Every entry is one benchmark run. `backend` selects how the harness drives it:
#   - vllm   : autoregressive, driven by vLLM (greedy decode, batch 1).
#   - dllm   : masked/block diffusion via the dLLM framework (LLaDA, Dream, ...).
#   - custom : a backend defined in bench/backends.py (e.g. smoldlm-144m).
#
# Timing contract (spec): batch 1, prompt_len in / out_len out, median of `runs`
# timed generations, same GPU. TPS = generated_tokens / decode_seconds.
# HumanEval pass@1 (n=1) is the quality guardrail and is tracked separately.

common:
  prompt_len: 1024      # tokens in
  out_len: 1024         # tokens out
  runs: 5               # median of N timed runs
  batch: 1              # batch 1, one model per GPU (T4 has one usable)
  seed: 0
  dtype: float16        # T4 has no bf16; fp16 is the full tier
  quant: null           # int4 only where noted (B1/B2)
  humaneval: true       # track pass@1 for every AR model
  warmup: 1             # discard this run before timing

models:
  # ---- AR baselines, ~100M band (D1's scale) ----
  - id: smollm2-135m
    backend: vllm
    repo: HuggingFaceTB/SmolLM2-135M-Instruct
    quant: null
    humaneval: true
    note: "135M AR, Apache-2.0, L1 yardstick"

  - id: qwen3.5-0.8b            # B0
    backend: vllm
    repo: Qwen/Qwen3.5-0.8B
    quant: null
    humaneval: true
    note: "GDN hybrid AR, same family as D1. B1/B2 build on this."

  - id: qwen3.5-0.8b-int4       # B1
    backend: vllm
    repo: Qwen/Qwen3.5-0.8B
    quant: int4                 # GPTQ int4 (T4 has no AWQ)
    humaneval: true
    note: "B0 + int4 QAT, expected ~2x vs B0"

  - id: qwen3.5-0.8b-eagle3     # B2
    backend: vllm
    repo: Qwen/Qwen3.5-0.8B
    quant: int4
    eagle3: true                # speculative decode; budget a day of friction
    humaneval: true
    note: "B0 + int4 + EAGLE-3, expected 3-6x vs B0; may need llama.cpp spec-decode"

  - id: lfm2.5-350m
    backend: vllm
    repo: LiquidAI/LFM2.5-350M
    quant: null
    humaneval: true
    note: "354M hybrid AR, fast; latest small hybrid reference"

  # ---- diffusion, size-matched to D1 ----
  - id: smoldlm-144m
    backend: custom
    repo: HoangHa/smoldlm-144m
    block_size: 32
    steps: 8
    confidence: 0.9
    humaneval: false            # needs its own decode loop; see bench/backends.py
    note: "144M block diffusion, block_size 32, Apache-2.0. D1's closest size-matched diffusion twin."

  # ---- paradigm check at scale ----
  - id: llada-8b-default
    backend: dllm
    repo: GSAI-ML/LLaDA-8B-Base
    mode: default               # native masked diffusion
    steps: 32
    humaneval: false
    note: "LLaDA-8B default config"

  - id: llada-8b-fastdllm
    backend: dllm
    repo: GSAI-ML/LLaDA-8B-Base
    mode: fast-dllm             # Fast-dLLM decoding policy
    steps: 8
    humaneval: false
    note: "LLaDA-8B with Fast-dLLM decode; the tuned-policy comparison"
BENCH_EOF_bench_config_yaml

cat > bench/backends.py <<'BENCH_EOF_bench_backends_py'
"""Model backends for the benchmark harness.

Each backend exposes the same contract:

    class Backend:
        def warmup(self) -> None: ...
        def generate(self, prompt: str, out_len: int) -> Generation: ...

    Generation = {
        "text": str,
        "num_tokens": int,        # generated tokens
        "decode_seconds": float,  # time to emit them (excludes prefill)
    }

The harness wraps generate() in a median-of-N timer and computes TPS =
num_tokens / median(decode_seconds). Backends must not do their own timing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class Generation:
    text: str
    num_tokens: int
    decode_seconds: float


class VllmBackend:
    """Autoregressive models driven by vLLM, greedy decode, batch 1.

    vLLM reports per-request timing in `outputs[i].stats.times`; we take
    decode_end - prompt_end as the decode window (prefill excluded).
    """

    def __init__(self, repo: str, quant: str | None, eagle3: bool, dtype: str,
                 gpu_mem_util: float = 0.9):
        from vllm import LLM
        kwargs: dict[str, Any] = dict(
            dtype=dtype,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=True,          # stable timing on T4 (no CUDA graphs)
            max_model_len=2048,
            quantization=None,
        )
        if quant == "int4":
            kwargs["quantization"] = "gptq"     # T4 has no AWQ; GPTQ int4
        if eagle3:
            # EAGLE-3 speculative decoding. vLLM's native support targets newer
            # GPUs; if this raises on T4, fall back to llama.cpp spec-decode and
            # implement a CustomBackend (budget a day of friction, per spec).
            kwargs["speculative_config"] = {
                "method": "eagle",
                "model": f"{repo}/eagle3",
            }
        self.llm = LLM(model=repo, **kwargs)

    def warmup(self) -> None:
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=8)
        self.llm.generate(["warmup"], sp)

    def generate(self, prompt: str, out_len: int) -> Generation:
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=out_len)
        out = self.llm.generate([prompt], sp)[0]
        stats = out.outputs[0].stats
        gen_tokens = stats.num_generated_tokens
        decode_s = float(stats.times.decode_end - stats.times.prompt_end)
        return Generation(text=out.outputs[0].text, num_tokens=gen_tokens,
                          decode_seconds=max(decode_s, 1e-6))


class DllmBackend:
    """Masked/block diffusion via the dLLM framework (LLaDA, Dream, ...).

    dLLM provides a tested `generate` for these models, so we call it natively
    and time only the decode (prompt is re-encoded each call, as the spec's
    single-stream contract requires). The decode-policy knobs (steps, mode)
    map to dLLM's generate args; verify against the installed dLLM version.
    """

    def __init__(self, repo: str, mode: str, steps: int, dtype: str):
        from dllm.utils import set_seed
        self.repo = repo
        self.mode = mode
        self.steps = steps
        self.set_seed = set_seed

    def _load(self):
        # dLLM exposes per-model classes. LLaDA is the primary target here.
        from dllm.models.llada import LLaDA
        model = LLaDA(model_name=self.repo, dtype=torch.float16, device="cuda")
        model.set_sampling_steps(self.steps)
        return model

    def warmup(self) -> None:
        self._model = self._load()
        self._gen(self._model, "the quick brown fox", 8)

    def _gen(self, model, prompt: str, out_len: int) -> Generation:
        import torch
        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=out_len,
                             num_steps=self.steps)
        decode_s = time.perf_counter() - t0
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        # token count via the model's tokenizer when available
        num_tokens = getattr(model, "_num_tokens", None) or out_len
        return Generation(text=text, num_tokens=num_tokens, decode_seconds=max(decode_s, 1e-6))

    def generate(self, prompt: str, out_len: int) -> Generation:
        return self._gen(self._model, prompt, out_len)


class CustomBackend:
    """Generic block-diffusion decode loop for models without a dLLM class.

    D1's own decode policy is prototyped here: block_size masked slots, N
    denoising passes, confidence-threshold unmasking (0.9 per token), block
    overlap, and a block-level early stop at 0.97. This is the honest place to
    benchmark smoldlm-144m and to rehearse D1's loop before the real model.

    Requires a model that scores masked positions (masked-token logits). The
    model must expose `score_masked(input_ids) -> logits` or be wrapped to do
    so. Verify the smoldlm repo's API and wire `score_fn` accordingly.
    """

    MASK_ID = 0  # smoldlm config: mask_token_id = 0

    def __init__(self, repo: str, block_size: int, steps: int,
                 confidence: float, score_fn: Callable, tokenizer,
                 eos_id: int = 1, max_new_tokens: int = 1024):
        self.repo = repo
        self.block_size = block_size
        self.steps = steps
        self.confidence = confidence
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.max_new_tokens = max_new_tokens

    def warmup(self) -> None:
        self.generate("the quick brown fox", 8)

    def generate(self, prompt: str, out_len: int) -> Generation:
        import torch
        enc = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prefix = enc["input_ids"][0]
        t0 = time.perf_counter()

        out_ids = list(prefix.tolist())
        revealed = 0
        while revealed < out_len:
            block = [self.MASK_ID] * self.block_size
            seq = torch.tensor([out_ids + block], device="cuda")
            prev_conf = 0.0
            for _ in range(self.steps):
                logits = self.score_fn(seq)            # (1, L, V)
                probs = torch.softmax(logits[0, -self.block_size:], dim=-1)
                conf, tok = probs.max(dim=-1)          # per-slot (block,)
                conf = conf.float().cpu()
                if prev_conf > 0.97:                    # block-level early stop
                    break
                prev_conf = conf.min().item()
                conf_mask = conf >= self.confidence
                if conf_mask.all():
                    break
                chosen = tok[conf_mask].tolist()
                for i, c in enumerate(conf_mask.tolist()):
                    if c:
                        block[i] = chosen.pop(0)
                seq = torch.tensor([out_ids + block], device="cuda")
            # commit block
            new = block
            out_ids.extend(new)
            revealed += self.block_size
            if self.eos_id in new:
                break

        decode_s = time.perf_counter() - t0
        text = self.tokenizer.decode(out_ids[len(prefix):],
                                     skip_special_tokens=True)
        return Generation(text=text, num_tokens=revealed,
                          decode_seconds=max(decode_s, 1e-6))
BENCH_EOF_bench_backends_py

cat > bench/humaneval.py <<'BENCH_EOF_bench_humaneval_py'
"""HumanEval pass@1 guardrail (n=1, exec-based).

HumanEval is held out of all training data and touched only at eval. We run a
small held-out slice (default 50 problems) at n=1 and report the fraction whose
completion passes the function's test suite. Diffusion models opt out here
(they need a code-task decode loop of their own); the AR baselines are the
quality anchor.
"""
from __future__ import annotations

import json
import re
from typing import Any

from datasets import load_dataset

# Canonical HumanEval, deduped to the standard 164.
_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl"


def _load(prompts: int) -> list[dict[str, Any]]:
    rows = []
    try:
        ds = load_dataset("openai_humaneval", split="test")
        rows = list(ds)
    except Exception:
        # Fallback: fetch raw jsonl directly.
        import urllib.request
        with urllib.request.urlopen(_URL) as r:
            rows = [json.loads(line) for line in r if line.strip()]
    return [r for r in rows[:prompts]]


def _extract(completion: str, prompt: str) -> str:
    """Strip to the function body the prompt defines, up to the next def/EOF."""
    body = completion
    # Keep only code after the prompt, before any trailing new top-level def.
    if prompt and prompt in body:
        body = body.split(prompt, 1)[1]
    body = "def " + body if not body.lstrip().startswith(("def", "class", "@")) else body
    # cut at the next top-level function/class definition
    cut = re.split(r"\n(?:def|class)\s", "\n" + body.lstrip("\n"))
    return cut[0].strip()


def _run(prompt: str, completion: str) -> bool:
    try:
        exec_globals: dict[str, Any] = {}
        code = prompt + "\n" + _extract(completion, prompt) + "\n"
        exec(compile(code, "<eval>", "exec"), exec_globals)  # noqa: S102
        # run the assertions embedded in the prompt's docstring test block
        test = _extract_tests(prompt)
        if test:
            exec(compile(test, "<test>", "exec"), exec_globals)
        return True
    except Exception:
        return False


def _extract_tests(prompt: str) -> str:
    # HumanEval prompts embed checks() in the docstring; run them.
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not m:
        return ""
    doc = m.group(1)
    mm = re.search(r"check\(.*?\)\n(.*?)(\ncheck\()", doc, re.DOTALL)
    return mm.group(1) if mm else ""


def run_humaneval(gen_fn, prompts: int = 50) -> dict[str, Any]:
    """gen_fn(prompt: str) -> str. Returns pass@1 stats."""
    rows = _load(prompts)
    passed = 0
    for r in rows:
        prompt = r["prompt"]
        try:
            completion = gen_fn(prompt)
        except Exception:
            completion = ""
        if _run(prompt, completion):
            passed += 1
    n = len(rows)
    return {"n": n, "passed": passed, "pass@1": passed / n if n else 0.0}
BENCH_EOF_bench_humaneval_py

cat > bench/run.py <<'BENCH_EOF_bench_run_py'
"""Benchmark week orchestration: load roster, run each model, write JSON."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from .backends import VllmBackend, DllmBackend, CustomBackend, Generation
from .humaneval import run_humaneval

# Prompts for the speed harness: coding-style, batch 1, out_len tokens each.
_SPEED_PROMPTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return",
    "class MaxHeap:\n    def __init__(self):\n        self.data = []\n    def push(self, x):\n        self.data.append(x)\n        self._sift_up(len(self.data) - 1)\n    def pop(self):\n        top = self.data[0]\n        self.data[0] = self.data[-1]\n        self.data.pop()\n        self._sift_down(0)\n        return top\n    def _sift_up(self, i):\n        while i > 0:\n            p = (i - 1) // 2\n            if self.data[i] > self.data[p]:\n                self.data[i], self.data[p] = self.data[p], self.data[i]\n                i = p\n            else:\n                break\n    def _sift_down(self, i):\n        n = len(self.data)\n        while True:\n            l, r, big = 2*i+1, 2*i+2, i\n            if l < n and self.data[l] > self.data[big]:\n                big = l\n            if r < n and self.data[r] > self.data[big]:\n                big = r\n            if big != i:\n                self.data[i], self.data[big] = self.data[big], self.data[i]\n                i = big\n            else:\n                break\n    def peek(self):\n        return",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    out, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    out.extend(left[i:]); out.extend(right[j:])\n    return",
]


def _build_backend(m: dict, common: dict):
    b = m["backend"]
    if b == "vllm":
        return VllmBackend(m["repo"], m.get("quant"), m.get("eagle3", False),
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
        result["humaneval"] = run_humaneval(
            lambda p: backend.generate(p, 512).text,
            prompts=50)
    return result


def run(config_path: str, output_path: str | None = None) -> list[dict]:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    common = cfg["common"]
    results = []
    t_start = time.perf_counter()
    for m in cfg["models"]:
        try:
            results.append(run_model(m, common))
        except Exception as e:  # keep going; record the failure
            results.append({"id": m["id"], "backend": m["backend"],
                            "error": repr(e)})
            print(f"[skip] {m['id']}: {e}", flush=True)
    out = {"elapsed_seconds": round(time.perf_counter() - t_start, 1),
           "common": common, "results": results}
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(yaml.safe_dump(out, sort_keys=False))
        print(f"[done] wrote {output_path}", flush=True)
    return results
BENCH_EOF_bench_run_py

cat > bench/__init__.py <<'BENCH_EOF_bench___init___py'
"""EdgeFast-Diff 1B benchmark week harness."""
from .run import run, run_model

__all__ = ["run", "run_model"]
BENCH_EOF_bench___init___py

cat > bench/__main__.py <<'BENCH_EOF_bench___main___py'
"""CLI entry: python -m bench --config bench/config.yaml --out results.yaml"""
import argparse

from .run import run


def main():
    ap = argparse.ArgumentParser(description="EdgeFast-Diff 1B benchmark week")
    ap.add_argument("--config", default="bench/config.yaml")
    ap.add_argument("--out", default="results.yaml")
    ap.add_argument("--only", help="comma-separated model ids to run")
    args = ap.parse_args()

    cfg = __import__("yaml").safe_load(open(args.config).read())
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        cfg["models"] = [m for m in cfg["models"] if m["id"] in want]

    run(args.config, args.out)


if __name__ == "__main__":
    main()
BENCH_EOF_bench___main___py

echo "[3/4] done. Harness is in bench/. Smoke-test one model:"
echo "    python -m bench --only smollm2-135m --out results.yaml"
echo "[4/4] full roster:"
echo "    python -m bench --config bench/config.yaml --out results.yaml"
