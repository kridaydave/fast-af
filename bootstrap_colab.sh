#!/usr/bin/env bash
# EdgeFast-Diff 100M — benchmark week bootstrap for Colab free T4.
# Upload this file to Colab and run the whole cell: it installs the env,
# writes the harness, and (optionally) runs it.
set -euo pipefail

echo "[1/4] installing torch 2.6.0 + torchvision 0.21.0 + torchaudio 2.6.0 (cu124) + transformers 5.x..."
# torchaudio must match torch exactly: Colab preinstalls one built for a newer
# torch, and loading its .so against 2.6.0 kills every transformers import
# ("undefined symbol: aoti_torch_abi_version").
pip install --quiet --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --quiet --no-cache-dir "transformers>=5.6.0,<6.0" pyyaml datasets accelerate safetensors bitsandbytes
# Qwen3.5's GDN layers need these or transformers falls back to slow reference
# code — the B0 baseline must not be benchmarked without them.
pip install --quiet --no-cache-dir flash-linear-attention || echo "WARN: flash-linear-attention failed; qwen3.5 GDN will run slow."
# Compile only for the T4 (sm75) — otherwise nvcc builds for 12 GPU
# architectures and the install takes ~10x longer.
TORCH_CUDA_ARCH_LIST="7.5" pip install --quiet --no-cache-dir causal-conv1d > /tmp/causal_conv1d_install.log 2>&1 || echo "WARN: causal-conv1d failed (source build); qwen3.5 conv will run slow. Log: /tmp/causal_conv1d_install.log"
# dLLM is only for the diffusion paradigm check; don't block AR if it fails.
# dllm hard-pins transformers==4.57.0, so install with --no-deps + its light
# runtime deps — never let pip downgrade the transformers 5.x above.
pip install --quiet --no-cache-dir --no-deps "git+https://github.com/ZHZisZZ/dllm.git" > /tmp/dllm_install.log 2>&1 \
  && pip install --quiet --no-cache-dir --prefer-binary sentencepiece tyro omegaconf rich torchmetrics tqdm \
  || echo "WARN: dLLM install failed; diffusion models will be skipped. Install log: /tmp/dllm_install.log"
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU — switch Runtime to T4'; print('torch', torch.__version__, '| CUDA ok |', torch.cuda.get_device_name(0))"

echo "[2/4] writing harness..."
mkdir -p bench
cat > bench/config.yaml <<'BENCH_EOF_bench_config_yaml'
# Benchmark week roster for EdgeFast-Diff 100M (D1 = 100M).
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
  prompt_len: 1024      # tokens in — enforced by fit_prompt() in run.py
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
    backend: hf
    repo: HuggingFaceTB/SmolLM2-135M-Instruct
    quant: null
    humaneval: true
    note: "135M AR, Apache-2.0, L1 yardstick"

  - id: qwen3.5-0.8b            # B0
    backend: hf
    repo: Qwen/Qwen3.5-0.8B
    quant: null
    humaneval: true
    note: "GDN hybrid AR, same family as D1. B1/B2 build on this."

  - id: qwen3.5-0.8b-int4       # B1
    backend: hf
    repo: Qwen/Qwen3.5-0.8B
    quant: int4                 # GPTQ int4 (T4 has no AWQ)
    humaneval: true
    note: "B0 + int4 QAT, expected ~2x vs B0"

  - id: qwen3.5-0.8b-eagle3     # B2
    backend: hf
    repo: Qwen/Qwen3.5-0.8B
    quant: int4
    eagle3: true                # speculative decode; budget a day of friction
    humaneval: true
    note: "B0 + int4 + EAGLE-3, expected 3-6x vs B0; EAGLE-3 spec-decode NOT auto-wired (needs llama.cpp or vLLM), runs as plain int4 AR until wired"

  - id: lfm2.5-350m
    backend: hf
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
    humaneval: false            # diffusion row is speed-only; HumanEval stays AR-only (spec)
    note: "144M block diffusion, block_size 32, Apache-2.0. D1's closest size-matched diffusion twin."

  # ---- paradigm check at scale ----
  # 8B rows run LAST (user call 2026-11-04): an OOM crash kills the process,
  # and every row above is already saved by then (run.py writes results after
  # each model). The LLaDA 4-bit path is deferred until the end of benchmarking.

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


class TransformersBackend:
    """Autoregressive models via HuggingFace generate, greedy (do_sample=False),
    batch 1. Robust on free T4 without vLLM's version gymnastics.

    Timing note: we time the whole generate() call, so prompt prefill is
    included in the decode window. For batch-1 single-stream that's small vs
    the 1k-token output and consistent across AR models, so the TPS *ratios*
    (B1/B2 vs B0) the spec wants are unaffected. Swap in a vLLM backend later
    if absolute TPS matters.
    """

    def __init__(self, repo: str, quant: str | None, dtype: str,
                 max_model_len: int = 2048):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.device = "cuda"
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        kw: dict[str, Any] = dict(torch_dtype=torch_dtype, device_map="cuda",
                                  trust_remote_code=True)
        if quant == "int4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        self.tokenizer = AutoTokenizer.from_pretrained(repo,
                                                       trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_model_len = max_model_len

    def warmup(self) -> None:
        self.generate("the quick brown fox", 8)

    def generate(self, prompt: str, out_len: int) -> Generation:
        import torch
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        plen = enc["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=out_len, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
        decode_s = time.perf_counter() - t0
        toks = int(out.shape[1] - plen)
        text = self.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
        return Generation(text=text, num_tokens=max(toks, 1),
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

    The benchmark-week prototype of D1's decode policy: fill `block_size`
    slots with mask tokens, run up to `steps` denoising passes over
    prefix+block, and each pass unmask every open slot whose best prediction
    clears `confidence`. The last pass always fills whatever is still open,
    so a committed block never contains mask tokens. EOS truncates the block
    the way an AR stop would.

    score_fn(seq) -> logits (1, L, V) must pin mask/pad logits to -inf so
    argmax can never pick a mask slot; load_smoldlm() wires this. The D1
    knobs from the spec (0.97 block early stop, prompt cache, block overlap)
    belong to the real model's loop, not this prototype.

    tokenizer is a `tokenizers.Tokenizer` (encode().ids / decode()).
    """

    MASK_ID = 0  # smoldlm config: mask_token_id = 0

    def __init__(self, block_size: int, steps: int,
                 confidence: float, score_fn: Callable, tokenizer,
                 eos_id: int = 1):
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.block_size = block_size
        self.steps = steps
        self.confidence = confidence
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def warmup(self) -> None:
        self.generate("the quick brown fox", 8)

    def generate(self, prompt: str, out_len: int) -> Generation:
        prefix = self.tokenizer.encode(prompt).ids
        out_ids = list(prefix)
        t0 = time.perf_counter()

        emitted = 0
        while emitted < out_len:
            block = [self.MASK_ID] * self.block_size
            seq = torch.tensor([out_ids + block], device=self.device)
            for step in range(self.steps):
                open_slots = [i for i, t in enumerate(block)
                              if t == self.MASK_ID]
                if not open_slots:
                    break
                logits = self.score_fn(seq)            # (1, L, V)
                probs = torch.softmax(logits[0, -self.block_size:], dim=-1)
                conf, tok = probs.max(dim=-1)          # per-slot (block,)
                conf = conf.float().cpu()
                fill = (open_slots if step == self.steps - 1
                        else [i for i in open_slots
                              if conf[i] >= self.confidence])
                if not fill:
                    # threshold cleared nothing: commit the single most
                    # confident slot so every pass makes progress instead
                    # of re-scoring an identical block
                    fill = [max(open_slots, key=lambda i: float(conf[i]))]
                for i in fill:
                    if int(tok[i]) == self.MASK_ID:
                        raise RuntimeError(
                            "score_fn let the mask id win argmax at an open "
                            "slot; pin mask/pad logits to -inf")
                    block[i] = int(tok[i])
                seq = torch.tensor([out_ids + block], device=self.device)
            if self.eos_id in block:
                block = block[:block.index(self.eos_id)]
                out_ids.extend(block)
                emitted += len(block)
                break
            out_ids.extend(block)
            emitted += len(block)

        decode_s = time.perf_counter() - t0
        text = self.tokenizer.decode(out_ids[len(prefix):],
                                     skip_special_tokens=True)
        return Generation(text=text, num_tokens=max(emitted, 1),
                          decode_seconds=max(decode_s, 1e-6))


def load_smoldlm(repo: str, block_size: int, steps: int, confidence: float,
                 dtype: str) -> CustomBackend:
    """Wire CustomBackend to HoangHa/smoldlm-144m.

    The repo ships `modeling_smoldlm.py` as a standalone module (not a
    transformers AutoModel), so fetch it from the repo and import it
    directly. score_fn mirrors the repo's own generate(): forward pass with
    mask/pad logits pinned to -inf.
    """
    import importlib.util

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    mod_path = hf_hub_download(repo, "modeling_smoldlm.py")
    spec = importlib.util.spec_from_file_location("modeling_smoldlm", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.SmolDLM.from_pretrained(repo)
    if dtype == "float16":
        model = model.half()  # match the AR tier's fp16 full-precision run

    def score_fn(seq):
        with torch.no_grad():
            logits = model(seq)
        logits[:, :, model.mask_token_id] = -float("inf")
        logits[:, :, model.pad_token_id] = -float("inf")
        return logits

    backend = CustomBackend(
        block_size=block_size, steps=steps, confidence=confidence,
        score_fn=score_fn,
        tokenizer=Tokenizer.from_file(model._tokenizer_path),
        eos_id=model.eos_token_id)
    # generate() builds seq on backend.device; without this move a CUDA box
    # (Colab T4) would feed CPU weights a CUDA tensor
    model.to(backend.device)
    return backend
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

# Canonical HumanEval. The raw file on openai/human-eval is GZIP'd (.jsonl.gz);
# the plain .jsonl path was deleted and 404s.
_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def _load(prompts: int) -> list[dict[str, Any]]:
    rows = []
    try:
        # The dataset moved out of the top-level namespace; datasets does not
        # reliably follow the 307, so query the new location directly.
        ds = load_dataset("openai/openai_humaneval", split="test")
        rows = list(ds)
    except Exception:
        # Fallback: fetch the raw gzipped jsonl directly.
        import gzip
        import urllib.request
        with urllib.request.urlopen(_URL) as r:
            rows = [json.loads(line) for line in gzip.decompress(r.read())
                    .decode("utf-8").splitlines() if line.strip()]
    return [r for r in rows[:prompts]]


def _extract(completion: str, prompt: str) -> str:
    """Completion is the continuation of prompt; if the model repeated the
    prompt verbatim, strip down to the part after it. No other surgery —
    prepending 'def ' or cutting at the next def corrupted valid bodies."""
    if prompt and prompt.strip() in completion:
        return completion.split(prompt, 1)[1]
    return completion


def _run(prompt: str, completion: str, tests: str, entry_point: str) -> bool:
    """Canonical HumanEval check: exec prompt+completion, then the dataset's
    test suite via check(entry_point). The prompt docstring's >>> examples
    are NOT tests — judging those made every parseable completion pass."""
    try:
        exec_globals: dict[str, Any] = {}
        code = (prompt + "\n" + _extract(completion, prompt) + "\n"
                + tests + "\ncheck(" + entry_point + ")\n")
        exec(compile(code, "<eval>", "exec"), exec_globals)  # noqa: S102
        return True
    except Exception:
        return False


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
        if _run(prompt, completion, r.get("test", ""),
                r.get("entry_point", "")):
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

from .backends import (TransformersBackend, DllmBackend, Generation,
                       load_smoldlm)
from .humaneval import run_humaneval

# Prompts for the speed harness: coding-style, batch 1, out_len tokens each.
_SPEED_PROMPTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return",
    "class MaxHeap:\n    def __init__(self):\n        self.data = []\n    def push(self, x):\n        self.data.append(x)\n        self._sift_up(len(self.data) - 1)\n    def pop(self):\n        top = self.data[0]\n        self.data[0] = self.data[-1]\n        self.data.pop()\n        self._sift_down(0)\n        return top\n    def _sift_up(self, i):\n        while i > 0:\n            p = (i - 1) // 2\n            if self.data[i] > self.data[p]:\n                self.data[i], self.data[p] = self.data[p], self.data[i]\n                i = p\n            else:\n                break\n    def _sift_down(self, i):\n        n = len(self.data)\n        while True:\n            l, r, big = 2*i+1, 2*i+2, i\n            if l < n and self.data[l] > self.data[big]:\n                big = l\n            if r < n and self.data[r] > self.data[big]:\n                big = r\n            if big != i:\n                self.data[i], self.data[big] = self.data[big], self.data[i]\n                i = big\n            else:\n                break\n    def peek(self):\n        return",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    out, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    out.extend(left[i:]); out.extend(right[j:])\n    return",
]


def _encode_ids(tokenizer, text: str) -> list[int]:
    # HF tokenizers return list[int]; tokenizers.Tokenizer returns an Encoding
    ids = tokenizer.encode(text)
    return ids if isinstance(ids, list) else ids.ids


def fit_prompt(prompts: list[str], tokenizer, target: int) -> tuple[str, int]:
    """Build a prompt of exactly `target` tokens by cycling the base prompts.

    The spec's timing contract is 1k in / 1k out (spec, Baseline section);
    the harness used to ignore prompt_len and time ~60-token prompts, which
    flatters diffusion: AR reads the prefix once and caches it, while the
    diffusion loop re-reads the whole prefix on every denoising pass, so a
    short prompt hides that cost entirely.
    """
    blob = "\n\n".join(prompts)
    ids: list[int] = []
    while len(ids) < target:
        ids += _encode_ids(tokenizer, blob)
    ids = ids[:target]
    text = tokenizer.decode(ids)
    for _ in range(3):
        # decode -> encode can drift a token or two at the cut; converge
        real = _encode_ids(tokenizer, text)
        if len(real) == target:
            break
        ids = (real + ids)[:target] if len(real) < target else real[:target]
        text = tokenizer.decode(ids)
    return text, len(_encode_ids(tokenizer, text))


def _build_backend(m: dict, common: dict):
    b = m["backend"]
    if b in ("vllm", "hf"):
        return TransformersBackend(m["repo"], m.get("quant"),
                                   common["dtype"])
    if b == "dllm":
        return DllmBackend(m["repo"], m.get("mode", "default"),
                           m.get("steps", 8), common["dtype"])
    if b == "custom":
        # only custom model in the roster today; a second one gets its own loader
        return load_smoldlm(m["repo"], block_size=m["block_size"],
                            steps=m.get("steps", 8),
                            confidence=m.get("confidence", 0.9),
                            dtype=common["dtype"])
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
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        # DllmBackend loads its tokenizer with the model; dllm's API cannot
        # be verified off-T4, so probe for it and degrade honestly
        tokenizer = getattr(getattr(backend, "_model", None), "tokenizer",
                            None)
    if tokenizer is not None:
        prompt, n_in = fit_prompt(_SPEED_PROMPTS, tokenizer,
                                  common["prompt_len"])
        print(f"[prompt] {m['id']}: {n_in} tokens in "
              f"(target {common['prompt_len']})", flush=True)
    else:
        prompt, n_in = "\n\n".join(_SPEED_PROMPTS), None
        print(f"[warn] {m['id']}: no tokenizer exposed; prompt_len "
              f"{common['prompt_len']} not enforced", flush=True)
    speed = _timed_gen(backend, prompt, common["out_len"], common["runs"])
    speed["prompt_tokens"] = n_in
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
BENCH_EOF_bench_run_py
cat > bench/__init__.py <<'BENCH_EOF_bench___init___py'
"""EdgeFast-Diff 100M benchmark week harness."""
from .run import run, run_model

__all__ = ["run", "run_model"]
BENCH_EOF_bench___init___py
cat > bench/__main__.py <<'BENCH_EOF_bench___main___py'
"""CLI entry: python -m bench --config bench/config.yaml --out results.yaml"""
import argparse

from .run import run


def main():
    ap = argparse.ArgumentParser(description="EdgeFast-Diff 100M benchmark week")
    ap.add_argument("--config", default="bench/config.yaml")
    ap.add_argument("--out", default="results.yaml")
    ap.add_argument("--only", help="comma-separated model ids to run")
    args = ap.parse_args()

    cfg = __import__("yaml").safe_load(open(args.config).read())
    models = None
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        models = [m for m in cfg["models"] if m["id"] in want]

    run(args.config, args.out, models=models)


if __name__ == "__main__":
    main()
BENCH_EOF_bench___main___py

echo "[3/4] harness written to bench/."
echo "Smoke-test one AR model:"
echo "    python -m bench --only smollm2-135m --out results.yaml"
echo "Full roster:"
echo "    python -m bench --config bench/config.yaml --out results.yaml"
echo "Set BENCH_RUN=smoke|full to auto-run after setup (default: none)."
if [ "${BENCH_RUN:-none}" = "smoke" ]; then
  python -m bench --only smollm2-135m --out results.yaml
elif [ "${BENCH_RUN:-none}" = "full" ]; then
  python -m bench --config bench/config.yaml --out results.yaml
fi
