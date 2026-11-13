"""Benchmark week orchestration: load roster, run each model, write JSON."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import (TransformersBackend, DllmBackend, Generation,
                       load_smoldlm)
from .humaneval import (DEFAULT_HUMANEVAL_N, EVAL_MODES, MAX_HUMANEVAL,
                        build_protocol, run_humaneval)

# Prompts for the speed harness: coding-style, batch 1, out_len tokens each.
_SPEED_PROMPTS = [
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return",
    "class MaxHeap:\n    def __init__(self):\n        self.data = []\n    def push(self, x):\n        self.data.append(x)\n        self._sift_up(len(self.data) - 1)\n    def pop(self):\n        top = self.data[0]\n        self.data[0] = self.data[-1]\n        self.data.pop()\n        self._sift_down(0)\n        return top\n    def _sift_up(self, i):\n        while i > 0:\n            p = (i - 1) // 2\n            if self.data[i] > self.data[p]:\n                self.data[i], self.data[p] = self.data[p], self.data[i]\n                i = p\n            else:\n                break\n    def _sift_down(self, i):\n        n = len(self.data)\n        while True:\n            l, r, big = 2*i+1, 2*i+2, i\n            if l < n and self.data[l] > self.data[big]:\n                big = l\n            if r < n and self.data[r] > self.data[big]:\n                big = r\n            if big != i:\n                self.data[i], self.data[big] = self.data[big], self.data[i]\n                i = big\n            else:\n                break\n    def peek(self):\n        return",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    out, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    out.extend(left[i:]); out.extend(right[j:])\n    return",
]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(config_path: str) -> dict:
    import torch
    config_bytes = Path(config_path).read_bytes()
    packages = {}
    for name in ("torch", "transformers", "datasets", "tokenizers",
                 "bitsandbytes", "pyyaml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_path": str(Path(config_path)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu,
        "cuda_version": torch.version.cuda,
        "packages": packages,
    }


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
    if m.get("speculative") or m.get("eagle3"):
        raise NotImplementedError(
            "speculative decoding is not wired; refusing to run plain int4 under a "
            "speculative-decoding label")
    if b == "vllm":
        raise NotImplementedError(
            "the vLLM backend is not implemented; use backend: hf explicitly")
    if b == "hf":
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
                            dtype=common["dtype"],
                            temperature=m.get("temperature", 0.0),
                            commit=m.get("commit", "confidence"),
                            kv_cache=m.get("kv_cache", False),
                            warm_batch=m.get("warm_batch", False))
    raise ValueError(f"unknown backend: {b}")


def _humaneval_spec(m: dict, common: dict) -> dict | None:
    """Resolve the eval protocol for a model, or None when eval is off.

    The prompt mode is explicit per model. Inferring it from a repo name
    (for example "Instruct") would silently mix protocols and make scores
    unauditable, so a missing or invalid ``eval_prompt`` is a hard error.
    """
    if not m.get("humaneval", common.get("humaneval", False)):
        return None
    mode = m.get("eval_prompt")
    if mode not in EVAL_MODES:
        raise ValueError(
            f"{m['id']} needs an explicit eval_prompt in {EVAL_MODES}; "
            "refusing to guess the evaluation protocol from the repo name")
    n = m.get("humaneval_n", common.get("humaneval_n",
                                            DEFAULT_HUMANEVAL_N))
    n = DEFAULT_HUMANEVAL_N if n is None else n
    if type(n) is not int or not 1 <= n <= MAX_HUMANEVAL:
        raise ValueError(
            f"{m['id']} humaneval_n must be 1..{MAX_HUMANEVAL}, got {n}")
    return {"mode": mode, "n": n}


def _timed_gen(backend, prompt: str, out_len: int, runs: int,
               seed: int = 0) -> dict:
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if out_len < 1:
        raise ValueError(f"out_len must be >= 1, got {out_len}")
    tps = []
    texts = []
    toks = []
    passes = []
    warm_passes = []
    blocks = []
    prefills = []
    prefill_kinds = []
    generation_seconds = []
    stop_reasons = []
    fill_counts = []
    fallback_commits = []
    token_ids = []
    for run_index in range(runs):
        _seed_all(seed + run_index)
        g: Generation = backend.generate(prompt, out_len)
        if g.num_tokens < 1:
            raise RuntimeError("backend emitted zero tokens")
        if g.num_tokens != out_len:
            raise RuntimeError(
                f"speed contract requires exactly {out_len} output tokens; "
                f"backend emitted {g.num_tokens} ({g.stop_reason})")
        tps.append(g.num_tokens / g.decode_seconds)
        texts.append(g.text)
        toks.append(g.num_tokens)
        passes.append(g.passes)
        warm_passes.append(g.warm_passes)
        blocks.append(g.blocks)
        prefills.append(g.prefill_seconds)
        prefill_kinds.append(g.prefill_kind)
        generation_seconds.append(g.decode_seconds)
        stop_reasons.append(g.stop_reason)
        fill_counts.append(g.fill_counts or [])
        fallback_commits.append(g.fallback_commits)
        token_ids.append(g.token_ids)
    out: dict = {
        "median_tps": round(statistics.median(tps), 3),
        "min_tps": round(min(tps), 3),
        "max_tps": round(max(tps), 3),
        "runs": runs,
        "sample_text": texts[0],
        "sample_text_preview": texts[0][:200],
        "sample_token_ids": token_ids[0],
        "tokens_per_run": toks,
        "tokens_emitted": sum(toks),
        "generation_seconds_per_run": [round(x, 6)
                                       for x in generation_seconds],
        "prefill_seconds_per_run": [round(x, 6) for x in prefills],
        "prefill_kinds": prefill_kinds,
        "stop_reasons": stop_reasons,
        "passes": sum(passes),
        "passes_per_run": passes,
        "warm_passes": sum(warm_passes),
        "warm_passes_per_run": warm_passes,
        "blocks": sum(blocks),
        "blocks_per_run": blocks,
        "fill_counts_per_run": fill_counts,
        "fallback_commits": sum(fallback_commits),
        "fallback_commits_per_run": fallback_commits,
    }
    if sum(passes) > 0:
        out["tokens_per_pass"] = round(sum(toks) / sum(passes), 3)
    if sum(passes) + sum(warm_passes) > 0:
        out["tokens_per_model_pass"] = round(
            sum(toks) / (sum(passes) + sum(warm_passes)), 3)
    return out


def _comparisons(results: list[dict]) -> list[dict]:
    """Build explicit token/speed comparisons requested by model config.

    `compare_to` never implies semantic equivalence. It merely makes the
    agreement (or disagreement) visible before anyone quotes a speed ratio.
    """
    by_id = {r.get("id"): r for r in results if "error" not in r}
    comparisons = []
    for candidate in results:
        baseline_id = candidate.get("config", {}).get("compare_to")
        baseline = by_id.get(baseline_id)
        if not baseline or "error" in candidate:
            continue
        a = baseline.get("speed", {}).get("sample_token_ids") or []
        b = candidate.get("speed", {}).get("sample_token_ids") or []
        denominator = max(len(a), len(b), 1)
        matches = sum(x == y for x, y in zip(a, b))
        base_tps = baseline["speed"]["median_tps"]
        candidate_tps = candidate["speed"]["median_tps"]
        comparisons.append({
            "baseline": baseline_id,
            "candidate": candidate.get("id"),
            "exact_token_match": a == b,
            "token_agreement": round(matches / denominator, 6),
            "matched_positions": matches,
            "baseline_tokens": len(a),
            "candidate_tokens": len(b),
            "median_tps_ratio": round(candidate_tps / base_tps, 6),
            "semantic_equivalence_claimed": bool(
                candidate.get("config", {}).get(
                    "semantic_equivalence_claimed", False)),
        })
    return comparisons


def run_model(m: dict, common: dict) -> dict:
    print(f"[run] {m['id']} ({m['backend']})", flush=True)
    # Validate the eval protocol before the expensive speed run: a missing
    # eval_prompt must not cost a full benchmark before failing.
    eval_spec = _humaneval_spec(m, common)
    seed = int(m.get("seed", common.get("seed", 0)))
    _seed_all(seed)
    backend = _build_backend(m, common)
    eval_protocol = None
    if eval_spec is not None:
        if isinstance(backend, TransformersBackend):
            eval_protocol = backend.evaluation_protocol(eval_spec["mode"])
        elif eval_spec["mode"] == "chat":
            raise ValueError("chat evaluation requires a chat-capable HF backend")
        else:
            eval_protocol = build_protocol(eval_spec["mode"])
    for _ in range(int(common.get("warmup", 1))):
        backend.warmup()
    # Warmup must not consume the random stream used by measured runs.
    _seed_all(seed)
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is None:
        # DllmBackend loads its tokenizer with the model; dllm's API cannot
        # be verified off-T4, so probe for it and degrade honestly
        tokenizer = getattr(getattr(backend, "_model", None), "tokenizer",
                            None)
    if tokenizer is None:
        raise RuntimeError(
            f"{m['id']} exposes no tokenizer; exact prompt_len cannot be verified")
    prompt, n_in = fit_prompt(_SPEED_PROMPTS, tokenizer,
                              common["prompt_len"])
    if n_in != common["prompt_len"]:
        raise RuntimeError(
            f"{m['id']} prompt has {n_in} tokens, expected "
            f"{common['prompt_len']}")
    print(f"[prompt] {m['id']}: {n_in} tokens in "
          f"(target {common['prompt_len']})", flush=True)
    speed = _timed_gen(backend, prompt, common["out_len"], common["runs"],
                       seed=seed)
    speed["prompt_tokens"] = n_in
    result: dict = {
        "id": m["id"],
        "backend": m["backend"],
        "repo": m.get("repo"),
        "note": m.get("note", ""),
        "config": dict(m),
        "seed": seed,
        "git_sha": _git("rev-parse", "HEAD"),
        "speed": speed,
    }
    if eval_spec is not None:
        print(f"[eval] HumanEval pass@1 for {m['id']} "
              f"(mode={eval_spec['mode']}, n={eval_spec['n']})", flush=True)
        try:
            def eval_generate(prompt: str) -> str:
                if isinstance(backend, TransformersBackend):
                    return backend.generate_eval(
                        prompt, eval_spec["mode"]).text
                if eval_spec["mode"] != "completion":
                    raise RuntimeError(
                        f"{m['id']} requests chat eval but "
                        f"{m['backend']} has no chat-template eval path")
                return backend.generate(prompt, 256).text

            result["humaneval"] = run_humaneval(
                eval_generate, prompts=eval_spec["n"],
                mode=eval_spec["mode"], protocol=eval_protocol)
        except Exception as e:  # speed result already in hand; don't discard it
            result["humaneval"] = {"error": repr(e),
                                   "protocol": eval_protocol}
            print(f"[skip] humaneval for {m['id']}: {e}", flush=True)
    return result


def _validate_common(common: dict) -> None:
    if common.get("batch") != 1:
        raise ValueError("benchmark contract requires batch: 1")
    for key in ("prompt_len", "out_len", "runs"):
        if int(common.get(key, 0)) < 1:
            raise ValueError(f"common.{key} must be >= 1")
    if int(common.get("warmup", 1)) < 0:
        raise ValueError("common.warmup must be >= 0")
    if common.get("humaneval_n") is not None:
        n = common["humaneval_n"]
        if type(n) is not int or not 1 <= n <= MAX_HUMANEVAL:
            raise ValueError(
                f"common.humaneval_n must be 1..{MAX_HUMANEVAL}, got {n}")


def run(config_path: str, output_path: str | None = None,
        models: list[dict] | None = None) -> list[dict]:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    common = cfg["common"]
    _validate_common(common)
    if models is None:
        models = cfg["models"]
    if not models:
        raise ValueError("no models selected")
    results = []
    t_start = time.perf_counter()
    provenance = _provenance(config_path)

    def _save():
        # Write after EVERY model: a hard crash late in the roster (e.g. the
        # 8B diffusion runs OOMing the process) must not lose earlier results.
        if output_path:
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            out = {"elapsed_seconds": round(time.perf_counter() - t_start, 1),
                   "provenance": provenance,
                   "common": common, "results": results,
                   "comparisons": _comparisons(results)}
            temp = target.with_name(target.name + ".part")
            try:
                temp.write_text(yaml.safe_dump(out, sort_keys=False))
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)

    for m in models:
        try:
            results.append(run_model(m, common))
        except Exception as e:  # keep going; record the failure
            results.append({"id": m["id"], "backend": m["backend"],
                            "config": dict(m),
                            "git_sha": provenance["git_sha"],
                            "error": repr(e)})
            print(f"[skip] {m['id']}: {e}", flush=True)
        _save()
    print(f"[done] wrote {output_path}", flush=True)
    return results
