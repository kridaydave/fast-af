"""Capture historical/current smoldlm tokens and compare G1 artifacts.

Run directly (python bench/token_baseline.py) for a dependency-free comparator.
Capture imports the GPU/model stack lazily. It never changes the checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "g1-token-baseline/v1"
THRESHOLD = 0.99
ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"[0-9a-f]{40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(doc, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)


def backend_source(ref):
    sha = git("rev-parse", "--verify", ref + "^{commit}").decode().strip()
    return sha, git("show", sha + ":bench/backends.py")


def _ids(value):
    return isinstance(value, list) and all(type(t) is int and t >= 0 for t in value)


def validate(doc, role):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("role") != role:
        raise ValueError(f"{role}: invalid schema/role")
    meta = doc.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError(f"{role}: missing metadata")
    required = {"model", "model_revision", "tokenizer_revision", "assets", "prompt",
                "prompt_token_ids", "output_length", "seed", "policy", "source_sha",
                "source_sha256", "environment", "stop_reason"}
    if not required <= meta.keys():
        raise ValueError(f"{role}: missing metadata: {sorted(required - meta.keys())}")
    for name in ("model_revision", "tokenizer_revision", "source_sha"):
        if not isinstance(meta[name], str) or not SHA.fullmatch(meta[name]):
            raise ValueError(f"{role}: {name} must be an immutable 40-character SHA")
    if not isinstance(meta["source_sha256"], str) or not HASH.fullmatch(meta["source_sha256"]):
        raise ValueError(f"{role}: missing source hash")
    assets = meta["assets"]
    if not isinstance(assets, dict) or set(assets) != {"modeling_smoldlm.py", "config.json", "model.safetensors", "tokenizer.json"}:
        raise ValueError(f"{role}: missing model/tokenizer asset hashes")
    if any(not isinstance(v, str) or not HASH.fullmatch(v) for v in assets.values()):
        raise ValueError(f"{role}: invalid asset hash")
    if not isinstance(meta["model"], str) or not meta["model"] or not isinstance(meta["prompt"], str) or not meta["prompt"]:
        raise ValueError(f"{role}: missing model/prompt")
    if not _ids(meta["prompt_token_ids"]) or not meta["prompt_token_ids"]:
        raise ValueError(f"{role}: missing prompt token IDs")
    if type(meta["output_length"]) is not int or meta["output_length"] < 1 or type(meta["seed"]) is not int or meta["seed"] < 0:
        raise ValueError(f"{role}: invalid output_length/seed")
    policy = meta["policy"]
    if not isinstance(policy, dict) or set(policy) != {"block_size", "steps", "confidence", "temperature", "commit", "kv_cache", "dtype"}:
        raise ValueError(f"{role}: missing decode policy")
    if any(type(policy[k]) is not int or policy[k] < 1 for k in ("block_size", "steps")):
        raise ValueError(f"{role}: invalid block_size/steps")
    confidence = policy["confidence"]
    if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"{role}: invalid confidence")
    if type(policy["temperature"]) not in (float, int) or policy["temperature"] != 0 or policy["commit"] != "confidence" or policy["kv_cache"] is not False or policy["dtype"] not in ("float16", "float32"):
        raise ValueError(f"{role}: G1 requires uncached, temperature 0, confidence commits")
    if meta["output_length"] % policy["block_size"]:
        raise ValueError(f"{role}: output length must be a multiple of block_size")
    env = meta["environment"]
    if not isinstance(env, dict) or not {"python", "torch", "tokenizers", "safetensors", "huggingface_hub", "device", "gpu", "cuda"} <= env.keys():
        raise ValueError(f"{role}: missing environment provenance")
    if any(not isinstance(env[k], str) or not env[k] for k in ("python", "torch", "tokenizers", "safetensors", "huggingface_hub", "device")):
        raise ValueError(f"{role}: invalid environment provenance")
    tokens = doc.get("token_ids")
    if not _ids(tokens) or len(tokens) != meta["output_length"] or meta["stop_reason"] != "max_new_tokens":
        raise ValueError(f"{role}: incomplete/invalid output tokens or stop reason")
    return meta


def compare(baseline, candidate):
    a, b = validate(baseline, "baseline"), validate(candidate, "candidate")
    keys = ("model", "model_revision", "tokenizer_revision", "assets", "prompt", "prompt_token_ids", "output_length", "seed", "policy", "environment")
    differences = [k for k in keys if a[k] != b[k]]
    if differences:
        raise ValueError("incompatible metadata: " + ", ".join(differences))
    if a["source_sha256"] == b["source_sha256"]:
        raise ValueError("baseline and candidate must use different backend sources")
    x, y = baseline["token_ids"], candidate["token_ids"]
    mismatches = [i for i, (u, v) in enumerate(zip(x, y)) if u != v]
    matches = len(x) - len(mismatches)
    return {"schema": "g1-comparison/v1", "baseline_source_sha": a["source_sha"],
            "candidate_source_sha": b["source_sha"], "matching_positions": matches,
            "total_positions": len(x), "mismatch_positions": mismatches,
            "agreement": matches / len(x), "threshold": THRESHOLD,
            "passed": matches * 100 >= len(x) * 99}


class RecordingTokenizer:
    """The historical Generation lacks IDs; observe decode's original IDs."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generated = None

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, ids, **kwargs):
        self.generated = list(ids)
        return self.tokenizer.decode(ids, **kwargs)


def load_backend(source, sha, score_fn, tokenizer, eos_id, args):
    # Execute the exact committed module; no source rewriting or checkout switch.
    module = types.ModuleType("g1_backend_" + sha)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, sha + ":bench/backends.py", "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    kwargs = dict(block_size=args.block_size, steps=args.steps,
                  confidence=args.confidence, score_fn=score_fn,
                  tokenizer=tokenizer, eos_id=eos_id)
    parameters = inspect.signature(module.CustomBackend).parameters
    for key, value in {"temperature": 0.0, "commit": "confidence", "kv_cache": False}.items():
        if key in parameters:
            kwargs[key] = value
    return module.CustomBackend(**kwargs)


def capture(args):
    if not SHA.fullmatch(args.model_revision) or not SHA.fullmatch(args.tokenizer_revision or args.model_revision):
        raise ValueError("model/tokenizer revisions must be immutable 40-character SHAs")
    if args.output_length < 1 or args.block_size < 1 or args.steps < 1 or args.output_length % args.block_size or args.seed < 0 or not 0 <= args.confidence <= 1:
        raise ValueError("use positive steps/block_size/output_length, whole output blocks, a nonnegative seed, and confidence in [0,1]")
    if not args.prompt:
        raise ValueError("prompt must not be empty")
    source_sha, source = backend_source(args.source_ref)
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from tokenizers import Tokenizer

    tokenizer_revision = args.tokenizer_revision or args.model_revision
    files = {name: Path(hf_hub_download(args.model, name, revision=(tokenizer_revision if name == "tokenizer.json" else args.model_revision)))
             for name in ("modeling_smoldlm.py", "config.json", "model.safetensors", "tokenizer.json")}
    module_spec = importlib.util.spec_from_file_location("g1_smoldlm", files["modeling_smoldlm.py"])
    model_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(model_module)
    model = model_module.SmolDLM(json.loads(files["config.json"].read_text()))
    incompatible = model.load_state_dict(load_file(str(files["model.safetensors"])), strict=False)
    if set(incompatible.missing_keys) - {"lm_head.weight"} or incompatible.unexpected_keys:
        raise ValueError(f"incompatible model weights: {incompatible}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=getattr(torch, args.dtype)).eval()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = RecordingTokenizer(Tokenizer.from_file(str(files["tokenizer.json"])))

    def score_fn(seq):
        with torch.no_grad():
            logits = model(seq)
        logits[:, :, model.mask_token_id] = -float("inf")
        logits[:, :, model.pad_token_id] = -float("inf")
        return logits

    backend = load_backend(source, source_sha, score_fn, tokenizer, model.eos_token_id, args)
    backend.device = device
    generation = backend.generate(args.prompt, args.output_length)
    tokens = tokenizer.generated
    policy = {"block_size": args.block_size, "steps": args.steps, "confidence": args.confidence,
              "temperature": 0.0, "commit": "confidence", "kv_cache": False, "dtype": args.dtype}
    env = {"python": platform.python_version(), "torch": str(torch.__version__), "device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "cuda": torch.version.cuda}
    for name in ("tokenizers", "safetensors", "huggingface_hub"):
        env[name] = importlib.metadata.version(name)
    doc = {"schema": SCHEMA, "role": args.role, "metadata": {
        "model": args.model, "model_revision": args.model_revision, "tokenizer_revision": tokenizer_revision,
        "assets": {name: file_digest(path) for name, path in files.items()},
        "prompt": args.prompt, "prompt_token_ids": tokenizer.encode(args.prompt).ids,
        "output_length": args.output_length, "seed": args.seed, "policy": policy,
        "source_sha": source_sha, "source_sha256": digest(source), "environment": env,
        "stop_reason": getattr(generation, "stop_reason", "max_new_tokens" if tokens and len(tokens) == args.output_length else "eos")},
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "token_ids": tokens}
    validate(doc, args.role)
    write_json(args.out, doc)
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="capture exact committed smoldlm sampler output")
    cap.add_argument("--role", choices=("baseline", "candidate"), required=True)
    cap.add_argument("--source-ref", required=True, help="historical baseline: b741770^; candidate: HEAD")
    cap.add_argument("--model", default="HoangHa/smoldlm-144m")
    cap.add_argument("--model-revision", required=True)
    cap.add_argument("--tokenizer-revision", help="defaults to model revision")
    prompt = cap.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    cap.add_argument("--output-length", type=int, default=1024)
    cap.add_argument("--block-size", type=int, default=32)
    cap.add_argument("--steps", type=int, default=8)
    cap.add_argument("--confidence", type=float, default=0.9)
    cap.add_argument("--seed", type=int, default=0)
    cap.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    cap.add_argument("--device", choices=("cpu", "cuda"))
    cap.add_argument("--out", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("baseline")
    cmp.add_argument("candidate")
    cmp.add_argument("--json-out")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            if args.prompt_file:
                args.prompt = Path(args.prompt_file).read_text()
            capture(args)
            print(f"Captured {args.role} to {args.out}")
            return 0
        if args.json_out and Path(args.json_out).resolve() in {Path(args.baseline).resolve(), Path(args.candidate).resolve()}:
            # Do not let the error handler overwrite an input artifact either.
            args.json_out = None
            raise ValueError("comparison report path must differ from input artifacts")
        result = compare(json.loads(Path(args.baseline).read_text()), json.loads(Path(args.candidate).read_text()))
        # Tie a report to the exact input artifact bytes, including provenance.
        result["baseline_sha256"] = digest(Path(args.baseline).read_bytes())
        result["candidate_sha256"] = digest(Path(args.candidate).read_bytes())
        if args.json_out:
            write_json(args.json_out, result)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    except (ValueError, OSError, TypeError, subprocess.CalledProcessError) as error:
        result = {"error": str(error), "passed": False}
        if args.command == "compare" and args.json_out:
            write_json(args.json_out, result)
        print(json.dumps(result), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
