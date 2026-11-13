"""G1 compatibility, threshold boundary, CLI, and historical capture tests."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench/token_baseline.py"
spec = importlib.util.spec_from_file_location("token_baseline", SCRIPT)
g1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1)


def artifacts():
    metadata = {
        "model": "HoangHa/smoldlm-144m", "model_revision": "a" * 40,
        "tokenizer_revision": "a" * 40,
        "assets": {name: "b" * 64 for name in ("modeling_smoldlm.py", "config.json", "model.safetensors", "tokenizer.json")},
        "prompt": "test", "prompt_token_ids": [10, 11], "output_length": 100,
        "seed": 0, "policy": {"block_size": 10, "steps": 8, "confidence": 0.9,
                               "temperature": 0.0, "commit": "confidence", "kv_cache": False, "dtype": "float32"},
        "source_sha": "c" * 40, "source_sha256": "c" * 64,
        "environment": {"python": "3", "torch": "2", "tokenizers": "1", "safetensors": "1",
                        "huggingface_hub": "1", "device": "cpu", "gpu": None, "cuda": None},
        "stop_reason": "max_new_tokens",
    }
    baseline = {"schema": g1.SCHEMA, "role": "baseline", "metadata": metadata, "token_ids": list(range(100))}
    candidate = deepcopy(baseline)
    candidate["role"] = "candidate"
    candidate["metadata"]["source_sha"] = "d" * 40
    candidate["metadata"]["source_sha256"] = "d" * 64
    return baseline, candidate


def test_threshold_boundary():
    a, b = artifacts()
    assert g1.compare(a, b)["agreement"] == 1
    b["token_ids"][0] = 999
    report = g1.compare(a, b)
    assert report["passed"] and report["matching_positions"] == 99
    assert report["mismatch_positions"] == [0]
    b["token_ids"][1] = 999
    assert not g1.compare(a, b)["passed"]


@pytest.mark.parametrize("key", ["model", "model_revision", "tokenizer_revision", "prompt_token_ids", "output_length", "seed", "policy", "source_sha", "source_sha256", "assets", "environment"])
def test_rejects_missing_even_when_both_artifacts_omit_field(key):
    a, b = artifacts()
    del a["metadata"][key]
    del b["metadata"][key]
    with pytest.raises(ValueError, match="missing metadata"):
        g1.compare(a, b)


@pytest.mark.parametrize("key,value", [("seed", 1), ("prompt", "other"), ("prompt_token_ids", [12]), ("model_revision", "e" * 40)])
def test_rejects_incompatible_fixture(key, value):
    a, b = artifacts()
    b["metadata"][key] = value
    with pytest.raises(ValueError, match="incompatible metadata"):
        g1.compare(a, b)


@pytest.mark.parametrize("tokens", [[], [1], [True] * 100, ["1"] * 100, [-1] * 100])
def test_rejects_incomplete_or_malformed_tokens(tokens):
    a, b = artifacts()
    b["token_ids"] = tokens
    with pytest.raises(ValueError):
        g1.compare(a, b)


def test_rejects_policy_and_same_source():
    a, b = artifacts()
    b["metadata"]["policy"]["kv_cache"] = True
    with pytest.raises(ValueError, match="G1 requires"):
        g1.compare(a, b)
    a, b = artifacts()
    b["metadata"]["source_sha256"] = a["metadata"]["source_sha256"]
    with pytest.raises(ValueError, match="different backend sources"):
        g1.compare(a, b)


def test_cli_is_standard_library_only_and_records_failure(tmp_path):
    a, b = artifacts()
    left, right, report = [tmp_path / name for name in ("baseline.json", "candidate.json", "report.json")]
    g1.write_json(left, a)
    g1.write_json(right, b)
    command = [sys.executable, "-S", str(SCRIPT), "compare", str(left), str(right), "--json-out", str(report)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    b["token_ids"][:2] = [999, 999]
    g1.write_json(right, b)
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert json.loads(report.read_text())["agreement"] == 0.98
    right.write_text("null")
    assert subprocess.run(command, capture_output=True).returncode == 2
    assert not json.loads(report.read_text())["passed"]


def test_original_historical_and_current_samplers_capture_raw_ids():
    import torch

    class Tokenizer:
        def encode(self, text):
            return SimpleNamespace(ids=[10, 11])

        def decode(self, ids, **kwargs):
            return "decoded"

    def score(seq):
        logits = torch.full((1, seq.shape[1], 32), -20.0)
        logits[:, :, 7] = 20.0
        return logits

    for ref in ("b741770^", "HEAD"):
        sha, source = g1.backend_source(ref)
        tokenizer = g1.RecordingTokenizer(Tokenizer())
        backend = g1.load_backend(source, sha, score, tokenizer, 1,
                                  SimpleNamespace(block_size=4, steps=2, confidence=0.9))
        backend.device = "cpu"
        generation = backend.generate("prompt", 8)
        assert generation.num_tokens == 8
        assert tokenizer.generated == [7] * 8


def test_capture_rejects_mutable_revision_before_importing_runtime():
    with pytest.raises(ValueError, match="immutable"):
        g1.capture(SimpleNamespace(model_revision="main", tokenizer_revision=None))


def test_capture_pins_all_assets_and_compares_historical_to_current(tmp_path, monkeypatch):
    import torch
    import types

    model_source = b'''import torch
class SmolDLM(torch.nn.Module):
    mask_token_id, eos_token_id, pad_token_id = 0, 1, 2
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
    def forward(self, seq):
        logits = torch.full((1, seq.shape[1], 32), -20., dtype=self.weight.dtype, device=seq.device)
        logits[:, :, 7] = 20.
        return logits
'''
    paths = {}
    for name, content in {"modeling_smoldlm.py": model_source, "config.json": b"{}", "model.safetensors": b"weights", "tokenizer.json": b"tokenizer"}.items():
        paths[name] = tmp_path / name
        paths[name].write_bytes(content)
    calls = []

    def download(repo, name, revision):
        calls.append((repo, name, revision))
        return str(paths[name])

    class Tokenizer:
        @staticmethod
        def from_file(path):
            return Tokenizer()

        def encode(self, text):
            return SimpleNamespace(ids=[10, 11])

        def decode(self, ids, **kwargs):
            return "decoded"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, "safetensors.torch", types.SimpleNamespace(load_file=lambda _: {"weight": torch.zeros(1)}))
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=Tokenizer))
    monkeypatch.setattr(g1.importlib.metadata, "version", lambda name: "test-version")
    docs = []
    for role, ref in (("baseline", "b741770^"), ("candidate", "HEAD")):
        args = SimpleNamespace(model="test/model", model_revision="a" * 40, tokenizer_revision="b" * 40,
                               source_ref=ref, role=role, prompt="test", output_length=8, block_size=4,
                               steps=2, seed=0, confidence=0.9, dtype="float32", device="cpu", out=tmp_path / (role + ".json"))
        docs.append(g1.capture(args))
        assert json.loads(args.out.read_text())["token_ids"] == [7] * 8
    assert g1.compare(*docs)["passed"]
    assert len(calls) == 8
    assert all(revision == ("b" * 40 if name == "tokenizer.json" else "a" * 40) for _, name, revision in calls)


def test_report_cannot_overwrite_baseline(tmp_path):
    a, b = artifacts()
    baseline, candidate = tmp_path / "base.json", tmp_path / "candidate.json"
    g1.write_json(baseline, a)
    g1.write_json(candidate, b)
    original = baseline.read_bytes()
    assert g1.main(["compare", str(baseline), str(candidate), "--json-out", str(baseline)]) == 2
    assert baseline.read_bytes() == original
