"""Build immutable task bundles and grade recorded completions, without a model dependency."""
from collections import Counter
import json
import math
from pathlib import Path
import time

from scripts.prepare_corpus import (
    SubstringIndex, benchmark_index, canonical, destination, digest, filehash,
    grams, norm, records, write_manifest, SPECIAL,
)
from .catalog import VERSION, generate, structure_hash
from .sandbox import MAX_CODE, WORKER, probe, verify

VERIFIER = "bubblewrap-json/v1"
TASK_FILES = ("private/tasks.jsonl", "private/audit.jsonl", "prompts/train.jsonl",
              "prompts/development.jsonl", "references-train.jsonl")


def implementation_hashes():
    return {name: filehash(Path(__file__).with_name(name))
            for name in ("catalog.py", "pipeline.py", "sandbox.py", "worker.py")}


class Exclusions:
    def __init__(self, config, base, smoke):
        self.ngrams, fragments, self.provenance = benchmark_index(config, base, smoke)
        self.substrings = SubstringIndex(fragments)

    def matches(self, text):
        return bool(grams(text) & self.ngrams) or self.substrings.matches(norm(text))


def jsonl(path, rows):
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def load_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    fingerprint = manifest.pop("fingerprint", None)
    if fingerprint != digest(canonical(manifest).encode()) or manifest.get("schema") != "python-tasks/v1":
        raise ValueError("invalid task manifest fingerprint/schema")
    expected = set(TASK_FILES) | ({"benchmarks.json"} if manifest["benchmarks"] else set())
    expected |= {f"private/benchmarks/{name}.jsonl" for name in manifest["benchmarks"]}
    if manifest["tokenizer"]:
        expected.add("tokenizer.json")
    if manifest["reference_review"] == "approved":
        expected.add("reference-review.json")
    if set(manifest["files"]) != expected:
        raise ValueError("unexpected bundle file list")
    for name, sha in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or filehash(path) != sha:
            raise ValueError(f"bundle file hash/path mismatch: {name}")
    return manifest


def public_task(task):
    return {k: task[k] for k in ("schema", "task_id", "family", "split", "description", "starter")}


def example(task, completion, smoke):
    prompt = task["description"] + "\n\n" + task["starter"]
    text = prompt + "\n# Solution\n" + completion
    return {"task_id": task["task_id"], "family": task["family"], "split": "train",
            "prompt": prompt, "completion": completion, "text": text, "bucket": "code",
            "subtype": "verified_task", "solution_start_char": len(prompt + "\n# Solution\n"),
            "smoke_only": smoke, "provenance": task["provenance"]}


def build(output, variants=2, seed=17, benchmarks=None, smoke=False,
          tokenizer=None, tokenizer_manifest=None, max_tokens=1024, review_file=None):
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not smoke and (not benchmarks or not tokenizer or not tokenizer_manifest):
        raise ValueError("production needs benchmarks, tokenizer and tokenizer-manifest; use --smoke for plumbing")
    config_path = Path(benchmarks) if benchmarks else None
    config = json.loads(config_path.read_text()) if config_path else {}
    exclusions = Exclusions(config, config_path.parent if config_path else Path.cwd(), smoke)
    catalog_hash = filehash(Path(__file__).with_name("catalog.py"))
    review = json.loads(Path(review_file).read_text()) if review_file else None
    if review is not None and (review.get("catalog_sha256") != catalog_hash or review.get("approved") is not True
                               or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        raise ValueError("review must approve the current catalog hash and name its reviewer")
    encode = None
    tokenizer_evidence = None
    if tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(tokenizer))
        if tok.truncation or tok.padding:
            raise ValueError("tokenizer must not silently truncate or pad")
        if not smoke and (tok.get_vocab_size() != 32000 or [tok.token_to_id(s) for s in SPECIAL] != list(range(5))):
            raise ValueError("production requires the frozen 32k tokenizer with project control IDs")
        if tokenizer_manifest:
            meta = json.loads(Path(tokenizer_manifest).read_text())
            fingerprint = meta.pop("fingerprint", None)
            if fingerprint != digest(canonical(meta).encode()) or meta.get("tokenizer_sha256") != filehash(tokenizer):
                raise ValueError("tokenizer manifest/hash mismatch")
            if not smoke and meta.get("smoke_only") is not False:
                raise ValueError("production needs a non-smoke tokenizer")
        encode = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
        tokenizer_evidence = {"sha256": filehash(tokenizer), "max_tokens": max_tokens,
                              "manifest_sha256": filehash(tokenizer_manifest) if tokenizer_manifest else None}
    probe()
    tasks, audits, structures = [], [], {}
    excluded = Counter()
    seen = set()
    for task in generate(variants, seed):
        text = example(task, task["reference"], smoke)["text"]
        if exclusions.matches(text):
            excluded["benchmark_overlap"] += 1
            continue
        if encode and encode(text) > max_tokens:
            excluded["context_limit"] += 1
            continue
        identity = digest(text.encode())
        if identity in seen:
            raise ValueError("duplicate task/solution example")
        seen.add(identity)
        owner = structures.setdefault(task["structure_id"], task["family"])
        if owner != task["family"]:
            raise ValueError("solution structure crosses family boundary")
        if verify(task["reference"], task)["status"] != "passed":
            raise ValueError(f"reference failed verification: {task['family']}")
        statuses = [verify(mutant, task)["status"] for mutant in task["mutants"]]
        if any(status != "wrong_answer" for status in statuses):
            raise ValueError(f"verifier did not reject incorrect solutions by answer: {task['family']}")
        task["reference_sha256"] = digest(task["reference"].encode())
        task["verifier"] = VERIFIER
        task["provenance"]["generator_sha256"] = catalog_hash
        task["provenance"]["reference_review"] = "approved" if review else "pending-human-review"
        tasks.append(task)
        audits.append({"task_id": task["task_id"], "reference": "passed", "cases": len(task["cases"]),
                       "mutants": statuses, "verifier": VERIFIER})
    counts = Counter(task["split"] for task in tasks)
    if set(counts) != {"train", "development", "evaluation"}:
        raise ValueError("filtering left an empty split")
    with destination(output) as out:
        (out / "private").mkdir(mode=0o700)
        (out / "prompts").mkdir()
        jsonl(out / "private/tasks.jsonl", tasks)
        jsonl(out / "private/audit.jsonl", audits)
        for split in ("train", "development"):
            jsonl(out / f"prompts/{split}.jsonl", [public_task(t) for t in tasks if t["split"] == split])
        jsonl(out / "references-train.jsonl", [example(t, t["reference"], smoke) for t in tasks if t["split"] == "train"])
        files = list(TASK_FILES)
        if tokenizer:
            (out / "tokenizer.json").write_bytes(Path(tokenizer).read_bytes())
            files.append("tokenizer.json")
        if review:
            (out / "reference-review.json").write_text(canonical(review) + "\n")
            files.append("reference-review.json")
        local_benchmarks = {}
        if exclusions.provenance:
            (out / "private/benchmarks").mkdir(mode=0o700)
            for name, info in exclusions.provenance.items():
                entry = config["benchmarks"][name]
                source = config_path.parent / (entry["path"] if isinstance(entry, dict) else entry)
                relative = f"private/benchmarks/{name}.jsonl"
                (out / relative).write_bytes(source.read_bytes())
                local_benchmarks[name] = {"path": relative, "revision": info.get("revision"), "dataset_repo": info.get("dataset_repo")}
                files.append(relative)
            (out / "benchmarks.json").write_text(canonical({"benchmarks": local_benchmarks}) + "\n")
            files.append("benchmarks.json")
        for path in (out / "private").rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        write_manifest(out / "manifest.json", {
            "schema": "python-tasks/v1", "generator": VERSION, "verifier": VERIFIER,
            "implementation_hashes": implementation_hashes(),
            "generator_sha256": filehash(Path(__file__).with_name("catalog.py")),
            "worker_sha256": filehash(WORKER), "seed": seed, "variants_per_family": variants,
            "smoke_only": smoke, "tokenizer": tokenizer_evidence,
            "context_checked": encode is not None, "reference_review": "approved" if review else "pending-human-review",
            "counts": dict(counts), "excluded": dict(excluded),
            "family_splits": {t["family"]: t["split"] for t in tasks},
            "benchmarks": local_benchmarks,
            "files": {name: filehash(out / name) for name in files},
        })
    return {"counts": dict(counts), "smoke_only": smoke, "excluded": dict(excluded)}


def grade(bundle, attempts, output, checkpoint, generation_seconds, gpu_seconds,
          split="train", min_accepted=1, min_successes_per_gpu_hour=0.0, previous_accepted=None):
    if split not in ("train", "development"):
        raise ValueError("sealed evaluation tasks are unavailable for feedback")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("checkpoint ID is required")
    for value in (generation_seconds, gpu_seconds, min_successes_per_gpu_hour):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("timing/yield values must be finite and nonnegative")
    if type(min_accepted) is not int or min_accepted < 1:
        raise ValueError("min_accepted must be positive")
    bundle = Path(bundle)
    manifest = load_manifest(bundle)
    manifest_hash = filehash(bundle / "manifest.json")
    if manifest["implementation_hashes"] != implementation_hashes() or manifest["verifier"] != VERIFIER:
        raise ValueError("bundle verifier differs from the installed worker")
    encode = None
    if manifest["tokenizer"]:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(bundle / "tokenizer.json"))
        encode = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    tasks = {t["task_id"]: t for t in records(bundle / "private/tasks.jsonl")}
    if Path(attempts).stat().st_size > 32 * 1024 * 1024:
        raise ValueError("attempt file exceeds 32MiB; grade smaller batches")
    rows = list(records(attempts))
    attempt_ids = set()
    for row in rows:
        if not isinstance(row.get("attempt_id"), str) or not row["attempt_id"].strip() or row["attempt_id"] in attempt_ids:
            raise ValueError("attempt_id must be a unique nonempty string")
        attempt_ids.add(row["attempt_id"])
        if not isinstance(row.get("task_id"), str) or row["task_id"] not in tasks or tasks[row["task_id"]]["split"] != split:
            raise ValueError("unknown task ID or wrong split")
        if not isinstance(row.get("completion"), str) or len(row["completion"].encode()) > MAX_CODE:
            raise ValueError("completion must be Python source up to 32KiB")
    if not rows:
        raise ValueError("attempt file is empty")
    config = json.loads((bundle / "benchmarks.json").read_text()) if manifest["benchmarks"] else {}
    exclusions = Exclusions(config, bundle, manifest["smoke_only"])
    probe()
    feedback, accepted, seen = [], [], set()
    if previous_accepted:
        for row in records(previous_accepted):
            if (not isinstance(row.get("task_id"), str) or row["task_id"] not in tasks
                    or tasks[row["task_id"]]["split"] != "train" or row.get("split") != "train"
                    or row.get("bundle_manifest_sha256") != manifest_hash):
                raise ValueError("previous accepted examples must belong to this bundle's training tasks")
            try:
                seen.add((row["task_id"], structure_hash(row["completion"])))
            except (SyntaxError, ValueError, RecursionError) as error:
                raise ValueError("invalid previous solution structure") from error
    started = time.monotonic()
    by_family = {}
    for row in rows:
        task = tasks[row["task_id"]]
        completion = row["completion"]
        text = example(task, completion, manifest["smoke_only"])["text"]
        try:
            code_hash = structure_hash(completion)
        except (SyntaxError, ValueError, RecursionError):
            code_hash = None
        if code_hash is None:
            status = "invalid_code"
        elif any(token in text for token in SPECIAL):
            status = "reserved_control_token"
        elif encode and encode(text) > manifest["tokenizer"]["max_tokens"]:
            status = "context_limit"
        elif exclusions.matches(text):
            status = "benchmark_overlap"
        else:
            status = verify(completion, task)["status"]
        duplicate = False
        if status == "passed" and split == "train":
            # Per-task structural dedup also catches whitespace-only completion variants.
            identity = (task["task_id"], code_hash)
            duplicate = identity in seen
            seen.add(identity)
            if not duplicate:
                accepted.append({**example(task, completion, manifest["smoke_only"]),
                                 "attempt_id": row["attempt_id"], "checkpoint": checkpoint,
                                 "verifier": VERIFIER, "bundle_manifest_sha256": manifest_hash})
        counts = by_family.setdefault(task["family"], Counter())
        counts["attempts"] += 1
        counts["passed"] += status == "passed"
        feedback.append({"attempt_id": row["attempt_id"], "task_id": task["task_id"],
                         "family": task["family"], "split": split, "status": status,
                         "duplicate": duplicate, "checkpoint": checkpoint, "verifier": VERIFIER})
    grading_seconds = time.monotonic() - started
    rate = len(accepted) * 3600 / gpu_seconds if gpu_seconds else None
    yield_met = len(accepted) >= min_accepted and rate is not None and rate >= min_successes_per_gpu_hour
    report = {"schema": "task-feedback/v1", "checkpoint": checkpoint, "split": split,
              "bundle_manifest_sha256": manifest_hash,
              "attempts_sha256": filehash(attempts), "smoke_only": manifest["smoke_only"],
              "previous_accepted_sha256": filehash(previous_accepted) if previous_accepted else None,
              "attempts": len(rows), "passed": sum(r["status"] == "passed" for r in feedback),
              "by_status": dict(Counter(r["status"] for r in feedback)),
              "accepted": len(accepted), "by_family": by_family,
              "generation_seconds_reported": generation_seconds, "gpu_seconds_reported": gpu_seconds,
              "grading_seconds_measured": grading_seconds,
              "verified_unique_examples_per_gpu_hour": rate,
              "yield_policy": {"min_accepted": min_accepted, "min_successes_per_gpu_hour": min_successes_per_gpu_hour},
              "yield_met": yield_met,
              "eligible_for_training": yield_met and not manifest["smoke_only"] and split == "train" and manifest["reference_review"] == "approved"}
    with destination(output) as out:
        jsonl(out / "feedback.jsonl", feedback)
        jsonl(out / "accepted.jsonl", accepted)
        (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
