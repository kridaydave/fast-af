#!/usr/bin/env python3
"""Offline, reproducible sample preparation and token packing."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import tempfile

BUCKETS = ("code", "web", "math")
SPECIAL = ("[MASK]", "[EOS]", "[PAD]", "[UNK]", "[BOS]")
PLANNED_MIX = {"code": 0.50, "web": 0.35, "math": 0.15}
NGRAM = 13


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    return " ".join(text.casefold().split())


def grams(text, n=NGRAM):
    words = norm(text).split()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def records(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: malformed JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            yield row


def document_hash(text):
    return digest(norm(text).encode())


def split_for(text_hash, seed, fraction):
    value = int(digest(f"split:{seed}:{text_hash}".encode()), 16)
    return "validation" if value / (1 << 256) < fraction else "train"


@contextmanager
def destination(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"output already exists: {path}; choose a new directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as tmp:
        staging = Path(tmp) / "result"
        staging.mkdir()
        yield staging
        if path.exists():
            raise ValueError(f"output appeared during preparation: {path}")
        os.rename(staging, path)


def write_manifest(path, meta):
    meta["fingerprint"] = digest(canonical(meta).encode())
    path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def prepared_manifest(path, train, validation=None):
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("schema") != "prepared-corpus/v1":
        raise ValueError("expected prepared-corpus/v1 manifest")
    claimed = meta.pop("fingerprint", None)
    if claimed != digest(canonical(meta).encode()):
        raise ValueError("prepared manifest fingerprint mismatch")
    for split, filename in (("train", train), ("validation", validation)):
        if filename is not None and meta.get("outputs", {}).get(split, {}).get("sha256") != filehash(filename):
            raise ValueError(f"{split} file does not match prepared manifest")
    return meta


def verify_source_export(src, base):
    path = base / src.get("export_manifest", "")
    if not src.get("export_manifest"):
        raise ValueError("verified source export manifest required outside smoke mode")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("schema") != "source-export/v1":
        raise ValueError("invalid source export schema")
    claimed = meta.pop("fingerprint", None)
    if claimed != digest(canonical(meta).encode()):
        raise ValueError("source export fingerprint mismatch")
    for key in ("bucket", "dataset_repo", "revision", "license"):
        if src[key] != meta.get("source", {}).get(key):
            raise ValueError(f"source export {key} mismatch")
    if meta.get("documents_sha256") != filehash(base / src["source_file"]):
        raise ValueError("source export document hash mismatch")
    return filehash(path)


class SubstringIndex:
    def __init__(self, fragments):
        self.fragments = fragments
        self.matcher = None
        if fragments:
            try:
                import ahocorasick_rs
                self.matcher = ahocorasick_rs.AhoCorasick(fragments)
            except ImportError:
                pass

    def matches(self, normalized):
        if self.matcher is not None:
            return bool(self.matcher.find_matches_as_indexes(normalized))
        return any(fragment in normalized for fragment in self.fragments)


def benchmark_index(config, base, smoke):
    supplied = config.get("benchmarks", {})
    if not isinstance(supplied, dict):
        raise ValueError("benchmarks must map human_eval and mbpp to local JSONL paths")
    if not smoke and set(supplied) != {"human_eval", "mbpp"}:
        raise ValueError("both human_eval and mbpp JSONL required; use explicit smoke mode to bypass")
    all_grams, fragments, provenance = set(), set(), {}
    fields = {"human_eval": ("prompt", "canonical_solution", "test"),
              "mbpp": ("text", "code", "test_list", "test_setup_code", "docstring")}
    for name, filename in supplied.items():
        if name not in fields:
            raise ValueError(f"unknown benchmark: {name}")
        entry = filename if isinstance(filename, dict) else {"path": filename}
        if not smoke and (not re.fullmatch(r"[0-9a-f]{40}", entry.get("revision", "")) or not entry.get("dataset_repo")):
            raise ValueError(f"{name}: pinned benchmark dataset_repo/revision required")
        path = base / entry["path"]
        count = 0
        task_ids, suite_grams, suite_fragments = set(), set(), set()
        for row in records(path):
            task_id = str(row.get("task_id", ""))
            if not smoke and (not task_id or task_id in task_ids):
                raise ValueError(f"{name}: missing or duplicate task_id")
            task_ids.add(task_id)
            texts = []
            for key in fields[name]:
                if key not in row:
                    continue
                value = row.get(key)
                expected_list = key in ("test_list",)
                if (expected_list and not isinstance(value, list)) or (not expected_list and not isinstance(value, str)):
                    raise ValueError(f"{name}: malformed {key}")
                if isinstance(value, str) and value.strip():
                    texts.append(value)
                elif isinstance(value, list):
                    if any(not isinstance(v, str) for v in value):
                        raise ValueError(f"{name}: {key} must contain strings")
                    texts.extend(v for v in value if v.strip())
            if not texts:
                raise ValueError(f"{name}: benchmark record has no supported nonempty text fields")
            for text in texts:
                suite_grams.update(grams(text))
                normalized = norm(text)
                if len(normalized) >= 32:
                    suite_fragments.add(normalized)
            count += 1
        if not count:
            raise ValueError(f"{name}: empty benchmark file")
        if not suite_grams and not suite_fragments:
            raise ValueError(f"{name}: benchmark has no usable signatures")
        if not smoke:
            expected = {f"HumanEval/{i}" for i in range(164)} if name == "human_eval" else {str(i) for i in range(1, 975)}
            if task_ids != expected:
                raise ValueError(f"{name}: incomplete benchmark task set")
        all_grams.update(suite_grams)
        fragments.update(suite_fragments)
        provenance[name] = {**entry, "path": str(path), "sha256": filehash(path), "records": count,
                            "ngram_signatures": len(suite_grams), "substring_signatures": len(suite_fragments)}
    if supplied and not all_grams and not fragments:
        raise ValueError("benchmark text is too short for the configured checks")
    return all_grams, sorted(fragments), provenance


def prepare(manifest, out, seed=17, limit=10000, smoke=False, validation_fraction=0.01):
    if type(seed) is not int or seed < 0 or type(limit) is not int or limit < 1:
        raise ValueError("seed must be nonnegative and limit must be positive")
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    manifest = Path(manifest)
    config = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list) or not config["sources"]:
        raise ValueError("manifest must contain a nonempty sources list")
    bg, fragments, benchmarks = benchmark_index(config, manifest.parent, smoke)
    substring_index = SubstringIndex(fragments)
    if __package__:
        from .near_dedup import NearDedup
    else:
        from near_dedup import NearDedup
    near = NearDedup()
    seen, docs, sources = set(), [], []
    totals = Counter()
    for index, src in enumerate(config["sources"]):
        if not isinstance(src, dict):
            raise ValueError("each source must be an object")
        for key in ("bucket", "dataset_repo", "revision", "license", "source_file"):
            if not isinstance(src.get(key), str) or not src[key].strip():
                raise ValueError(f"source {index}: missing {key}")
        if src["bucket"] not in BUCKETS:
            raise ValueError(f"source {index}: invalid bucket")
        if not smoke and not re.fullmatch(r"[0-9a-f]{40}", src["revision"]):
            raise ValueError(f"source {index}: immutable 40-character dataset revision required outside smoke mode")
        path = manifest.parent / src["source_file"]
        source_meta = {**src, "sha256": filehash(path)}
        if not smoke:
            source_meta["export_manifest_sha256"] = verify_source_export(src, manifest.parent)
        source_counts, selected = Counter(), []
        for number, row in enumerate(records(path)):
            source_counts["read"] += 1
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{path}: record {number+1} has non-string text")
            normalized = norm(text)
            if not normalized:
                source_counts["empty"] += 1
                continue
            if any(token in text for token in SPECIAL):
                source_counts["reserved_literal"] += 1
                continue
            h = document_hash(text)
            if h in seen:
                source_counts["duplicate"] += 1
                continue
            seen.add(h)
            if bg and grams(text) & bg:
                source_counts["benchmark_ngram"] += 1
                continue
            if substring_index.matches(normalized):
                source_counts["benchmark_substring"] += 1
                continue
            if near.duplicate(normalized):
                source_counts["near_duplicate"] += 1
                continue
            source_counts["eligible"] += 1
            upstream = {key: row[key] for key in ("upstream_file", "upstream_row", "source_record") if key in row}
            row = {"text": text, "bucket": src["bucket"], "source": {
                "dataset_repo": src["dataset_repo"], "revision": src["revision"],
                "license": src["license"], "input_sha256": source_meta["sha256"], "document_sha256": h}}
            row["source"].update(upstream)
            priority = int(digest(f"sample:{seed}:{h}".encode()), 16)
            item = (-priority, number, row)
            if len(selected) < limit:
                heapq.heappush(selected, item)
            elif priority < -selected[0][0]:
                heapq.heapreplace(selected, item)
        retained = [item[2] for item in sorted(selected, key=lambda item: (-item[0], item[1]))]
        source_counts["selected"] = len(retained)
        source_counts["sampled_out"] = source_counts["eligible"] - len(retained)
        source_meta["counts"] = dict(source_counts)
        totals.update(source_counts)
        sources.append(source_meta)
        docs.extend(retained)
    if not docs:
        raise ValueError("no documents survived preparation")
    split_counts = {s: Counter() for s in ("train", "validation")}
    with destination(out) as target:
        handles = {s: (target / f"{s}.jsonl").open("w", encoding="utf-8") for s in split_counts}
        try:
            for row in docs:
                split = split_for(row["source"]["document_sha256"], seed, validation_fraction)
                handles[split].write(canonical(row) + "\n")
                split_counts[split][row["bucket"]] += 1
        finally:
            for handle in handles.values():
                handle.close()
        write_manifest(target / "manifest.json", {
            "schema": "prepared-corpus/v1", "config_sha256": filehash(manifest),
            "implementation_sha256": filehash(__file__),
            "near_dedup_implementation_sha256": filehash(Path(__file__).with_name("near_dedup.py")),
            "near_dedup": "64-bit SimHash over unique normalized 5-word shingles; Hamming distance <=3; minimum 20 words",
            "seed": seed, "limit_per_source": limit, "validation_fraction": validation_fraction,
            "sampling": "lowest SHA256(seed, normalized document hash) per source",
            "split": "SHA256(seed, normalized document hash)",
            "planned_token_mix": PLANNED_MIX, "token_mix_enforced": False,
            "smoke_only": smoke, "benchmark_checks": {"ngram_words": NGRAM, "substring_min_chars": 32},
            "benchmarks": benchmarks, "sources": sources, "counts": dict(totals),
            "outputs": {s: {"path": f"{s}.jsonl", "sha256": filehash(target / f"{s}.jsonl"),
                             "documents": sum(counts.values()), "bucket_documents": dict(counts)} for s, counts in split_counts.items()}})
    return Path(out) / "manifest.json"


def pack(args):
    from tokenizers import Tokenizer
    if type(args.seq_len) is not int or args.seq_len < 1:
        raise ValueError("seq_len must be positive")
    tok = Tokenizer.from_file(str(args.tokenizer))
    if tok.truncation is not None or tok.padding is not None:
        raise ValueError("tokenizer truncation/padding must be disabled for packing")
    if [tok.token_to_id(t) for t in SPECIAL] != list(range(5)):
        raise ValueError("token IDs must be MASK0 EOS1 PAD2 UNK3 BOS4")
    if Path(args.train).resolve() == Path(args.validation).resolve():
        raise ValueError("train and validation must be different files")
    provenance = None
    if getattr(args, "prepared_manifest", None):
        provenance = prepared_manifest(args.prepared_manifest, args.train, args.validation)
    token_meta = None
    if getattr(args, "tokenizer_manifest", None):
        token_meta = json.loads(Path(args.tokenizer_manifest).read_text(encoding="utf-8"))
        if not isinstance(token_meta, dict) or token_meta.get("schema") != "tokenizer/v1":
            raise ValueError("expected tokenizer/v1 manifest")
        claimed = token_meta.pop("fingerprint", None)
        if claimed != digest(canonical(token_meta).encode()) or token_meta.get("tokenizer_sha256") != filehash(args.tokenizer):
            raise ValueError("tokenizer manifest/hash mismatch")
        if not provenance or token_meta.get("prepared_manifest_sha256") != filehash(args.prepared_manifest):
            raise ValueError("tokenizer and packed data must share the same prepared manifest")
    seen, split_meta, total_buckets = {}, {}, Counter()
    with destination(args.output) as target:
        for split, filename in (("train", args.train), ("validation", args.validation)):
            counts, bucket_tokens, pending = Counter(), Counter(), []
            shard = target / f"{split}.jsonl"
            with shard.open("w", encoding="utf-8") as stream:
                for row in records(filename):
                    if not isinstance(row.get("text"), str) or not norm(row["text"]) or row.get("bucket") not in BUCKETS:
                        raise ValueError(f"{filename}: expected nonempty text and valid bucket")
                    if any(token in row["text"] for token in SPECIAL):
                        raise ValueError("document contains reserved special-token literal")
                    h = document_hash(row["text"])
                    if h in seen:
                        raise ValueError(f"duplicate/leaked document in {split}; first seen in {seen[h]}")
                    seen[h] = split
                    tokens = tok.encode(row["text"], add_special_tokens=False).ids
                    if any(t < 5 for t in tokens):
                        raise ValueError("document contains reserved special tokens; sanitize or exclude it explicitly")
                    tokens.append(1)
                    bucket_tokens[row["bucket"]] += len(tokens)
                    counts["documents"] += 1
                    counts["encoded_tokens_including_eos"] += len(tokens)
                    pending.extend(tokens)
                    while len(pending) >= args.seq_len:
                        ids, pending = pending[:args.seq_len], pending[args.seq_len:]
                        stream.write(canonical({"input_ids": ids, "attention_mask": [1] * args.seq_len}) + "\n")
                        counts["sequences"] += 1
            counts["tail_dropped"] = len(pending)
            counts["packed_tokens"] = counts["sequences"] * args.seq_len
            total_buckets.update(bucket_tokens)
            split_meta[split] = {**dict(counts), "bucket_tokens_before_tail_drop": dict(bucket_tokens),
                                 "input_sha256": filehash(filename), "shard_sha256": filehash(shard), "path": shard.name}
        total = sum(total_buckets.values())
        if not total:
            raise ValueError("cannot pack an empty corpus")
        write_manifest(target / "manifest.json", {
            "schema": "packed-corpus/v1", "tokenizer_sha256": filehash(args.tokenizer),
            "implementation_sha256": filehash(__file__),
            "prepared_manifest_sha256": filehash(args.prepared_manifest) if provenance else None,
            "tokenizer_manifest_sha256": filehash(args.tokenizer_manifest) if token_meta else None,
            "smoke_only": (provenance["smoke_only"] or token_meta["smoke_only"]) if provenance and token_meta else True,
            "seq_len": args.seq_len, "special_ids": dict(zip(SPECIAL, range(5))),
            "tail_policy": "drop separately for each split", "splits": split_meta,
            "planned_token_mix": PLANNED_MIX, "token_mix_enforced": False,
            "observed_token_mix_before_tail_drop": {k: total_buckets[k] / total for k in BUCKETS}})
    return Path(args.output) / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=17)
    prep.add_argument("--limit", type=int, default=10000, help="maximum selected documents per source")
    prep.add_argument("--validation-fraction", type=float, default=0.01)
    prep.add_argument("--no-benchmarks-smoke", action="store_true", help="marks all outputs smoke-only; allows missing benchmarks/unpinned revisions")
    packed = sub.add_parser("pack")
    for key in ("tokenizer", "train", "validation", "output"):
        packed.add_argument("--" + key, required=True)
    packed.add_argument("--seq-len", type=int, default=1024)
    packed.add_argument("--prepared-manifest", help="verify prepared train/validation hashes and retain smoke status")
    packed.add_argument("--tokenizer-manifest", help="verify tokenizer hash and its training-corpus lineage")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            path = prepare(args.manifest, args.output, args.seed, args.limit, args.no_benchmarks_smoke, args.validation_fraction)
        else:
            path = pack(args)
        print(path)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
