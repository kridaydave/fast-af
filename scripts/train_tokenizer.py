#!/usr/bin/env python3
"""Train byte-level BPE from explicit TRAIN-only JSONL sample files."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import random

if __package__:
    from .prepare_corpus import (BUCKETS, SPECIAL, destination, filehash,
                                 prepared_manifest, records, write_manifest)
else:
    from prepare_corpus import (BUCKETS, SPECIAL, destination, filehash,
                                prepared_manifest, records, write_manifest)


def load(paths):
    docs = []
    if len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError("duplicate input paths")
    for filename in paths:
        path = Path(filename)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing or empty input: {path}")
        for row in records(path):
            if not isinstance(row.get("text"), str) or not row["text"].strip() or row.get("bucket") not in BUCKETS:
                raise ValueError(f"{path}: expected nonempty text and bucket code|web|math")
            if any(token in row['text'] for token in SPECIAL):
                raise ValueError(f"{path}: document contains reserved special-token literal")
            docs.append((row['text'], row['bucket']))
    if not docs:
        raise ValueError("training corpus has no documents")
    return docs


def report(tok, docs):
    document_counts, byte_counts, token_counts, failures, unknown = [Counter() for _ in range(5)]
    for text, bucket in docs:
        ids = tok.encode(text, add_special_tokens=False).ids
        document_counts[bucket] += 1
        byte_counts[bucket] += len(text.encode('utf-8'))
        token_counts[bucket] += len(ids)
        unknown[bucket] += ids.count(tok.token_to_id('[UNK]'))
        failures[bucket] += tok.decode(ids, skip_special_tokens=False) != text
    total_tokens = sum(token_counts.values())
    return {bucket: {'documents': document_counts[bucket], 'utf8_bytes': byte_counts[bucket],
                     'encoded_tokens': token_counts[bucket], 'unknown_tokens': unknown[bucket],
                     'roundtrip_failures': failures[bucket],
                     'bytes_per_token': byte_counts[bucket] / token_counts[bucket] if token_counts[bucket] else None,
                     'token_fraction': token_counts[bucket] / total_tokens if total_tokens else 0}
            for bucket in BUCKETS}


def train(inputs, output, vocab_size=32000, max_docs=100000,
          max_bytes=100_000_000, seed=42, prepared=None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    if any(type(v) is not int or v < 1 for v in (max_docs, max_bytes, vocab_size)) or type(seed) is not int or seed < 0:
        raise ValueError('positive integer limits/vocab and nonnegative seed required')
    if vocab_size < len(SPECIAL) + 256:
        raise ValueError('vocab_size must be at least 261 for byte alphabet and special tokens')
    provenance = None
    if prepared:
        if len(inputs) != 1:
            raise ValueError('prepared manifest verification requires exactly its train.jsonl input')
        provenance = prepared_manifest(prepared, inputs[0])
    docs = load(inputs)
    random.Random(seed).shuffle(docs)
    chosen, used = [], 0
    for text, bucket in docs:
        size = len(text.encode('utf-8'))
        if len(chosen) >= max_docs:
            break
        if used + size <= max_bytes:
            chosen.append((text, bucket))
            used += size
    if not chosen:
        raise ValueError('sample limits exclude all training text')
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator((text for text, _ in chosen), trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=list(SPECIAL),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(f'insufficient sample for requested vocab: got {tok.get_vocab_size()}, wanted {vocab_size}')
    if [tok.token_to_id(token) for token in SPECIAL] != list(range(5)):
        raise ValueError('trainer did not preserve the special-token contract')
    sample_report = report(tok, chosen)
    if any(row['roundtrip_failures'] or row['unknown_tokens'] for row in sample_report.values()):
        raise ValueError('tokenizer roundtrip/byte coverage validation failed')
    with destination(output) as out:
        token_path = out / 'tokenizer.json'
        tok.save(str(token_path))
        manifest = {'schema': 'tokenizer/v1', 'tokenizer_sha256': filehash(token_path),
                    'implementation_sha256': filehash(__file__),
                    'inputs': [{'path': str(Path(p)), 'sha256': filehash(p)} for p in inputs],
                    'prepared_manifest_sha256': filehash(prepared) if prepared else None,
                    'smoke_only': provenance['smoke_only'] if provenance else True,
                    'vocab_size': tok.get_vocab_size(), 'seed': seed, 'max_docs': max_docs,
                    'max_bytes': max_bytes, 'sample_docs': len(chosen), 'sample_bytes': used,
                    'special_ids': dict(zip(SPECIAL, range(5))),
                    'sampler': 'seeded document shuffle; skip documents exceeding remaining byte budget',
                    'sampling_is_token_mix': False, 'sample_bucket_docs': dict(Counter(b for _, b in chosen)),
                    'sample_report': sample_report, 'tokenizers_version': version('tokenizers')}
        (out / 'config').mkdir()
        write_manifest(out / 'config/manifest.json', manifest)
    return tok, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', nargs='+', required=True, help='explicit cleaned TRAIN-only JSONL')
    parser.add_argument('--output', required=True)
    parser.add_argument('--vocab-size', type=int, default=32000)
    parser.add_argument('--max-docs', type=int, default=100000)
    parser.add_argument('--max-bytes', type=int, default=100_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepared-manifest', help='verify train-only input hash and record corpus provenance')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    try:
        _, manifest = train(args.train, args.output, args.vocab_size, args.max_docs, args.max_bytes, args.seed, args.prepared_manifest)
        if args.report:
            import json
            print(json.dumps(manifest['sample_report'], indent=2))
        else:
            print(Path(args.output) / 'config/manifest.json')
    except (ValueError, OSError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
