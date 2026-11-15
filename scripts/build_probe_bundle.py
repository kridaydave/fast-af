#!/usr/bin/env python3
"""Slice a verified corpus bundle into a small, reproducible probe bundle.

The probe bundle is a read-only prefix of real training-v1 data: the first N
train sequences and the first M validation sequences, byte for byte. It exists
so GPU correctness and throughput jobs can ship a few hundred kilobytes to a
remote runtime instead of the full corpus. Nothing about it is synthetic, and
every shard hash is declared in the manifest so a recipient can re-derive it.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

if __package__:
    from .prepare_corpus import canonical, digest, filehash
    from .train_diffusion import bundle_provenance, load_manifest
else:
    from prepare_corpus import canonical, digest, filehash
    from train_diffusion import bundle_provenance, load_manifest


def take_sequences(corpus: Path, info: dict, sequences: int, seq_len: int) -> bytes:
    """Return the first `sequences` packed sequences of a split, in shard order."""
    wanted = sequences * seq_len * 2
    chunks: list[bytes] = []
    for shard in info['shards']:
        path = corpus / shard['path']
        with path.open('rb') as stream:
            data = stream.read(wanted)
        chunks.append(data)
        wanted -= len(data)
        if not wanted:
            break
    if wanted:
        raise ValueError(f'split has fewer than {sequences} sequences')
    return b''.join(chunks)


def build(corpus: Path, out: Path, train_sequences: int, validation_sequences: int,
          scope: str, seq_len: int | None = None) -> dict:
    corpus, out = Path(corpus), Path(out)
    for sequences in (train_sequences, validation_sequences):
        if type(sequences) is not int or sequences < 1:
            raise ValueError('sequence counts must be positive integers')
    meta = load_manifest(corpus)
    provenance = bundle_provenance(corpus, meta)
    seq_len = meta['seq_len'] if seq_len is None else seq_len
    if type(seq_len) is not int or seq_len < 1:
        raise ValueError('seq_len must be a positive integer')
    for name, count in (('train', train_sequences), ('validation', validation_sequences)):
        if count * seq_len > meta['splits'][name]['tokens']:
            raise ValueError(f'{name} split has fewer than {count} sequences')
    out.mkdir(parents=True, exist_ok=False)
    splits = {}
    for name, sequences in (('train', train_sequences), ('validation', validation_sequences)):
        if name not in meta['splits']:
            raise ValueError(f'corpus has no {name} split')
        data = take_sequences(corpus, meta['splits'][name], sequences, seq_len)
        path = out / f'{name}.bin'
        path.write_bytes(data)
        splits[name] = {
            'sequences': sequences,
            'shards': [{'path': path.name, 'sequences': sequences,
                        'sha256': filehash(path), 'tokens': sequences * seq_len}],
            'tokens': sequences * seq_len,
        }
    for name in ('tokenizer.json', 'tokenizer_manifest.json'):
        source = corpus / name
        if not source.is_file():
            raise ValueError(f'corpus has no {name}')
        shutil.copyfile(source, out / name)

    bundle = {'derived_from': provenance,
              'dtype': meta['dtype'],
              'schema': 'training-mixture/v1',
              'scope': scope,
              'seq_len': seq_len,
              'smoke_only': True,
              'special_ids': meta['special_ids'],
              'splits': splits,
              'tokenizer_manifest_sha256': filehash(out / 'tokenizer_manifest.json'),
              'tokenizer_sha256': filehash(out / 'tokenizer.json')}
    bundle['fingerprint'] = digest(canonical(bundle).encode())
    (out / 'manifest.json').write_text(json.dumps(bundle, indent=2, sort_keys=True) + '\n',
                                       encoding='utf-8')
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True,
                        help='verified training-mixture/v1 bundle to slice')
    parser.add_argument('--out', type=Path, required=True, help='destination bundle directory')
    parser.add_argument('--train-sequences', type=int, default=128)
    parser.add_argument('--validation-sequences', type=int, default=32)
    parser.add_argument('--scope', default=('read-only subset of training-v1 for bounded '
                                            'infrastructure and probe checks; no new audit or '
                                            'quality claim'))
    parser.add_argument('--seq-len', type=int, default=None,
                        help='sequence length of the sliced bundle (default: the corpus value)')
    args = parser.parse_args(argv)
    bundle = build(args.corpus, args.out, args.train_sequences, args.validation_sequences,
                   args.scope, args.seq_len)
    print(json.dumps({'out': str(args.out), 'fingerprint': bundle['fingerprint'],
                      'splits': {k: v['tokens'] for k, v in bundle['splits'].items()}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
