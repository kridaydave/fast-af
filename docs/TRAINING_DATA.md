# Reproducible training corpus

The training-v1 build targets approximately 100M unique training tokens and a
separate 1M-token validation set. This is the first smoke-pretrain corpus in the
project plan, not the later multi-billion-token pretrain. Sources are fixed in
`configs/training_sources.lock.json`: The Stack Dedup Python, FineWeb-Edu
sample-10BT, and FineMath finemath-4plus. Each source and evaluation set has an
immutable Hugging Face commit. Source dataset cards and available per-record
attribution/license metadata are retained with the exports and document index.

Install the versions recorded in `requirements-data.txt`. Source access to The
Stack requires an HF read token for an account with accepted dataset terms; the
export CLI accepts a local secure token file and does not log its value.

```bash
python scripts/export_benchmarks.py --output artifacts/run/benchmarks
python scripts/fetch_training_sources.py --bucket code --output artifacts/run/raw-code --token-file /path/to/secure/token
python scripts/fetch_training_sources.py --bucket web --output artifacts/run/raw-web
python scripts/fetch_training_sources.py --bucket math --output artifacts/run/raw-math
```

Each exporter streams bounded, ordered pinned parquet shards, retaining text of
200–200,000 UTF-8 bytes and recording shard paths, row indices, byte counts,
export hashes, card hash, lock hash, and package versions. `source_record` retains
upstream metadata (including URLs or repository/license fields when present).
These are deterministic source prefixes, not a random sample of the entire
upstream dataset. The code corpus uses its upstream deduplication plus our local
exact and conservative near-deduplication.

Build a sources JSON as documented in `DATA_PIPELINE.md`, with `export_manifest`
for each source and benchmark entries as objects containing `path`,
`dataset_repo`, and `revision`. Supply both complete evaluation suites:
HumanEval/0..163 and all 974 MBPP tasks across full/prompt, train, validation,
and test. This decontaminates all splits and is independent of which MBPP tasks
will later be scored. Benchmark text is used for exclusion, never model training.

```bash
python scripts/prepare_corpus.py prepare --manifest artifacts/run/sources.json \
  --output artifacts/run/prepared --limit 200000 --seed 17 --validation-fraction 0.02
python scripts/train_tokenizer.py --train artifacts/run/prepared/train.jsonl \
  --prepared-manifest artifacts/run/prepared/manifest.json \
  --output artifacts/run/tokenizer --vocab-size 32000 --seed 42 --report
python scripts/build_training_mix.py --prepared artifacts/run/prepared \
  --tokenizer-dir artifacts/run/tokenizer --output artifacts/run/final \
  --train-tokens 100000000 --validation-tokens 1000000 --seq-len 1024
python scripts/audit_training_data.py --bundle artifacts/run/final \
  --prepared artifacts/run/prepared --source-config artifacts/run/sources.json \
  --out artifacts/run/final/audit.json
```

Preparation validates source export hashes/revisions and benchmark completeness.
Present benchmark fields must have the expected type and each suite must produce
usable signatures. Every document is checked against 13-word n-grams and
normalized benchmark substrings of at least 32 characters. Substring search uses
Aho-Corasick when installed. Exact deduplication normalizes case and whitespace.
Near-deduplication computes 64-bit SimHash from unique normalized 5-word shingles,
with a maximum Hamming distance of 3 (documents with fewer than 20 words skip
this check). These conservative checks do not catch all semantic/paraphrased or
structural duplicates; a zero detected-overlap count is scoped to these methods.
Hash splitting prevents identical documents across training and validation.

The 32k byte-level tokenizer learns only from the cleaned training split, on a
seeded sample bounded to 100MB. It preserves whitespace/Unicode and reserves IDs
MASK=0, EOS=1, PAD=2, UNK=3, BOS=4. A small requested vocabulary fails if the
corpus cannot support it. Saved truncation/padding and literal control-token
strings are rejected at packing. The tokenizer manifest records the input hash,
prepared manifest hash, library version, sample coverage and tokenizer hash.

`build_training_mix.py` enforces exact **50% code, 35% web, 15% math by tokens**
for each split. It builds unique token pools with EOS between documents and
interleaves full sequences in seeded 20-sequence cycles (10 code, 7 web, 3 math).
There is no oversampling/repetition to satisfy quotas: insufficient unique tokens
fail the build. Budgets round down to whole cycles, yielding 99,983,360 training
and 983,040 validation tokens at the defaults. Quota boundaries may cut a document;
partially used documents and unused candidate tokens are explicitly counted.
Each sequence contains tokens from one bucket; EOS-marked documents may share it.

The final bundle stores little-endian uint16 `.bin` shards, a standalone tokenizer,
source/tokenizer manifests, and sequence/document provenance indices. Load a shard
with `np.memmap(path, dtype='<u2', mode='r').reshape(-1, 1024)`, then cast selected
batches to the training framework's integer dtype. Attention masks are all ones.
Diffusion masking/corruption is generated dynamically in the training loop; the
stored IDs remain original targets and contain no MASK/PAD/UNK/BOS tokens.

The audit checks fingerprints, lineage and shard hashes, vocabulary/control IDs,
exact quotas and sequence offsets. With prepared data supplied, it independently
rescans the full candidate corpus for wrong split assignments, detected benchmark
overlaps, and exact/near duplicates. The audit result is a separate hashed report.
Source-export metadata and the complete source configuration remain in the run
folder alongside the bundle. Generated source text and tokens are ignored by Git;
commit the tools/configuration, not the corpus or credentials.
