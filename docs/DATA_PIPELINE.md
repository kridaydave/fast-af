# Local corpus preparation and packing

For the actual ~100M-token build, start with [TRAINING_DATA.md](TRAINING_DATA.md).
The lower-level tooling prepares reproducible data from local UTF-8 JSONL files. It
performs no downloads. Planned sources are FineWeb-Edu sample-10BT (web), The
Stack Dedup Python (code), and FineMath finemath-4plus (math). The intended final
mixture is 50% code / 35% web / 15% math **by tokens**. Sampling limits documents
per source; this foundation reports observed token fractions but does not enforce
that final mixture. Verify the mixture before a production training run.

Each source row must contain `text`. Use a JSON source config such as:

```json
{
  "sources": [
    {"bucket": "code", "dataset_repo": "bigcode/the-stack-dedup",
     "revision": "<40-character-source-commit>", "license": "<verified-source-license>",
     "source_file": "samples/code.jsonl", "export_manifest": "samples/export.json"}
  ],
  "benchmarks": {
    "human_eval": {"path":"evaluation/humaneval.jsonl", "dataset_repo":"openai/openai_humaneval", "revision":"<40-character-commit>"},
    "mbpp": {"path":"evaluation/mbpp.jsonl", "dataset_repo":"google-research-datasets/mbpp", "revision":"<40-character-commit>"}
  }
}
```

Paths resolve relative to the config. Add web/math entries as needed. Revisions
and licenses describe the actual exported source files; a file hash alone cannot
recover missing upstream provenance. Both complete benchmark files (164 HumanEval tasks and all 974 MBPP tasks) are required
outside smoke mode, along with source export manifests from fetch_training_sources.py. The loader recognizes HumanEval prompt/canonical_solution/test
and MBPP text/code/test_list/test_setup_code/docstring fields. Benchmark file
hashes and record counts are recorded. Production mode checks the complete distinct task ID sets and rejects malformed
present fields and suites without usable signatures. Fetch them from pinned revisions.

```bash
python scripts/prepare_corpus.py prepare --manifest sources.json \
  --output artifacts/prepared --seed 17 --limit 10000 --validation-fraction 0.01
```

`--limit` bounds selected documents **per source** using seeded content-hash
priorities. Global casefolded, whitespace-normalized exact deduplication happens
before selection, followed by conservative 64-bit SimHash near-deduplication
(over unique 5-word shingles; distance <=3; minimum 20 words). These checks are
conservative for case-sensitive code and do not cover every structural duplicate. Train/validation assignment hashes each normalized
document and seed; appending documents cannot move an existing document between
splits. Deduplication prevents identical normalized documents across splits.

Decontamination checks 13-word n-grams plus normalized substring matches of
benchmark fields at least 32 characters long. It records separate removal counts.
These checks can miss paraphrases, formatting changes, short fragments, and near
duplicates; they can also over-remove. Counts describe detected overlaps, not a
proof of zero contamination. The production build includes a separate residual audit; its zero-overlap result
is scoped to the recorded methods and does not establish semantic independence.

Outputs are `train.jsonl`, `validation.jsonl`, and `manifest.json`. Rows retain
text, bucket, source revision/license/input hash, and normalized document hash.
The manifest includes input/output hashes, filtering/sampling counts, bucket
counts, split rules, benchmark hashes, seed, and a stable fingerprint. Small
samples can yield an empty split: inspect counts and enlarge the sample rather
than changing the seed to optimize evaluation results. The implementation keeps
unique-document hashes and selected sample text in memory; it is not yet a
multi-billion-token distributed data builder.

For existing samples with incomplete provenance or no benchmark files, pass
`--no-benchmarks-smoke`. This explicitly marks the output `smoke_only: true` and
allows unpinned revisions. Such output is for plumbing checks, not production
training. The marker propagates through verified tokenizer and packing manifests.

Train the tokenizer on **train.jsonl only** using [TOKENIZER.md](TOKENIZER.md).
Then pack separately by split, with lineage verification:

```bash
python scripts/prepare_corpus.py pack \
  --tokenizer artifacts/tokenizer/tokenizer.json \
  --tokenizer-manifest artifacts/tokenizer/config/manifest.json \
  --prepared-manifest artifacts/prepared/manifest.json \
  --train artifacts/prepared/train.jsonl --validation artifacts/prepared/validation.jsonl \
  --output artifacts/packed --seq-len 1024
```

Packing verifies tokenizer IDs MASK=0, EOS=1, PAD=2, UNK=3, BOS=4, appends EOS per
document, and writes fixed-length `input_ids` with all-one `attention_mask`.
Incomplete tails are dropped separately and counted. Documents may span sequences;
training mask corruption is a later training-loop operation. A second global
deduplication check rejects cross-split leaks. Preparation removes and counts reserved special-token literals. Packing rejects
remaining literals and saved tokenizer truncation/padding settings.

The packed manifest records source/manifest/tokenizer/shard hashes, document and
token counts, dropped tails, and token fractions **before tail dropping**. Omitting
lineage manifests forces the packed output to remain smoke-only. Every stage
publishes a fresh output directory atomically and refuses to overwrite one.
