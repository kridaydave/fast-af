# CPU byte-level BPE tokenizer

Dependency: `tokenizers==0.23.3` (the exact installed version is recorded). Use
only cleaned **training** JSONL with `{ "text": "...", "bucket": "code|web|math" }`.
The source metadata produced by the prep pipeline is preserved in its manifest.

```bash
python scripts/train_tokenizer.py --train artifacts/prepared/train.jsonl \
  --prepared-manifest artifacts/prepared/manifest.json \
  --output artifacts/tokenizer --vocab-size 32000 --seed 42 \
  --max-docs 100000 --max-bytes 100000000 --report
```

The prepared-manifest hash verifies that the input is its training split. It
rejects accidentally passing validation data. Without that manifest, explicit
inputs are accepted but outputs are marked smoke-only. The trainer never fetches
data, uses validation to train merges, or changes model/benchmark tokenizers.

All 256 bytes are in the initial alphabet. ByteLevel pre-tokenization uses
`add_prefix_space=False`, ByteLevel decoding preserves code whitespace/Unicode,
and there is no destructive normalizer. Special IDs are `[MASK]=0`, `[EOS]=1`,
`[PAD]=2`, `[UNK]=3`, `[BOS]=4`. Reserved literals in content are rejected. The
trainer checks exact round trips and unknown-token counts on its selected sample.

Sampling shuffles documents with a fixed seed, accepts up to `max_docs`, and
skips documents exceeding the remaining UTF-8 byte budget. The resulting document
sample need not match the final corpus mixture by tokens. The report gives bucket
document/byte/token counts, bytes per token, token fractions, unknown counts, and
roundtrip failures. It describes the selected tokenizer sample, not final training
shards. Input sample text is held in memory; use pre-bounded local sample exports.

Outputs are `tokenizer.json` and `config/manifest.json`: input/tokenizer/prepared
manifest hashes, vocabulary/special IDs, sampling limits/counts/seed, bucket
coverage, package version, and a stable fingerprint. Insufficient corpus for the
requested vocabulary fails instead of silently emitting a smaller tokenizer.
Use a small vocabulary for tiny fixtures; do not freeze a fixture tokenizer for
production. Repeated runs must use identical input order, file bytes, parameters,
and library version. Output directories are atomic and cannot be overwritten.

Freeze production artifacts only after upstream source revisions/licenses,
complete benchmark decontamination, bucket coverage, and held-out tokenizer
inspection have been reviewed. A smoke-only prepared corpus remains smoke-only
through tokenizer training and packing.
