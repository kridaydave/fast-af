"""Sample the three pretrain buckets, dropping HumanEval-contaminated docs.

Spec (edgefast-diff-100m.md): "Decontamination (n-gram check vs HumanEval)
runs inside the prep script on everything." A doc is dropped when it shares
any 13-word n-gram with a HumanEval prompt, canonical solution, or test
suite.
"""
import gzip
import json
import os
import sys
import urllib.request

from datasets import load_dataset

N_DOCS = 2000
NGRAM = 13

# Same canonical HumanEval source as bench/humaneval.py: eval and
# decontamination must see the same text.
_HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/"
    "HumanEval.jsonl.gz")


def load_humaneval() -> list[dict]:
    try:
        return list(load_dataset("openai/openai_humaneval", split="test"))
    except Exception:
        with urllib.request.urlopen(_HUMANEVAL_URL) as r:
            raw = gzip.decompress(r.read()).decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def humaneval_ngrams(rows: list[dict], n: int = NGRAM) -> set[tuple]:
    grams = set()
    for row in rows:
        # HumanEval's solution key is canonical_solution, not completion --
        # reading the wrong key once silently dropped every solution n-gram
        for field in ("prompt", "canonical_solution", "test"):
            words = row.get(field, "").lower().split()
            grams.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return grams


def contaminated(text: str, grams: set[tuple], n: int = NGRAM) -> bool:
    words = text.lower().split()
    return any(tuple(words[i:i + n]) in grams for i in range(len(words) - n + 1))


def sample(dataset, key: str, path: str, grams: set[tuple]) -> None:
    written = dropped = 0
    tmp = path + ".part"
    try:
        with open(tmp, "w") as f:
            for i, doc in enumerate(dataset):
                if i >= N_DOCS:
                    break
                text = doc[key]
                if contaminated(text, grams):
                    dropped += 1
                    continue
                f.write(json.dumps({"text": text}) + "\n")
                written += 1
        os.replace(tmp, path)  # never leave a half-written output behind
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"wrote {written} docs to {path} (dropped {dropped} contaminated)")
    for line in open(path).readlines()[:2]:
        print("---", json.loads(line)["text"][:200].replace("\n", " "))


def main() -> None:
    code = 1
    try:
        code = 0 if _run() else 1
    except BaseException:
        # any failure (even KeyboardInterrupt) still needs the hard exit
        # below, or datasets' atexit join hangs the script
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


def _run() -> bool:
    """Run all buckets. Returns True when every bucket succeeded."""
    os.makedirs("data", exist_ok=True)
    grams = humaneval_ngrams(load_humaneval())
    print(f"HumanEval n-gram set: {len(grams)} x {NGRAM}-word grams")

    # One gated or down source must not lose the other buckets: each runs,
    # failures are named, and the script still exits non-zero if any failed.
    buckets = [
        ("data/sample_web.jsonl", "text",
         lambda: load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                              split="train", streaming=True)),
        ("data/sample_code.jsonl", "content",
         lambda: load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                              split="train", streaming=True)),
        ("data/sample_math.jsonl", "text",
         lambda: load_dataset("HuggingFaceTB/finemath", "finemath-4plus",
                              split="train", streaming=True)),
    ]
    failures = []
    for path, key, load in buckets:
        try:
            sample(load(), key, path, grams)
        except Exception as e:
            failures.append(path)
            print(f"[fail] {path}: {type(e).__name__}: {e}")
    return not failures


if __name__ == "__main__":
    main()
