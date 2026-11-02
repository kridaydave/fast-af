import json
import os

from datasets import load_dataset

N_DOCS = 2000

os.makedirs("data", exist_ok=True)

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
)

n_written = 0
with open("data/sample_web.jsonl", "w") as f:
    for i, doc in enumerate(dataset):
        if i >= N_DOCS:
            break
        f.write(json.dumps({"text": doc["text"]}) + "\n")
        n_written += 1

print(f"wrote {n_written} docs to data/sample_web.jsonl")
for doc_text in [json.loads(line)["text"] for line in open("data/sample_web.jsonl").readlines()[:2]]:
    print("---", doc_text[:200].replace("\n", " "))


# Code

dataset_code = load_dataset("bigcode/the-stack-dedup",data_dir="data/python",split="train",streaming=True)
n_written_code = 0
with open("data/sample_code.jsonl", "w") as f:
    for i, doc in enumerate(dataset_code):
        if i >= N_DOCS:
            break
        f.write(json.dumps({"text": doc["content"]}) + "\n")
        n_written_code += 1

print(f"wrote {n_written_code} docs to data/sample_code.jsonl")
for doc_text in [json.loads(line)["text"] for line in open("data/sample_code.jsonl").readlines()[:2]]:
    print("---", doc_text[:200].replace("\n", " "))


# Math
dataset_math = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
n_written_math = 0
with open("data/sample_math.jsonl", "w") as f:
    for i, doc in enumerate(dataset_math):
        if i >= N_DOCS:
            break
        f.write(json.dumps({"text": doc["text"]}) + "\n")
        n_written_math += 1

print(f"wrote {n_written_math} docs to data/sample_math.jsonl")
for doc_text in [json.loads(line)["text"] for line in open("data/sample_math.jsonl").readlines()[:2]]:
    print("---", doc_text[:200].replace("\n", " "))
