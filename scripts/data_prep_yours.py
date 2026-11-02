import huggingface_hub
from datasets import load_dataset
import os

dataset = load_dataset("HuggingFaceFW/fineweb",name="sample-10BT",split="train",streaming=True)

# Wrap
for i,doc in enumerate(dataset):
    if i >= 2000:
        break
os.makedirs("data", exist_ok=True)
print(doc)
