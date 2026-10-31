# Colab free T4 setup for benchmark week.
# Run on a T4 runtime (sm75, 16 GB). T4 has no bf16, no AWQ.

set -e

# --- torch (CUDA 12.x wheel for the T4) ---
pip install --quiet --no-cache-dir \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# --- vLLM for the AR baselines ---
# Pin a vLLM version known to run on sm75; newer wheels drop Turing.
pip install --quiet --no-cache-dir "vllm==0.6.3.post2"

# --- dLLM for the diffusion models (LLaDA / Dream) ---
pip install --quiet --no-cache-dir \
  "git+https://github.com/ZHZisZZ/dllm.git"

# --- misc ---
pip install --quiet --no-cache-dir pyyaml datasets accelerate safetensors

git clone https://github.com/openai/human-eval.git || true

echo "Setup complete. Run: python -m bench --config bench/config.yaml --out results.yaml"
