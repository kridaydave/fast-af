#!/usr/bin/env bash
set -euo pipefail

# Colab/Kaggle bootstrap for the benchmark harness.
#
# The harness used to be duplicated below as heredocs. That copy drifted from
# bench/ and silently overwrote newer sampler/cache/accounting code. This
# script now installs dependencies and runs the checked-out repository only.

REPO_URL="${BENCH_REPO_URL:-https://github.com/kridaydave/fast-af.git}"
REPO_REF="${BENCH_REF:-master}"

if [ -d .git ] && [ -f bench/config.yaml ]; then
  REPO_DIR="$PWD"
else
  REPO_DIR="${BENCH_REPO_DIR:-$PWD/fast-af}"
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
  fi
  git -C "$REPO_DIR" fetch origin "$REPO_REF"
  git -C "$REPO_DIR" checkout "$REPO_REF"
  git -C "$REPO_DIR" merge --ff-only FETCH_HEAD
fi

cd "$REPO_DIR"

echo "[1/3] Installing benchmark dependencies."
pip install --quiet --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install --quiet --no-cache-dir \
  "transformers>=5.6.0,<6.0" pyyaml datasets accelerate safetensors \
  bitsandbytes tokenizers huggingface_hub pytest

pip install --quiet --no-cache-dir flash-linear-attention \
  || echo "WARN: flash-linear-attention failed; Qwen3.5 GDN may use fallback kernels."
TORCH_CUDA_ARCH_LIST="7.5" pip install --quiet --no-cache-dir causal-conv1d \
  > /tmp/causal_conv1d_install.log 2>&1 \
  || echo "WARN: causal-conv1d failed. Log: /tmp/causal_conv1d_install.log"
pip install --quiet --no-cache-dir --no-deps \
  "git+https://github.com/ZHZisZZ/dllm.git" \
  > /tmp/dllm_install.log 2>&1 \
  && pip install --quiet --no-cache-dir --prefer-binary \
     sentencepiece tyro omegaconf rich torchmetrics tqdm \
  || echo "WARN: dLLM failed. Tier 3 remains unavailable. Log: /tmp/dllm_install.log"

echo "[2/3] Verifying the repository harness."
python -m py_compile bench/*.py
python -m pytest -q tests/test_backends.py tests/test_run.py
git rev-parse HEAD

echo "[3/3] Harness ready at $REPO_DIR"
echo "Run from this checkout: cd $REPO_DIR"
echo "Smoke: python -m bench --only smollm2-135m --out results.yaml"
echo "Full:  python -m bench --config bench/config.yaml --out results.yaml"

case "${BENCH_RUN:-none}" in
  smoke)
    python -m bench --only smollm2-135m --out results.yaml
    ;;
  full)
    python -m bench --config bench/config.yaml --out results.yaml
    ;;
  none)
    ;;
  *)
    echo "BENCH_RUN must be one of: none, smoke, full" >&2
    exit 2
    ;;
esac
