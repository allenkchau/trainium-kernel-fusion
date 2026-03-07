#!/usr/bin/env bash
set -euo pipefail

# Pre-compile and smoke test NKI kernels on Trainium.
# Default target is trn2 since this repo is now tested there.

TARGET="trn2"
SEQLEN="512"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --seqlen)
      SEQLEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: $0 [--target trn2|trn1] [--seqlen 512|1024|...]"
      exit 1
      ;;
  esac
done

if [[ "$TARGET" != "trn1" && "$TARGET" != "trn2" ]]; then
  echo "Unsupported target '$TARGET'. Use trn1 or trn2."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Preserve user flags and append target if missing.
TARGET_FLAG="--target=${TARGET}"
if [[ " ${NEURON_CC_FLAGS:-} " != *" ${TARGET_FLAG} "* ]]; then
  export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:-} ${TARGET_FLAG}"
fi

echo "[compile] ROOT_DIR=${ROOT_DIR}"
echo "[compile] NEURON_CC_FLAGS=${NEURON_CC_FLAGS}"
echo "[compile] seqlen=${SEQLEN}"
export SEQLEN_OVERRIDE="${SEQLEN}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Neither 'python' nor 'python3' was found in PATH."
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import os
import numpy as np

from kernels.baseline.attention import baseline_attention
from kernels.fused.attention import fused_attention
from kernels.baseline.softmax import softmax_kernel

seqlen = int(os.environ.get("SEQLEN_OVERRIDE", "512"))
d_head = 128

rng = np.random.default_rng(0)
q = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(np.float32)
k = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(np.float32)
v = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(np.float32)
x = rng.normal(size=(1024, seqlen)).astype(np.float32)

# First call triggers NKI JIT compile + execute.
baseline_attention(q, k, v)
fused_attention(q, k, v)
softmax_kernel(x)

print("[compile] PASS: baseline_attention, fused_attention, softmax_kernel")
PY
