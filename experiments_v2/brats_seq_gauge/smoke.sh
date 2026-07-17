#!/usr/bin/env bash
# Build a synthetic BraTS-shaped fixture and run a one-seed pipeline smoke test.
set -euo pipefail

cd "$(dirname "$0")"
PY="${PY:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PY" >&2
  echo "Set PY to an available Python 3 interpreter." >&2
  exit 1
fi

if ! "$PY" -c "import numpy, PIL, torch" >/dev/null 2>&1; then
  echo "BraTS smoke dependencies are missing for $PY." >&2
  echo "Install NumPy, Pillow, and PyTorch, or set PY to an environment that provides them." >&2
  exit 1
fi

"$PY" make_smoke_fixture.py --out data
SMOKE=1 SEEDS="${SEEDS:-0}" PY="$PY" ./run.sh
