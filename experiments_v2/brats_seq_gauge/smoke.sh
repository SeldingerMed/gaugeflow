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

missing_dependencies=()
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
  missing_dependencies+=("NumPy")
fi
if ! "$PY" -c "import PIL" >/dev/null 2>&1; then
  missing_dependencies+=("Pillow")
fi
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  missing_dependencies+=("PyTorch")
fi
if [ "${#missing_dependencies[@]}" -gt 0 ]; then
  echo "BraTS smoke dependencies missing for $PY: ${missing_dependencies[*]}." >&2
  echo "Install NumPy, Pillow, and PyTorch, or set PY to an environment that provides them." >&2
  exit 1
fi

if [ ! -r make_smoke_fixture.py ]; then
  echo "BraTS smoke fixture generator not found or not readable: $(pwd)/make_smoke_fixture.py" >&2
  exit 1
fi

if [ ! -x run.sh ]; then
  echo "BraTS smoke runner not found or not executable: $(pwd)/run.sh" >&2
  exit 1
fi

if ! mkdir -p data; then
  echo "Unable to create BraTS smoke data directory: $(pwd)/data" >&2
  exit 1
fi

if ! "$PY" make_smoke_fixture.py --out data; then
  echo "Failed to create the BraTS smoke fixture in $(pwd)/data." >&2
  exit 1
fi
SMOKE=1 SEEDS="${SEEDS:-0}" PY="$PY" ./run.sh
