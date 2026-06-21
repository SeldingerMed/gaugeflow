#!/usr/bin/env bash
# Fetch MSD Task01 / BraTS (the raw was purged from quest tmp: raw_temp_deleted=True).
# After download, run prepare_brats_cases.py to emit cases.jsonl, then ./run.sh (no SMOKE).
set -euo pipefail
cd "$(dirname "$0")"
echo "MSD Task01 (BrainTumour) is ~7 GB. Options:"
echo "  1. Medical Segmentation Decathlon: http://medicaldecathlon.com  (Task01_BrainTumour.tar)"
echo "  2. Or reuse the DeepScientist loader that already fetched it:"
echo "     ${DATA_ROOT:-/path/to/data}"
echo
echo "Then:  python prepare_brats_cases.py --brats-root <path/to/Task01_BrainTumour>"
echo "Then:  ./run.sh        # full 5-seed run on real BraTS slices"
