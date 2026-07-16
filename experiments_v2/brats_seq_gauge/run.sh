#!/usr/bin/env bash
# BraTS sequence-gauge: baseline vs gaugeflow vs negctrl, then analyze.
# Uses the self-contained gaugeflow_lite trainer + dataset.py adapter.
#   SMOKE=1 ./run.sh   -> 1 epoch on the synthetic fixture (proves the pipeline)
#   ./run.sh           -> full run (needs real BraTS via prepare_brats_cases.py first)
set -euo pipefail
cd "$(dirname "$0")"; HERE="$(pwd)"
PY="${PY:-python}"
TRAINER="$HERE/../trainer/gaugeflow_lite.py"; DS="$HERE/dataset.py"
SMOKE_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
  SMOKE_FLAG="--smoke"
  SEEDS="${SEEDS:-0 1}"
else
  SEEDS="${SEEDS:-0 1 2 3 4}"
fi
RESULTS="$HERE/results.jsonl"; : > "$RESULTS"

emit () {
  local arm="$1" cfg="$2" seed="$3" extra="${4:-}" out="$HERE/runs/$1/seed$3"
  "$PY" "$TRAINER" --config "$cfg" --dataset "$DS" $SMOKE_FLAG \
      --override "seed=$seed" "output_dir=$out" "cases_jsonl=$HERE/data/cases.jsonl" $extra
  "$PY" - "$out/per_case_metrics.jsonl" "$arm" "$seed" "$RESULTS" <<'PY'
import json,sys; src,arm,seed,dst=sys.argv[1:5]
with open(dst,"a") as o:
    for l in open(src):
        r=json.loads(l)
        o.write(json.dumps({"arm":arm,"seed":int(seed),"cluster":r["case_id"],
                            "metric":r["dice"],"leakage":r.get("seq_separability")})+"\n")
PY
}

for s in $SEEDS; do
  emit baseline  "$HERE/config_brats_baseline.json"  "$s"
  emit gaugeflow "$HERE/config_brats_gaugeflow.json" "$s"
  emit negctrl   "$HERE/config_brats_gaugeflow.json" "$s" "gauge_shuffle=true"
done

"$PY" "$HERE/../common/analyze.py" --results "$RESULTS" --gates "$HERE/gates.json" \
    --direction higher --out "$HERE/verdict.json" || true
echo "verdict -> $HERE/verdict.json"
