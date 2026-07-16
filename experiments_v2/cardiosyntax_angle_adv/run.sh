#!/usr/bin/env bash
# CardioSYNTAX angle-adversarial: raw-baseline vs gaugeflow(+adv) vs negctrl, then analyze.
# Uses gaugeflow_lite + dataset.py reading the LOCAL CardioSYNTAX manifest + shard zips.
#   SMOKE=1 ./run.sh   -> few studies, 1 epoch (proves the pipeline on real data)
#   ./run.sh           -> full run over locally available shards
set -euo pipefail
cd "$(dirname "$0")"; HERE="$(pwd)"
PY="${PY:-python}"
TRAINER="$HERE/../trainer/gaugeflow_lite.py"; DS="$HERE/dataset.py"
SMOKE_FLAG=""; STUDIES="${MAX_STUDIES:-0}"
if [ "${SMOKE:-0}" = "1" ]; then
  SMOKE_FLAG="--smoke"
  SEEDS="${SEEDS:-0 1}"
  STUDIES="${MAX_STUDIES:-12}"
else
  SEEDS="${SEEDS:-0 1 2 3 4}"
fi
RESULTS="$HERE/results.jsonl"; : > "$RESULTS"

emit () {
  local arm="$1" cfg="$2" seed="$3" extra="${4:-}" out="$HERE/runs/$1/seed$3"
  "$PY" "$TRAINER" --config "$cfg" --dataset "$DS" $SMOKE_FLAG \
      --override "seed=$seed" "output_dir=$out" "max_studies=$STUDIES" $extra
  "$PY" - "$out/per_study_metrics.jsonl" "$arm" "$seed" "$RESULTS" <<'PY'
import json,sys; src,arm,seed,dst=sys.argv[1:5]
with open(dst,"a") as o:
    for l in open(src):
        r=json.loads(l)
        o.write(json.dumps({"arm":arm,"seed":int(seed),"cluster":r["study_id"],
                            "metric":r["retrieval_top1"],"leakage":r.get("angle_r2")})+"\n")
PY
}

for s in $SEEDS; do
  emit baseline  "$HERE/config_cardiosyntax_baseline.json"  "$s"
  emit gaugeflow "$HERE/config_cardiosyntax_gaugeflow.json" "$s"
  emit negctrl   "$HERE/config_cardiosyntax_gaugeflow.json" "$s" "gauge_shuffle=true"
done

"$PY" "$HERE/../common/analyze.py" --results "$RESULTS" --gates "$HERE/gates.json" \
    --direction higher --out "$HERE/verdict.json" || true
echo "verdict -> $HERE/verdict.json"
