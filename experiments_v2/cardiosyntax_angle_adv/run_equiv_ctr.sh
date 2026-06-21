#!/usr/bin/env bash
# CardioSYNTAX gauge-EQUIVARIANCE test: baseline vs equiv(predict z_b from z_a+Δangle, no GRL) vs negctrl.
# Mirrors run.sh. negctrl shuffles partners so Δangle is meaningless -> equivariance term should go inert.
#   SMOKE=1 ./run_equiv.sh   -> few studies, 1 epoch (pipeline check)
#   ./run_equiv.sh           -> full run over locally available shards
set -euo pipefail
cd "$(dirname "$0")"; HERE="$(pwd)"
PY="${PY:-python}"
TRAINER="$HERE/../trainer/gaugeflow_lite.py"; DS="$HERE/dataset.py"
SMOKE_FLAG=""; SEEDS="${SEEDS:-0 1 2 3 4}"; STUDIES="${MAX_STUDIES:-0}"
if [ "${SMOKE:-0}" = "1" ]; then SMOKE_FLAG="--smoke"; SEEDS="${SEEDS:-0 1}"; STUDIES="${MAX_STUDIES:-12}"; fi
RESULTS="$HERE/results_equiv_ctr.jsonl"; : > "$RESULTS"

emit () {
  local arm="$1" cfg="$2" seed="$3" extra="${4:-}" out="$HERE/runs_equiv_ctr/$1/seed$3"
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
  emit baseline  "$HERE/config_cardiosyntax_baseline.json" "$s"
  emit gaugeflow "$HERE/config_cardiosyntax_equiv_ctr.json"     "$s"   # treatment arm = equivariance (named for analyze.py)
  emit negctrl   "$HERE/config_cardiosyntax_equiv_ctr.json"     "$s" "gauge_shuffle=true"
done

"$PY" "$HERE/../common/analyze.py" --results "$RESULTS" --gates "$HERE/gates.json" \
    --direction higher --out "$HERE/verdict_equiv_ctr.json" || true
echo "verdict -> $HERE/verdict_equiv_ctr.json"
