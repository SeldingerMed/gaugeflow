#!/usr/bin/env bash
# M&Ms vendor-gauge ED->ES prediction: baseline vs gaugeflow vs negctrl, 5 seeds.
# Pre-req: download M&Ms, run prepare_mms_cases.py, set GAUGEFLOW_TRAINER.
set -euo pipefail
cd "$(dirname "$0")"; HERE="$(pwd)"
TRAINER="${GAUGEFLOW_TRAINER:?set GAUGEFLOW_TRAINER=/path/to/coin_obstopo/run_cfpath_train.py}"
PY="${PY:-python3}"; SEEDS="${SEEDS:-0 1 2 3 4}"
RESULTS="$HERE/results.jsonl"; : > "$RESULTS"

emit () {
  local arm="$1" cfg="$2" seed="$3" out="$HERE/runs/$1/seed$3"
  "$PY" "$TRAINER" --config "$cfg" --override "seed=$seed" "output_dir=$out" "run_id=run-mms-$arm-seed$seed"
  # adapter writes out/per_patient_metrics.jsonl: patient_id, ed_es_mae, vendor_separability
  "$PY" - "$out/per_patient_metrics.jsonl" "$arm" "$seed" "$RESULTS" <<'PY'
import json,sys; src,arm,seed,dst=sys.argv[1:5]
with open(dst,"a") as o:
    for l in open(src):
        r=json.loads(l)
        o.write(json.dumps({"arm":arm,"seed":int(seed),"cluster":r["patient_id"],
                            "metric":r["ed_es_mae"],"leakage":r.get("vendor_separability")})+"\n")
PY
}

for s in $SEEDS; do
  emit baseline  "$HERE/config_mms_baseline.json"  "$s"
  emit gaugeflow "$HERE/config_mms_gaugeflow.json" "$s"
  tmp="$HERE/.cfg_negctrl_$s.json"
  "$PY" -c "import json;c=json.load(open('$HERE/config_mms_gaugeflow.json'));c['gauge_shuffle']=True;c['run_id']='run-mms-negctrl-seed$s';json.dump(c,open('$tmp','w'))"
  emit negctrl "$tmp" "$s"
done

"$PY" "$HERE/../common/analyze.py" --results "$RESULTS" --gates "$HERE/gates.json" \
    --direction lower --out "$HERE/verdict.json"
echo "verdict -> $HERE/verdict.json"
