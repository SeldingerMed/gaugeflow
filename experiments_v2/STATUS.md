# RUN STATUS — experiments_v2 (BraTS + CardioSYNTAX wired and runnable)

The adapter hook is built as a self-contained trainer (`trainer/gaugeflow_lite.py`):
shared state encoder + task head (segmentation or embedding/retrieval) +
gauge-consistency / state-variance losses over **real paired views** + shuffled-gauge
negative control + gauge-leakage probe. It reads the same config schema and emits
per-cluster metrics into `common/analyze.py`. No model is trained on test data.

## Verified (smoke = wiring proof, not a scientific result)

| Experiment | Data | Smoke run | What ran |
|---|---|---|---|
| `cardiosyntax_angle_adv` | **REAL, local** (manifest + shard zips) | ✅ `SMOKE=1 ./run.sh` (12 studies × 3 arms × 2 seeds, ~96 s) | zip `.npy` frames → encoder + angle-adversarial head → retrieval@1 + angle-R² leakage → CI + permutation null + gate → `verdict.json` |
| `brats_seq_gauge` | raw **purged** (`fetch_brats.sh`) | ✅ `SMOKE=1 ./run.sh` on synthetic fixture | seg + sequence-gauge consistency → Dice + sequence-separability → gate → `verdict.json` |

Both smoke verdicts are `PASS=false` by design: 1 epoch / tiny n / 64 px is noise.
The value is that the full path (load → train 3 arms → eval → stats → gate) executes
on real code; the science needs the full run below.

## Full runs

**CardioSYNTAX (runnable now, real local data):**
```
cd cardiosyntax_angle_adv
PY=/path/to/python ./run.sh          # all locally available shards, 5 seeds, full epochs
cat verdict.json
```
Note: only the shard zips present on disk are used (the loader skips videos whose
manifest path points at a missing shard). For the published 80-pair scale, ensure the
relevant shards are downloaded.

**BraTS (one download away):**
```
cd brats_seq_gauge
./fetch_brats.sh                                   # get MSD Task01
python prepare_brats_cases.py --brats-root <path>  # emit real cases.jsonl
./run.sh                                            # full 5-seed run
```

## What is faithful vs. simplified

`gaugeflow_lite` implements the GaugeFlow *core* (state-consistency + variance +
angle-adversarial gauge terms, real paired views, gauge-shuffle control). It is **not**
the full 2800-line DIAS trainer — the commutator term and prototype/style machinery are
not reproduced. For a headline submission result, port these two configs into the
canonical `run_cfpath_train.py` with the same `gauge_source="real_pairs"` data path;
the dataset adapters and the pre-registered gates here transfer unchanged.
