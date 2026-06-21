# SPEC — M&Ms vendor-gauge cine CMR, trained prediction (pre-registered)

## Why
ACDC is the strongest non-angiographic positive (fixed-state retrieval top-1 1.0,
margin 0.37), but its result report flags the binding caveat itself: the observation
axis is **synthetic perturbations** — ACDC has no scanner/vendor/protocol metadata —
and the metric is frozen-descriptor retrieval (already saturated at 1.0, no headroom).

## Hypothesis
On **M&Ms** (multi-vendor, multi-centre cine CMR with real vendor/centre labels),
treating **vendor/centre as the observation gauge** and training GaugeFlow's
gauge-consistency terms improves an **end-diastole→end-systole frame-prediction** task
(MAE) vs. a matched no-consistency baseline, while driving vendor separability of the
learned state to ≈0. This retires the "synthetic axis, saturated retrieval" caveat in
one move: a real acquisition gauge + a trained task metric with headroom.

## Data
- **M&Ms** cine CMR (Multi-Centre, Multi-Vendor & Multi-Disease). **Requires
  registration/download** — not in the quest tmp (it is an unrun `latest_run=none`
  backlog idea: `coin-mms-cine-cmr`). This is the one feasibility blocker among the three.
- Gauge levels: vendor (A/B/C/D) and/or centre. Cluster unit = **patient**.
- Task: ED→ES short-axis frame prediction (the cine analogue of DIAS adjacent-frame),
  so the existing prediction trainer + metric contract (MAE/RMSE/PSNR/SSIM) apply directly.

## Arms
- **baseline**: prediction task loss only; `gaugeflow_enabled=false`.
- **gaugeflow**: + gauge-consistency/commutator/state-variance over **same-patient,
  cross-vendor** view pairs (real acquisition gauge, not synthetic perturbations).
- **negctrl**: `gauge_shuffle=true` — vendor labels permuted across patients.

## Metrics
- Primary: ED→ES prediction **MAE** (lower better), per patient; report RMSE/PSNR/SSIM too.
- Gauge leakage: vendor-ID separability of the learned state (≈0 target).
- Comparator: a per-patient persistence baseline (copy ED as ES), mirroring DIAS.

## Pre-registered pass gate (`gates.json`)
`MAE delta vs baseline ≤ 0 (improves) AND delta 95% CI excludes 0 AND vendor
separability ≤ 0.10 AND negctrl MAE delta ≈ 0`.

## Rigor
5 seeds; patient-cluster bootstrap CIs; permutation null on the per-patient MAE delta;
shuffled-vendor negative control. Via `common/analyze.py --direction lower`.

## New code required
1. **Data acquisition** — download M&Ms (registration); biggest blocker.
2. `prepare_mms_cases.py` — emit `cases.jsonl` (per patient: ED/ES frames, vendor label,
   centre, split) + `gauge_pairs.jsonl` (same-patient cross-vendor pairs *if* paired
   acquisitions exist; otherwise use the vendor label as an environment for the
   consistency term across the cohort).
3. Reuse the DIAS prediction trainer + metric contract unchanged (ED→ES = adjacent-frame).

## Claim impact if it passes
Retires ACDC's synthetic-axis limitation and gives a **real-acquisition-gauge,
trained-task** CMR result — the cleanest possible non-angiographic generalization
evidence. If M&Ms cannot be obtained, the fallback (ACDC ED→ES trained prediction with
disease-group environments) still upgrades ACDC from retrieval to a trained task, but
keeps the axis semi-synthetic — note that explicitly.
