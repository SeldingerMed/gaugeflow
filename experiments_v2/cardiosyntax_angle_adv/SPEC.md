# SPEC — CardioSYNTAX view-angle invariance, adversarial repair (pre-registered)

## Why
CardioSYNTAX is the strongest *external angiographic* surface, but the 80-pair
balanced scaleup (`A6`) has a **named failure**: learned fold-fit angle-leakage R²
is below pooled features (0.21 vs 0.59) but **not below raw** (0.21 vs **0.035**).
Raw features already leak less than the learned gauge — so the current "win" is only a
partial repair. Retrieval signal is positive (same-study/same-artery 0.515→0.758).

## Hypothesis
Adding an explicit **angle-adversarial / orthogonality** term to the gauge objective
drives learned angle-leakage R² to or below the raw-feature level (≤0.035) **while
retaining** same-study/same-artery retrieval (≥0.758). Closing the raw gap converts a
partial repair into a clean view-angle-invariance result.

## Data
- CardioSYNTAX external coronary angiography (local archive in quest `tmp`; reuse the
  validated `analysis-373cf226` balanced-pair feature extraction unchanged).
- Unit/cluster = **study** (a study's videos share patient/anatomy). 80 balanced
  same-study/same-artery pairs / 160 videos / 80 studies (the published scale).

## Arms
- **baseline (raw)**: raw extracted features, no gauge fitting. This is the bar to beat
  on *both* metrics — leakage 0.035, retrieval ~raw.
- **gaugeflow**: learned gauge + `orthogonality_loss_weight` + new
  `angle_adv_loss_weight` (gradient-reversal angle predictor on the state code).
- **negctrl**: gaugeflow with `gauge_shuffle=true` (permute angle labels across studies)
  — the adversarial term should then neither help retrieval nor reduce leakage below raw.

## Metrics
- Primary: same-study/same-artery **retrieval@1** (higher), per study (leave-one-study-out).
- Leakage: fold-fit **angle-leakage R²** of the learned state (lower; raw=0.035 is the bar).

## Pre-registered pass gate (`gates.json`)
`retrieval delta vs raw ≥ 0 AND delta 95% CI excludes 0-downside (non-inferior)
AND angle-leakage R² ≤ 0.035 AND negctrl retrieval delta ≈ 0`.
The binding clause is `leakage ≤ 0.035` — beating *raw*, not just pooled.

## Rigor
5 seeds (feature-fit + adversarial training stochasticity); study-cluster bootstrap
CIs; permutation null on the per-study retrieval delta; shuffled-angle negative
control. Via `common/analyze.py`.

## New code required
`prepare_cardiosyntax_pairs.py` emits `pairs.jsonl` (study, artery, angle label, two
video feature URIs). The adversarial head is a small addition to the gauge fitter in
the `analysis-373cf226` implementation (gradient-reversal layer + angle MLP); the
feature extractor and retrieval metric are reused verbatim so the result is comparable
to the published 80-pair row.

## Claim impact if it passes
Turns CardioSYNTAX from "partial leakage repair" into a **clean view-angle-invariance
result on real external coronary angiography** — the best non-DIAS angiographic
evidence. If leakage stays above raw, report it as a bounded negative: gauge-consistency
improves retrieval but does not beat raw features on angle leakage.
