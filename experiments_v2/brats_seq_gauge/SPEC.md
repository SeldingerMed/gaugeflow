# SPEC — BraTS sequence-gauge, trained segmentation (pre-registered)

## Why
Current BraTS evidence (`A10`, `completed_limited_positive`) is a **representation
probe**: the gauge op is hand-crafted per-sequence mean-centering, scored by
cross-sequence retrieval (0.53→0.60) and sequence-ID separability (0.76→0.0). Two
problems: (1) mean-centering can drive separability to 0 trivially by destroying
signal; (2) the learned rep (0.60) sits far below the label-morphology anchor (0.98),
so there is explicit headroom; (3) there is no task metric and no trained baseline.

## Hypothesis
If pulse sequence is treated as a controllable observation gauge and GaugeFlow's
gauge-consistency/commutator terms are trained on **real** same-case sequence pairs,
the learned state becomes sequence-invariant **without** losing task signal — i.e.
tumor segmentation Dice is preserved or improved while sequence-ID separability stays
≈0. Mean-centering cannot show this (it has no task head).

## Data
- BraTS / MSD Task01 (local: quest `tmp`; the A10 scaleup already fetched 511 MB).
- Modalities = gauge levels: FLAIR, T1w, t1gd, T2w. Native labels: tumor masks → Dice.
- Cluster unit = **case** (all sequences of a case share anatomy). 64+ cases, case-level split.

## Arms (identical except the gauge terms)
- **baseline**: `config_brats_baseline.json` — seg task loss only; `gaugeflow_enabled=false`,
  all `gaugeflow_*_loss_weight=0`.
- **gaugeflow**: `config_brats_gaugeflow.json` — same backbone + Dice head, plus
  `gaugeflow_state_consistency`, `gaugeflow_commutator`, `gaugeflow_transformed_target`,
  `gaugeflow_state_variance` over **real sequence pairs** (not synthetic perturbations).
- **negctrl**: gaugeflow config with `gauge_shuffle=true` — sequence labels permuted
  within case so the consistency term pairs mismatched views. Must NOT beat baseline.

## Metrics
- Primary task: tumor **Dice** (higher better), per case.
- Gauge leakage: adjusted sequence-ID separability of the learned state (≈0 target).
- Headroom reference: label-morphology anchor Dice/retrieval (0.98) — report the gap closed.

## Pre-registered pass gate (`gates.json`)
`Dice delta vs baseline ≥ 0 AND delta 95% CI excludes 0 (non-inferiority→superiority)
AND sequence-ID separability ≤ 0.10 AND negctrl Dice delta ≈ 0 (|Δ| ≤ 0.01)`.
The conjunction is the point: invariance (sep≈0) **with** retained task signal
(Dice≥baseline) is what mean-centering cannot demonstrate.

## Rigor
5 seeds; case-cluster bootstrap CIs; permutation null on the per-case Dice delta;
negative control above. All via `common/analyze.py`.

## New code required (the one adapter hook)
`prepare_brats_cases.py` emits `cases.jsonl` (DIAS schema) + `gauge_pairs.jsonl`
(same-case sequence pairs). Inside `run_cfpath_train.py`, a `gauge_source="real_pairs"`
branch must read `gauge_pairs.jsonl` and feed paired sequences to the existing
gauge-consistency loss instead of generating synthetic perturbation envs. Everything
else (losses, gates, optimiser) is unchanged from `gaugeflow_balance_v2`.

## Claim impact if it passes
Upgrades BraTS from tier-3 (axis exists) to **tier-2 (trained method wins a task on a
real, non-angiographic modality)** — the single biggest generalization-claim jump.
If it fails the gate, report honestly: the gauge axis exists but the trained objective
does not transfer to segmentation, which still bounds the claim.
