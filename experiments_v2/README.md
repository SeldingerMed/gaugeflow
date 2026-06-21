# experiments_v2 — claim-upgrade experiments

Three experiments that convert GaugeFlow's external evidence from **representation
probes** (retrieval / separability / leakage R²) into **trained-method results**
(task metric vs. a matched no-consistency baseline, with seeds + CI + a negative
control). This is the gap that separates the DIAS win (a trained task result) from
the ACDC / BraTS / CardioSYNTAX checks (axis-existence only).

Each experiment drops into the existing pipeline: the trainer is
`run_cfpath_train.py` (the GaugeFlow config variant), data is a `cases.jsonl` in the
DIAS schema + a frame root, and results are gated against a baseline metric
contract — exactly as `config_gaugeflow_balance_v2.json` runs on DIAS.

## What's shared

- **`common/analyze.py`** — the rigor harness the claim actually needs, modality-agnostic
  and runnable now: per-arm cluster-bootstrap CIs, a permutation null on the
  arm effect, the negative-control delta, and a pre-registered gate check → verdict JSON.
  Run `python common/analyze.py --demo` for the self-check.

## The three experiments (by leverage)

| Dir | Modality / gauge | Task metric | Pre-registered win gate | Claim upgrade |
|---|---|---|---|---|
| `brats_seq_gauge/` | BraTS multi-sequence MRI; gauge = pulse sequence | tumor **Dice** | `seq-ID separability ≈ 0 AND Dice ≥ baseline` (CI-backed) | tier-3 → **tier-2** on a real non-angio modality |
| `cardiosyntax_angle_adv/` | CardioSYNTAX coronary angio; gauge = projection view angle | same-study/same-artery retrieval | `leakage R² ≤ raw (≤0.035) AND retrieval ≥ 0.758` | partial → **clean** angiographic invariance |
| `acdc_mms_vendor_gauge/` | M&Ms cine CMR; gauge = **vendor/centre** (real, not synthetic) | ED→ES prediction MAE (or seg Dice) | `MAE ≤ baseline AND vendor separability ≈ 0` | retires ACDC's "synthetic axis" caveat |

## Per-experiment files

- `SPEC.md` — pre-registered protocol (hypothesis, data, two arms, metrics, gate, seeds, negative control, claim impact, data-access feasibility).
- `config_*_gaugeflow.json` / `config_*_baseline.json` — config deltas on the proven `balance_v2` schema. The GaugeFlow arm sets `gaugeflow_enabled=true`; the baseline arm zeroes the consistency/commutator weights.
- `prepare_*.py` — emits `cases.jsonl` (DIAS schema) + the gauge-pair index for that modality. The dataset adapter is the one piece of new trainer code each experiment needs; the stub documents the exact contract.
- `run.sh` — the two-arm × N-seed invocations against the trainer, plus the shuffled-gauge negative-control run, then `analyze.py`.

## Honest status

These are **scaffolds**, not completed runs. Each needs: (1) the dataset downloaded
(BraTS + CardioSYNTAX archives are local in the quest tmp; **M&Ms needs registration/
download**), and (2) a small dataset-adapter hook inside `run_cfpath_train.py` that
reads the modality's `cases.jsonl` and samples gauge-paired views. Everything else —
configs, gates, seeds, CI, negative control, analysis — is specified and the analysis
harness is runnable today.

## Run order

`brats_seq_gauge` first (biggest claim jump, local data, native task with labels),
then `cardiosyntax_angle_adv` (cheap, closes a named failure), then
`acdc_mms_vendor_gauge` (best real-acquisition axis, but M&Ms download required).
