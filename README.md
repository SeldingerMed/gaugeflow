# GaugeFlow

**Equivariance, not invariance, is the right prior for continuous physical gauges in self-supervised medical imaging.**

A self-supervised objective that helps on one medical-imaging task often collapses on the next. GaugeFlow gives a *mechanistic* account of when it transfers, organized around one distinction: whether a task's nuisance axis is a **continuous physical gauge** with coincident free supervision, and whether the downstream utility is **invariant or covariant** under that gauge.

This repository contains the portable trainer, dataset adapters, pre-registered analyzer, and the complete per-seed evidence behind the [preprint](paper/main.pdf).

---

## The result in one table

Projection-angle gauge (CardioSYNTAX coronary angiography), same-study/same-artery retrieval@1, 5 seeds:

| Objective | Retrieval Δ vs. baseline | Gauge used? (true vs. shuffled Δangle) | Angle-leakage R² |
|---|---|---|---|
| Adversarial invariance (GRL, single embedding) | **−0.044**, *p*≈0 | — | 0.31 |
| Adversarial invariance (GRL, dual content/style path) | −0.014 | — | 0.09 |
| **Equivariance, auxiliary** (SO(2) Δangle head) | **−0.018**, *p*=0.11 (n.s.) | 0.079 vs. 0.028 — **used** | 0.30 |
| Equivariance, primary (canonical eval) | −0.080, *p*≈0 | 0.0089 vs. 0.0105 — none | 0.70 |
| Equivariance, primary (pairwise-transport eval) | −0.050, *p*≈0 | 0.0093 vs. 0.0089 — none | 0.73 |

Leakage cap is 0.035 (raw-feature reference). Per-run baselines drift ~0.014 (MPS nondeterminism); the valid statistic is the **within-run paired delta**.

**Reading:** adversarial invariance is a no-free-lunch wall — no adversary weight holds retrieval non-inferior *and* meets the leakage cap. Gauge-equivariance **dominates it as an auxiliary** (removes the significant harm, demonstrably uses the gauge) but **fails as a primary objective**, because same-study-across-angles retrieval is itself an invariance metric and the projection-angle gauge is not a recoverable group action on the encoder — confirmed protocol-independent by the pairwise-transport evaluation.

## The decision rule

Gauge-equivariant self-supervision is the right prior **iff**:

1. the nuisance is a **continuous physical gauge** with an observed parameter (e.g. positioner angle, contrast phase — not a discrete style/vendor bucket), **and**
2. the downstream utility is **covariant** under the gauge (depends on it), **and**
3. the gauge acts as a **recoverable transformation** of the representation.

When the utility is invariance-shaped, no gauge machinery — adversarial or equivariant — beats simply not fighting the gauge. All three conditions are checkable *before* a full training run: (1) from metadata, (2) from the task definition, (3) with a transport-vs-shuffle control.

DIAS contrast-front prediction satisfies all three (the future frame is gauge-covariant free supervision) and the gauge-consistent objective helps there. Projection-angle retrieval satisfies (1) but not (2)–(3), and it does not.

---

## Layout

```
paper/                       preprint (LaTeX source + compiled PDF + bib)
experiments_v2/
  trainer/gaugeflow_lite.py  portable trainer: baseline · GRL adversary (single + dual-path)
                             · SO(2) equivariance head (auxiliary) · gauge-aligned contrastive (primary)
  common/analyze.py          pre-registered analyzer: cluster-bootstrap CI, permutation null, shuffled-gauge control
  cardiosyntax_angle_adv/    projection-angle gauge: dataset adapter, configs, run scripts, per-seed verdicts
  brats_seq_gauge/           pulse-sequence gauge (MSD Task01/BraTS)
  RESULTS_v2.md              full verdicts
```

Every objective is a config flag on one trainer; the no-op config reproduces the baseline exactly.

## Reproduce

The trainer needs only PyTorch + NumPy.

```bash
cd experiments_v2/cardiosyntax_angle_adv

./run.sh            # baseline vs. GRL adversary vs. shuffled-gauge control
./run_equiv.sh      # baseline vs. SO(2) equivariance auxiliary vs. control
./run_equiv_ptr.sh  # equivariance as primary objective, pairwise-transport eval
```

Each script writes per-seed `per_study_metrics.jsonl` and a `verdict_*.json` from the analyzer. The four `verdict_*.json` in `cardiosyntax_angle_adv/` are the numbers in the table above.

The key flags on `gaugeflow_lite.py` (config JSON):

| flag | objective |
|---|---|
| `angle_adv_loss_weight` | gradient-reversal adversary (invariance) |
| `gauge_equiv_loss_weight` | SO(2) equivariance head, auxiliary |
| `gauge_equiv_contrastive` | gauge-aligned contrastive, equivariance as primary |
| `pairwise_transport_retrieval` | rigorous transport-frame retrieval at eval |
| `gauge_shuffle` | shuffled-gauge negative control |

## Data

The CardioSYNTAX, DIAS, and MSD Task01/BraTS imaging data are licensed by their respective providers and are **not redistributed here**. The dataset adapters read local shards; point them at your licensed copy. The complete per-seed *evidence* (metrics + verdicts) is included so every reported number is independently checkable without rerunning, and is also mirrored as a dataset on the Hugging Face Hub.

## Citation

```bibtex
@misc{son2026gaugeflow,
  title  = {GaugeFlow: Equivariance, Not Invariance, Is the Right Prior for
            Continuous Physical Gauges in Self-Supervised Medical Imaging},
  author = {Son, Colin},
  year   = {2026},
  note   = {Seldinger, Inc.}
}
```

## License

MIT — see [LICENSE](LICENSE).
