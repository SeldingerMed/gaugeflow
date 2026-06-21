# experiments_v2 — REAL full-run verdicts (2026-06-19)

Trainer: `trainer/gaugeflow_lite.py` (faithful CORE: shared encoder + gauge-consistency
MSE + state-variance + angle-adversarial GRL; gauge_shuffle = negative control).
Omits the full DIAS trainer's commutator/prototype/style machinery.

## CardioSYNTAX — projection-angle gauge, retrieval@1 (real local shards, 5 seeds)
- baseline retrieval@1 = 0.0821 (n=203 clusters)
- gaugeflow retrieval@1 = 0.0466 (n=200)
- negctrl   retrieval@1 = 0.0527 (n=198)
- gaugeflow − baseline = **−0.0442**, CI95 [−0.0634, −0.0248], perm-p = 0.0
- angle-leakage R² (gaugeflow) = 0.314 (cap 0.035)
- gates: delta_meets_min FAIL, leakage_under_cap FAIL, negctrl_is_null FAIL → **PASS=false**
- Reading: lite-GaugeFlow SIGNIFICANTLY DEGRADES angle-gauge retrieval and does not
  control angle leakage. Negative transfer.

## BraTS — pulse-sequence gauge, tumour Dice (real MSD Task01, 48 cases, 5 seeds)
- baseline Dice = 0.3710, gaugeflow Dice = 0.3707, negctrl Dice = 0.3684 (n=6 test/seed)
- gaugeflow − baseline = **−0.0003**, CI95 [−0.0019, 0.0014], perm-p = 0.78
- sequence-ID adjusted separability = **0.0 for ALL arms** (baseline already gauge-invariant)
- gates: delta_meets_min FAIL (delta≈0), delta_ci_excludes_0 FAIL, leakage_under_cap PASS,
  negctrl_is_null PASS → **PASS=false**
- Reading: lite-GaugeFlow is INERT for pulse-sequence segmentation — no Dice gain, and the
  baseline encoder is already sequence-invariant so the gauge terms have nothing to fix. Null.

## Bottom line
Under a faithful compact reproduction, the GaugeFlow contrast-dynamics benefit does NOT
transfer to the projection-angle (hurts) or pulse-sequence (inert) gauges. This BOUNDS the
mechanism to the contrast-phase angiographic setting it was designed for; it does not
support a broad cross-gauge / cross-modality generalization claim. Earlier published
external rows used different hand-crafted DESCRIPTOR pipelines, not this neural trainer.

## BraTS data prep
`prepare_brats_cases.py` streams MSD Task01 from the public S3 tarball via HTTP Range
(reuses quest-012 `stream_msd_task01_schema.py`, + retry/backoff), takes the max-tumour
axial slice per case across the 4 sequences + seg, deletes raw NIfTI. Bug fixed 2026-06-19:
binary masks must NOT be percentile-normalized (collapsed them to empty → degenerate Dice=1.0).

## Chase-a-positive diagnostics (2026-06-19)

### Full trainer port is INFEASIBLE as a config swap
`coin_obstopo/run_cfpath_train.py` (the 3619-line full trainer) is hardwired to DIAS:
`front.build_pair_records`/`cache_pairs` cache DIAS frame->next-frame pairs; the "gauge"
is a SYNTHETIC perturbation of DIAS frames (`perturb_environment`/`style_augment`); task =
DIAS next-frame prediction + vessel/front-mask seg; outputs DIAS metrics. There is NO
`gauge_source=real_pairs` key, NO external-dataset adapter, NO retrieval/seg head for
CardioSYNTAX/BraTS. The commutator/style-prototype terms are defined over a parameterized
continuous perturbation operator — projection-angle and pulse-sequence pairs are DISCRETE
real gauges with no such operator, so those terms don't even define without redesign.
STATUS.md's "configs transfer unchanged" is inaccurate. A real port is a multi-day rebuild.

### CardioSYNTAX ablation: the GRL caused the negative; mechanism is null
Cheap lever (lite trainer, 5 seeds, real shards):
- baseline retr@1 = 0.0821
- full config (angle_adv GRL w=0.25): -0.044, perm_p~0  -> SIGNIFICANTLY HURTS
- consistency-only (angle_adv=0): retr@1 0.0932, delta +0.0055, CI95 [-0.0123, 0.0237],
  perm_p 0.545 -> NULL (non-inferior, NOT a significant positive); leakage 0.265 (cap 0.035, FAIL)
- low-adv (angle_adv=0.05): retr@1 0.0442 -> still hurts (GRL harmful at any weight here)

Conclusion: removing the angle-adversarial term removes the harm but does NOT yield a
positive. Under the lite trainer the gauge-consistency mechanism is INERT on both external
gauges (CardioSYNTAX retrieval, BraTS Dice) and controls neither angle-leakage nor adds task
signal. No positive transfer was found by cheap means. The only remaining route to a positive
is the full-trainer rebuild above (multi-day, uncertain applicability).

## Full-machinery rebuild: gaugeflow_dualpath.py (2026-06-19)

Built `trainer/gaugeflow_dualpath.py` — ports the transferable part of the canonical DIAS
trainer that lite lacked: dual content(z_topo)/style(z_style) paths + style-prototype EMA bank
+ GRL style-classifier on the CONTENT path + style-absorb head on the STYLE path. Task uses
z_topo. DROPPED by design: commutator/operator/next-frame terms (world-model DYNAMICS; undefined
for static retrieval/seg with no temporal transition). Same dataset contract / probes / gates / output.

### Result vs lite (5 seeds each)
CardioSYNTAX (retrieval@1 / angle-R2 leakage):
- lite gaugeflow: delta -0.0442 (p~0, HURTS), leakage ~0.31 (no reduction vs baseline 0.28)
- dualpath: baseline 0.0503, gaugeflow 0.0330, delta -0.0141 CI[-0.0285,0.0002] p=0.057;
  angle-leakage baseline 0.274 -> gaugeflow **0.092** (3x reduction). gates FAIL (leakage cap 0.035, retrieval still <0).
- READING: the lite negative WAS the single-embedding GRL. Dual content/style separation
  restores leakage control (the GaugeFlow thesis) and softens the retrieval harm, but a residual
  accuracy/invariance tradeoff remains; no clean PASS.

BraTS (Dice / sequence-separability):
- lite: delta -0.0003 (p=0.78, null), sep 0.0 (baseline already invariant)
- dualpath: baseline 0.3712, gaugeflow 0.3760, delta +0.0048 CI[0.0001,0.0099] p=0.157 (bootstrap-sig,
  perm borderline); BUT content separability baseline 0.0 -> gaugeflow 0.167 (richer content retains
  MORE gauge info; fails cap 0.10). Marginal Dice gain at the cost of invariance. Not a clean win.

### Bottom line
The rebuild moved the needle (esp. CardioSYNTAX leakage 0.27->0.09), confirming the dual-path
machinery matters and the lite negative was an artifact. But NO experiment clears its pre-registered
gate. Next lever: sweep the GRL weight for a non-inferior-retrieval + low-leakage sweet spot.

### GRL-weight sweep (CardioSYNTAX dualpath, 5 seeds) — the tradeoff is monotonic
| topo_style_grl_weight | retrieval delta vs baseline | angle-leakage |
|---|---|---|
| 0.03 | +0.0059 (CI[-0.012,0.022], p=0.50, non-inferior) | 0.432 (no invariance) |
| 0.05 | -0.0064 (p=0.47, null) | 0.260 |
| 0.10 | -0.0141 (CI[-0.029,0.0002], p=0.057) | 0.092 (best) |

More GRL -> lower leakage, worse retrieval, monotonically. NO weight gives non-inferior
retrieval AND leakage <= 0.035 cap. The accuracy/invariance tradeoff has no PASS point in range.

## FINAL (quest-012 experiments_v2, all real data)
Under faithful reproduction at two fidelities (lite single-embedding AND full-machinery
dual content/style), GaugeFlow's gauge-consistency does NOT cleanly transfer to the
projection-angle (CardioSYNTAX) or pulse-sequence (BraTS) gauges: no experiment clears its
pre-registered gate. The dual-path machinery is NOT cosmetic — it restores leakage control
(CardioSYNTAX 0.27->0.09) and converts the lite negative into a characterized accuracy/
invariance tradeoff — but the contrast-phase/DIAS benefit remains gauge-type-specific.
This is a clean, defensible BOUNDED result, not a broad-generalization positive.
