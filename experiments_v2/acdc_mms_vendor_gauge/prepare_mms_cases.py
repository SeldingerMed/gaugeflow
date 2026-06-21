#!/usr/bin/env python3
"""Build trainer inputs for the M&Ms vendor-gauge ED->ES prediction experiment.

REQUIRES the M&Ms dataset (Multi-Centre, Multi-Vendor & Multi-Disease cardiac MRI),
which must be registered for and downloaded first -- it is NOT in the quest tmp.

Emits:
  data/cases.jsonl       -- per patient: ED + ES short-axis mid-slice, vendor/centre, split
  data/gauge_pairs.jsonl -- same-patient cross-vendor pairs IF the cohort has paired
                            re-scans; otherwise vendor acts as a cohort-level environment
  data/persistence_metrics.json -- persistence comparator (copy ED as ES), DIAS-style contract

cases.jsonl row schema (DIAS-compatible + acquisition fields):
  {"case_id": "mms_A_017", "patient_id": "mms_017", "dataset": "mms",
   "data_kind": "cine_ed_es", "modality": "MRI", "vendor": "A", "centre": "1",
   "split": "train|val|test", "cluster_id": "mms_017",
   "ed_uri": ".../mms_017_ed.png", "es_uri": ".../mms_017_es.png"}

The ED->ES pair is the prediction task (input ED, target ES) -- the cine analogue of
DIAS adjacent-frame, so the existing prediction trainer + MAE/RMSE/PSNR/SSIM contract
apply unchanged. `vendor` is the observation gauge.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def iter_mms_patients(mms_root: Path):
    """TODO: yield (patient_id, vendor, centre, ed_slice_path, es_slice_path).
    Take one matched short-axis mid-slice at ED and ES per patient from the M&Ms
    nifti volumes + the provided ED/ES frame indices."""
    raise NotImplementedError(
        "Download M&Ms (registration required), then read per-patient ED/ES short-axis "
        "mid-slices + vendor/centre labels from the M&Ms metadata csv."
    )


def split_for(i: int) -> str:
    r = i % 20
    return "train" if r < 14 else ("val" if r < 17 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mms-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cf = open(a.out / "cases.jsonl", "w")
    n = 0
    for i, (pid, vendor, centre, ed, es) in enumerate(iter_mms_patients(a.mms_root)):
        cf.write(json.dumps({
            "case_id": f"mms_{vendor}_{pid}", "patient_id": pid, "dataset": "mms",
            "data_kind": "cine_ed_es", "modality": "MRI", "vendor": vendor, "centre": centre,
            "split": split_for(i), "cluster_id": pid, "ed_uri": str(ed), "es_uri": str(es),
        }) + "\n")
        n += 1
    cf.close()
    print(f"wrote {n} patients -> {a.out}/cases.jsonl "
          f"(gauge_pairs.jsonl + persistence_metrics.json: emit from the same loader)")


if __name__ == "__main__":
    main()
