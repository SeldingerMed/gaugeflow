#!/usr/bin/env python3
"""Build pairs.jsonl for the CardioSYNTAX angle-adversarial experiment.

Reuses the validated analysis-373cf226 balanced-pair selection + feature extraction.
This stub only documents the row contract; fill iter_balanced_pairs() against the
local CardioSYNTAX archive and the existing extractor.

pairs.jsonl row schema:
  {"study_id": "s0123", "artery": "LAD", "angle_label": 7,
   "video_a_uri": ".../s0123_LAD_a.npy", "video_b_uri": ".../s0123_LAD_b.npy"}
  (same study + same artery, different projection angle = the gauge pair;
   angle_label is the discretised view angle used by the adversarial/leakage probe)

The 80-pair balanced selection (160 videos / 80 studies) must match the published
A6 scaleup so the new result is comparable to the existing CardioSYNTAX row.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def iter_balanced_pairs(cardiosyntax_root: Path, pair_limit: int = 80):
    """TODO: reuse analysis-373cf226 balanced-pair selection + feature extraction.
    Yield (study_id, artery, angle_label, feat_a_path, feat_b_path) for `pair_limit`
    same-study/same-artery pairs (balanced over angle)."""
    raise NotImplementedError(
        "Wire to local CardioSYNTAX archive; reuse run_cardiosyntax_balanced_scaleup.py "
        "selection + extractor. CARDIOSYNTAX_PAIR_LIMIT=80 for the published scale."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cardiosyntax-root", type=Path, required=True)
    ap.add_argument("--pair-limit", type=int, default=80)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out / "pairs.jsonl", "w") as f:
        for study_id, artery, angle, fa, fb in iter_balanced_pairs(a.cardiosyntax_root, a.pair_limit):
            f.write(json.dumps({"study_id": study_id, "artery": artery, "angle_label": angle,
                                "video_a_uri": str(fa), "video_b_uri": str(fb)}) + "\n")
            n += 1
    print(f"wrote {n} pairs -> {a.out}/pairs.jsonl")


if __name__ == "__main__":
    main()
