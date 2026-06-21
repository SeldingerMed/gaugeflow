#!/usr/bin/env python3
"""Synthetic BraTS-shaped smoke fixture: proves the seg+sequence-gauge pipeline runs
end-to-end while the real MSD Task01/BraTS download is pending (raw was purged).

Each case has a shared underlying tumor "shape" (the anatomy/state) rendered under 4
sequence "gauges" with different intensity/contrast styles. A good gauge-invariant
model recovers the shared shape (Dice) while the sequence becomes unpredictable from
the state code (separability -> 0). Writes PNG slices + masks + cases.jsonl.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

SEQS = ["FLAIR", "T1w", "t1gd", "T2w"]


def disk(size, cx, cy, r):
    yy, xx = np.mgrid[0:size, 0:size]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    img_dir = a.out / "frames"; img_dir.mkdir(parents=True, exist_ok=True)
    cases = open(a.out / "cases.jsonl", "w")
    pairs = open(a.out / "gauge_pairs.jsonl", "w")
    for c in range(a.n_cases):
        split = "train" if c % 20 < 14 else ("val" if c % 20 < 17 else "test")
        cx, cy = rng.integers(20, a.size - 20, 2); r = rng.integers(6, 14)
        shape = disk(a.size, cx, cy, r)                      # shared anatomy/state
        base = 0.2 + 0.1 * disk(a.size, a.size // 2, a.size // 2, a.size // 3)  # background
        paths = {}
        for si, seq in enumerate(SEQS):
            # sequence-specific intensity style (the gauge) applied to the SAME shape
            style_fg = 0.4 + 0.15 * si
            style_bg = 0.15 + 0.05 * ((si + 1) % 4)
            img = base * style_bg + shape * style_fg + rng.normal(0, 0.02, shape.shape)
            img = np.clip(img, 0, 1)
            ip = img_dir / f"case{c:03d}_{seq}.png"
            Image.fromarray((img * 255).astype("uint8")).save(ip)
            paths[seq] = ip
            cases.write(json.dumps({
                "case_id": f"case{c:03d}", "cluster_id": f"case{c:03d}", "dataset": "brats_smoke",
                "modality": "MRI", "gauge_level": seq, "split": split,
                "image_uri": str(ip), "mask_uri": str(img_dir / f"case{c:03d}_seg.png"),
            }) + "\n")
        Image.fromarray((shape * 255).astype("uint8")).save(img_dir / f"case{c:03d}_seg.png")
        for x in range(len(SEQS)):
            for y in range(x + 1, len(SEQS)):
                pairs.write(json.dumps({"case_id": f"case{c:03d}", "split": split,
                                        "view_a": SEQS[x], "view_b": SEQS[y]}) + "\n")
    cases.close(); pairs.close()
    print(f"wrote {a.n_cases} synthetic cases -> {a.out}/")


if __name__ == "__main__":
    main()
