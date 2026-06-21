"""BraTS sequence-gauge dataset adapter for gaugeflow_lite.

Reads cases.jsonl (DIAS-compatible schema from prepare_brats_cases.py): one row per
(case, sequence) view with image_uri + mask_uri. gauge = pulse sequence; cluster = case.
Works for the real MSD Task01/BraTS slices OR the synthetic smoke fixture.

Config keys:
  cases_jsonl : path to cases.jsonl
  image_size  : resize (default 128; 64 under smoke)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _load_png(path, size):
    from PIL import Image
    import torch, torch.nn.functional as F
    a = np.asarray(Image.open(path).convert("L"), dtype="float32") / 255.0
    t = torch.tensor(a)[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def load(config, split):
    size = 64 if config.get("_smoke") else int(config.get("image_size", 128))
    rows = [json.loads(l) for l in open(config["cases_jsonl"]) if l.strip()]
    out = []
    for r in rows:
        if r.get("split") != split:
            continue
        img = _load_png(r["image_uri"], size)
        mask = _load_png(r["mask_uri"], size)
        out.append({"cluster": r["cluster_id"], "gauge": r["gauge_level"],
                    "image": img, "mask": (mask > 0.5).astype("float32")})
    return out
