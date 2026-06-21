"""CardioSYNTAX dataset adapter for gaugeflow_lite (real local data).

Reads the local manifest (tmp/sources/cardiosyntax/part*.json) and frame arrays from
the local shard zips (tmp/cardiosyntax/<shard>.zip). The observation gauge is the
projection view angle (PositionerPrimary/SecondaryAngle); group=artery; cluster=study.

Only shards present on disk are used (the full set is 0..9.zip; this machine has a
subset). Videos whose path points at a missing shard are skipped.

Config keys:
  cardiosyntax_manifest_glob : default tmp/sources/cardiosyntax/part*.json
  cardiosyntax_zip_dir       : default tmp/cardiosyntax
  max_studies                : cap studies (smoke); 0 = all available
  image_size                 : resize middle frame to this (default 128)
"""
from __future__ import annotations
import glob, io, json, os, zipfile
from pathlib import Path
import numpy as np

Q = os.environ.get("CARDIOSYNTAX_ROOT", "/path/to/cardiosyntax")
DEFAULT_MANIFEST = f"{Q}/tmp/sources/cardiosyntax/part*.json"
DEFAULT_ZIPDIR = f"{Q}/tmp/cardiosyntax"
_ZIP_CACHE: dict[str, zipfile.ZipFile] = {}


def _zip(zipdir, shard):
    if shard not in _ZIP_CACHE:
        p = Path(zipdir) / f"{shard}.zip"
        _ZIP_CACHE[shard] = zipfile.ZipFile(p) if p.exists() else None
    return _ZIP_CACHE[shard]


def _available_names(zf):
    return set(zf.namelist()) if zf else set()


def _resize(a, size):
    import torch, torch.nn.functional as F
    t = torch.tensor(a, dtype=torch.float32)[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def _split_of(study_uid, split):
    h = abs(hash(("cardio", study_uid))) % 10
    want = "train" if h < 7 else "test"   # 70/30 study-level
    return want == split


def load(config, split):
    size = 64 if config.get("_smoke") else int(config.get("image_size", 128))
    zipdir = config.get("cardiosyntax_zip_dir", DEFAULT_ZIPDIR)
    manifests = sorted(glob.glob(config.get("cardiosyntax_manifest_glob", DEFAULT_MANIFEST)))
    max_studies = int(config.get("max_studies", 0))

    # which shards exist locally
    shard_ids = [p.stem for p in Path(zipdir).glob("*.zip")]
    zips = {s: _zip(zipdir, s) for s in shard_ids}
    names = {s: _available_names(z) for s, z in zips.items()}

    views, n_studies = [], 0
    for mf in manifests:
        for study in json.load(open(mf)):
            uid = study["study_uid"]
            if not _split_of(uid, split):
                continue
            kept = []
            for v in study.get("videos", []):
                path = v["path"]; shard = path.split("/", 1)[0]
                zf = zips.get(shard)
                if zf is None:
                    continue
                # manifest path is a prefix; find the matching .npy entry in the shard
                cand = [n for n in names[shard] if n.startswith(path) and n.endswith(".npy")]
                if not cand:
                    continue
                kept.append((cand[0], shard, v))
            if len(kept) < 2:   # need >=2 angles in the same study for a gauge pair
                continue
            for entry, shard, v in kept:
                try:
                    arr = np.load(io.BytesIO(zips[shard].read(entry)))
                except Exception:
                    continue
                frame = arr[arr.shape[0] // 2] if arr.ndim == 3 else arr  # middle frame
                frame = frame.astype("float32")
                frame = (frame - frame.min()) / (np.ptp(frame) + 1e-6)
                prim = float(v.get("PositionerPrimaryAngle", 0) or 0)
                sec = float(v.get("PositionerSecondaryAngle", 0) or 0)
                views.append({
                    "cluster": uid, "group": v.get("artery", "NA"),
                    "gauge": f"{round(prim/10)}_{round(sec/10)}",  # discretised angle bucket
                    "angle": prim,
                    "image": _resize(frame, size),
                })
            n_studies += 1
            if max_studies and n_studies >= max_studies:
                return views
    return views
