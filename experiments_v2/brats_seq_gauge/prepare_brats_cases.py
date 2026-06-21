#!/usr/bin/env python3
"""Build the trainer inputs for the BraTS sequence-gauge experiment — REAL DATA.

Streams MSD Task01/BraTS from the public S3 tarball via HTTP Range requests (no
full 7 GB download), extracts one matched axial slice per case for all four pulse
sequences + the seg mask, and writes the two jsonl files gaugeflow_lite consumes:

  data/cases.jsonl       -- one row per (case, sequence) view, DIAS-compatible schema
  data/gauge_pairs.jsonl -- same-case sequence pairs (the REAL gauge)

Reuses the warning-clean tar-scan/fetch path from the DeepScientist quest-012
streaming loader (stream_msd_task01_schema.py): scan_tar_headers / case_id /
fetch_member. Raw NIfTI payloads go to a temp dir and are deleted right after
slice extraction (same raw-data policy as the original run).

Slice choice: the axial z with the largest tumour area (max label>0 pixels), so the
segmentation target is non-trivial and identical across the four co-registered
sequences. This is the real gauge: same anatomy/state, different observation channel.
"""
from __future__ import annotations
import argparse, importlib.util, json, tempfile
from pathlib import Path
import numpy as np

SEQUENCES = ["FLAIR", "T1w", "t1gd", "T2w"]  # MSD Task01 channel order

# DeepScientist quest-012 streaming loader (HTTP Range tar reader).
SCHEMA_PATH = Path(
    "${DATA_ROOT:-/path/to/data}"
    "experiments/main/run-coin-mpmri-msd-task01-stream-schema-v2/stream_msd_task01_schema.py"
)


def load_schema():
    spec = importlib.util.spec_from_file_location("msd_task01_schema", SCHEMA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load schema module from {SCHEMA_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # S3 resets connections during the long sequential header scan; wrap the
    # module-global http_range (referenced at call time by scan_tar_headers /
    # fetch_member) with bounded retries + linear backoff.
    raw = mod.http_range

    def retrying(url, start, end, timeout=120, _tries=6):
        import time
        for attempt in range(_tries):
            try:
                return raw(url, start, end, timeout=timeout)
            except Exception:
                if attempt == _tries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
    mod.http_range = retrying
    return mod


def select_pairs(schema, case_count: int, max_image_bytes: int):
    """Deterministic smallest-eligible matched image/label pairs (same rule as the
    published scaleup run)."""
    url = schema.DEFAULT_URL
    size = schema.object_size(url)
    members, _ = schema.scan_tar_headers(url, size, 5000)
    files = [m for m in members if m["typeflag"] in {"0", "\0", ""}]
    images = {schema.case_id(m["name"], "imagesTr"): m for m in files if schema.case_id(m["name"], "imagesTr")}
    labels = {schema.case_id(m["name"], "labelsTr"): m for m in files if schema.case_id(m["name"], "labelsTr")}
    pairs = []
    for case in sorted(set(images).intersection(labels)):
        img = images[case]
        if int(img["size"]) <= max_image_bytes:
            pairs.append({"case_id": case, "image": img, "label": labels[case],
                          "total": int(img["size"]) + int(labels[case]["size"])})
    pairs.sort(key=lambda r: (r["total"], r["case_id"]))
    return pairs[:case_count], url


def best_axial_z(label_arr: np.ndarray) -> int:
    """Axial slice (last spatial axis) with the most tumour voxels; fall back to mid."""
    if label_arr.ndim != 3:
        return label_arr.shape[2] // 2
    per_z = (label_arr > 0).reshape(-1, label_arr.shape[2]).sum(axis=0)
    return int(per_z.argmax()) if per_z.max() > 0 else label_arr.shape[2] // 2


def slice_to_png(slice2d: np.ndarray):
    """1-99 percentile normalize a 2D float slice to an 8-bit grayscale PNG image."""
    from PIL import Image
    a = np.asarray(slice2d, dtype=np.float32)
    nz = a[np.isfinite(a) & (a != 0)]
    if nz.size:
        lo, hi = np.percentile(nz, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    else:
        a = np.zeros_like(a)
    return Image.fromarray((a * 255).astype("uint8"))


def mask_to_png(mask2d: np.ndarray):
    """Binary seg mask -> 0/255 PNG. Must NOT percentile-normalize: a constant-valued
    mask would collapse to all-black and make Dice degenerate (eps/eps = 1.0)."""
    from PIL import Image
    return Image.fromarray(((np.asarray(mask2d) > 0).astype("uint8") * 255))


def split_for(i: int) -> str:
    r = i % 20  # deterministic 70/15/15 case-level split
    return "train" if r < 14 else ("val" if r < 17 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-count", type=int, default=48)
    ap.add_argument("--max-image-bytes", type=int, default=160 * 1024 * 1024)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    a = ap.parse_args()

    import nibabel as nib
    schema = load_schema()
    frames = a.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    pairs, url = select_pairs(schema, a.case_count, a.max_image_bytes)
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(pairs)} eligible cases — need >=8 for case-level stats")

    cases_f = open(a.out / "cases.jsonl", "w")
    pairs_f = open(a.out / "gauge_pairs.jsonl", "w")
    n_written = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, pair in enumerate(pairs):
            case = pair["case_id"]
            img_bytes = schema.fetch_member(url, pair["image"], a.max_image_bytes)
            lab_bytes = schema.fetch_member(url, pair["label"], 50 * 1024 * 1024)
            ip, lp = td / f"{case}_img.nii.gz", td / f"{case}_lab.nii.gz"
            ip.write_bytes(img_bytes); lp.write_bytes(lab_bytes)
            try:
                image = np.asanyarray(nib.load(str(ip), mmap=False).dataobj).astype("float32")
                label = np.asanyarray(nib.load(str(lp), mmap=False).dataobj)
                if image.ndim != 4 or image.shape[-1] != 4:
                    raise ValueError(f"{case}: expected 4D x4ch, got {image.shape}")
                z = best_axial_z(label)
                split = split_for(i)
                mask_rel = f"data/frames/{case}_seg_z{z}.png"
                mask_to_png(label[:, :, z]).save(frames / f"{case}_seg_z{z}.png")
                seq_paths = {}
                for ci, seq in enumerate(SEQUENCES):
                    rel = f"data/frames/{case}_{seq}_z{z}.png"
                    slice_to_png(image[:, :, z, ci]).save(frames / f"{case}_{seq}_z{z}.png")
                    seq_paths[seq] = rel
                    cases_f.write(json.dumps({
                        "case_id": case, "dataset": "brats", "data_kind": "image_volume_slice",
                        "modality": "MRI", "gauge_level": seq, "split": split, "cluster_id": case,
                        "image_uri": rel, "mask_uri": mask_rel,
                    }) + "\n")
                for x in range(len(SEQUENCES)):
                    for y in range(x + 1, len(SEQUENCES)):
                        pairs_f.write(json.dumps({
                            "case_id": case, "split": split,
                            "view_a": SEQUENCES[x], "view_b": SEQUENCES[y],
                            "image_a_uri": seq_paths[SEQUENCES[x]], "image_b_uri": seq_paths[SEQUENCES[y]],
                            "mask_uri": mask_rel,
                        }) + "\n")
                n_written += 1
                print(f"[{i+1}/{len(pairs)}] {case} z={z} split={split}")
            finally:
                ip.unlink(missing_ok=True); lp.unlink(missing_ok=True)
    cases_f.close(); pairs_f.close()
    print(f"wrote {n_written} real BraTS cases -> {a.out}/cases.jsonl, gauge_pairs.jsonl "
          f"(raw NIfTI payloads deleted)")


if __name__ == "__main__":
    main()
