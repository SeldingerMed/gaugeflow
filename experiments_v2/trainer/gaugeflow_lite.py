#!/usr/bin/env python3
"""gaugeflow_lite — compact, faithful GaugeFlow trainer for experiments_v2.

Implements the GaugeFlow core that the external claims need but the probe-only
checks lacked: a shared state encoder, a task head (segmentation OR embedding/retrieval),
gauge-consistency + state-variance losses over REAL paired views (the `real_pairs`
adapter hook), the shuffled-gauge negative control, and a gauge-leakage probe.

It is NOT the full DIAS trainer; it is a self-contained, runnable implementation that
reads the same config schema and emits per-cluster metrics for common/analyze.py.

Dataset contract: a dataset module exposing
    load(config, split) -> list[view]
where view = {"cluster": str, "gauge": hashable, "group": str(optional),
              "angle": float(optional), "image": HxW float32 in [0,1],
              "mask": HxW float32 (segmentation only)}.
The trainer forms gauge pairs (same cluster, different gauge); gauge_shuffle permutes
the partner assignment for the negative control.

Usage:
    python gaugeflow_lite.py --config <cfg.json> --dataset <path/dataset.py> \
        [--override k=v ...] [--smoke]
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_dataset_module(path):
    spec = importlib.util.spec_from_file_location("ds_mod", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---------------- model ----------------
class Encoder(nn.Module):
    def __init__(self, ch=16, zdim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ch, 3, 2, 1), nn.GroupNorm(4, ch), nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.GroupNorm(4, ch * 2), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.GroupNorm(4, ch * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch * 4, zdim)
        self.ch = ch

    def forward(self, x):
        f = self.net(x).flatten(1)
        return self.head(f), f  # state code z, raw feature


class SegDecoder(nn.Module):
    def __init__(self, ch=16, zdim=64, out=64):
        super().__init__()
        self.fc = nn.Linear(zdim, ch * 4 * 8 * 8)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.GELU(),
            nn.Conv2d(ch, 1, 1))
        self.out = out; self.ch = ch

    def forward(self, z):
        x = self.fc(z).view(-1, self.ch * 4, 8, 8)
        x = self.up(x)
        return F.interpolate(x, size=(self.out, self.out), mode="bilinear", align_corners=False)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


def dice_loss(logits, mask, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * mask).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + eps
    return (1 - num / den).mean()


def dice_score(logits, mask, eps=1.0):
    p = (torch.sigmoid(logits) > 0.5).float()
    num = 2 * (p * mask).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + mask.sum((1, 2, 3)) + eps
    return (num / den).detach().cpu().numpy()


# ---------------- probes (pure torch; no sklearn dependency) ----------------
def angle_r2(Z, ang):
    """Ridge closed-form R^2 of angle from embeddings, train/val half-split."""
    Z = torch.tensor(Z, dtype=torch.float32); y = torch.tensor(ang, dtype=torch.float32)
    n = len(Z); idx = torch.randperm(n); tr, va = idx[:n // 2], idx[n // 2:]
    if len(va) < 2: return float("nan")
    Xtr = torch.cat([Z[tr], torch.ones(len(tr), 1)], 1)
    w = torch.linalg.lstsq(Xtr.T @ Xtr + 1e-2 * torch.eye(Xtr.shape[1]), Xtr.T @ y[tr]).solution
    Xva = torch.cat([Z[va], torch.ones(len(va), 1)], 1)
    pred = Xva @ w
    ss_res = ((y[va] - pred) ** 2).sum(); ss_tot = ((y[va] - y[va].mean()) ** 2).sum() + 1e-8
    return float((1 - ss_res / ss_tot).clamp(min=0))


def gauge_separability(Z, labels):
    """Adjusted multinomial-logistic probe accuracy (acc - chance), train/val split."""
    classes = sorted(set(labels)); k = len(classes)
    if k < 2: return 0.0
    y = torch.tensor([classes.index(l) for l in labels])
    Z = torch.tensor(Z, dtype=torch.float32)
    n = len(Z); idx = torch.randperm(n); tr, va = idx[:n // 2], idx[n // 2:]
    if len(va) < 2: return 0.0
    clf = nn.Linear(Z.shape[1], k)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad(); F.cross_entropy(clf(Z[tr]), y[tr]).backward(); opt.step()
    acc = (clf(Z[va]).argmax(1) == y[va]).float().mean().item()
    return max(0.0, acc - 1.0 / k)


def retrieval_top1(Z, study, artery):
    """Leave-one-out NN by cosine; hit if NN shares study+artery."""
    Z = torch.tensor(Z, dtype=torch.float32)
    Zn = F.normalize(Z, dim=1); S = Zn @ Zn.T
    S.fill_diagonal_(-1e9)
    nn_idx = S.argmax(1).numpy()
    hits = [(study[i] == study[j] and artery[i] == artery[j]) for i, j in enumerate(nn_idx)]
    return hits  # per-item bool


def retrieval_top1_transport(Z, ang, study, artery, equiv):
    """Pairwise-transport NN: transport query i into candidate j's gauge frame via the equiv
    head (Δ=ang[j]-ang[i]) before scoring, then cosine NN. The rigorous eval for an equivariant
    representation (vs common-reference canonicalization). hit if NN shares study+artery."""
    Zt = torch.tensor(Z, dtype=torch.float32, device=DEV)
    a = torch.tensor(ang, dtype=torch.float32, device=DEV)
    Zn = F.normalize(Zt, dim=1)
    n = len(Zt)
    hits = []
    with torch.no_grad():
        for i in range(n):
            d = (a - a[i]) * (math.pi / 180.0)                       # transport i -> each j's frame
            q = equiv(torch.cat([Zt[i:i+1].expand(n, -1),
                                 torch.stack([torch.sin(d), torch.cos(d)], 1)], 1))
            s = (F.normalize(q, dim=1) * Zn).sum(1)                  # cos(transport_i->j, z_j) per j
            s[i] = -1e9
            j = int(s.argmax())
            hits.append(study[i] == study[j] and artery[i] == artery[j])
    return hits


# ---------------- training ----------------
def build_pairs(views, shuffle, rng):
    """Pairs of indices: same cluster, different gauge. shuffle => mismatched partners."""
    by_c = {}
    for i, v in enumerate(views):
        by_c.setdefault(v["cluster"], []).append(i)
    pairs = []
    for c, idxs in by_c.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if views[idxs[a]]["gauge"] != views[idxs[b]]["gauge"]:
                    pairs.append((idxs[a], idxs[b]))
    if shuffle and pairs:  # negative control: repartner b's at random across all views
        bs = [p[1] for p in pairs]; rng.shuffle(bs)
        pairs = [(p[0], bs[k]) for k, p in enumerate(pairs)]
    return pairs


def to_batch(views, idxs, key, size):
    arr = np.stack([views[i][key] for i in idxs]).astype("float32")
    t = torch.tensor(arr).unsqueeze(1)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.to(DEV)


def train(cfg, ds, task, smoke):
    seed_all(int(cfg.get("seed", 0)))
    cfg["_smoke"] = smoke
    size = 64 if smoke else int(cfg.get("image_size", 128))
    epochs = int(cfg.get("smoke_epochs", 1)) if smoke else int(cfg.get("epochs", 24))
    zdim = 64
    enc = Encoder(int(cfg.get("base_channels", 16)), zdim).to(DEV)
    params = list(enc.parameters())
    dec = None
    if task == "segmentation":
        dec = SegDecoder(int(cfg.get("base_channels", 16)), zdim, size).to(DEV); params += list(dec.parameters())
    adv = None
    if cfg.get("angle_adv_loss_weight", 0) and task != "segmentation":
        adv = nn.Sequential(nn.Linear(zdim, int(cfg.get("angle_adv_head_dim", 64))), nn.GELU(), nn.Linear(int(cfg.get("angle_adv_head_dim", 64)), 1)).to(DEV)
        params += list(adv.parameters())
    equiv = None
    equiv_ctr = bool(cfg.get("gauge_equiv_contrastive", False))
    if (cfg.get("gauge_equiv_loss_weight", 0) or equiv_ctr) and task != "segmentation":
        h = int(cfg.get("gauge_equiv_head_dim", 64))
        equiv = nn.Sequential(nn.Linear(zdim + 2, h), nn.GELU(), nn.Linear(h, zdim)).to(DEV)
        params += list(equiv.parameters())
    opt = torch.optim.Adam(params, lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-6)))

    train_views = ds.load(cfg, "train")
    rng = random.Random(int(cfg.get("seed", 0)))
    shuffle = bool(cfg.get("gauge_shuffle", False))
    gf_on = bool(cfg.get("gaugeflow_enabled", False))
    w_state = float(cfg.get("gaugeflow_state_consistency_loss_weight", 0)) if gf_on else 0.0
    w_var = float(cfg.get("gaugeflow_state_variance_loss_weight", 0)) if gf_on else 0.0
    w_adv = float(cfg.get("angle_adv_loss_weight", 0)) if gf_on else 0.0
    w_equiv = float(cfg.get("gauge_equiv_loss_weight", 0)) if gf_on else 0.0
    pairs = build_pairs(train_views, shuffle, rng)
    bs = int(cfg.get("batch_size", 16))

    for ep in range(epochs):
        rng.shuffle(pairs)
        plist = pairs[:cfg.get("smoke_max_train_pairs", 24)] if smoke else pairs
        for s in range(0, max(1, len(plist)), bs):
            chunk = plist[s:s + bs]
            if not chunk: continue
            ia = [p[0] for p in chunk]; ib = [p[1] for p in chunk]
            xa, xb = to_batch(train_views, ia, "image", size), to_batch(train_views, ib, "image", size)
            za, fa = enc(xa); zb, fb = enc(xb)
            loss = torch.tensor(0.0, device=DEV)
            if task == "segmentation":
                ma = to_batch(train_views, ia, "mask", size)
                loss = loss + dice_loss(dec(za), ma) + F.binary_cross_entropy_with_logits(dec(za), ma)
            elif equiv_ctr:  # gauge-aligned InfoNCE: transport a->b's frame through the equiv head, then contrast
                aa = torch.tensor([train_views[i]["angle"] for i in ia], dtype=torch.float32, device=DEV)
                ab = torch.tensor([train_views[i]["angle"] for i in ib], dtype=torch.float32, device=DEV)
                def xport(z, d_deg):
                    d = d_deg * (math.pi / 180.0)
                    return equiv(torch.cat([z, torch.stack([torch.sin(d), torch.cos(d)], 1)], 1))
                tgt = torch.arange(len(za), device=DEV)
                a2b, b2a = xport(za, ab - aa), xport(zb, aa - ab)   # symmetric transport
                loss = loss + F.cross_entropy(F.normalize(a2b, 1) @ F.normalize(zb, 1).T / 0.1, tgt)
                loss = loss + F.cross_entropy(F.normalize(b2a, 1) @ F.normalize(za, 1).T / 0.1, tgt)
            else:  # embedding/retrieval: InfoNCE with the paired view as positive
                zan, zbn = F.normalize(za, dim=1), F.normalize(zb, dim=1)
                logits = zan @ zbn.T / 0.1
                loss = loss + F.cross_entropy(logits, torch.arange(len(zan), device=DEV))
            if w_state:                       # gauge-consistency: paired views share state
                loss = loss + w_state * F.mse_loss(za, zb)
            if w_var:                         # discourage unstable per-pair state dispersion
                loss = loss + w_var * (za - zb).var(0).mean()
            if w_adv and adv is not None:     # angle-adversarial via gradient reversal
                ang = torch.tensor([train_views[i]["angle"] for i in ia], dtype=torch.float32, device=DEV).unsqueeze(1)
                pa = adv(GradReverse.apply(za, 1.0))
                loss = loss + w_adv * F.mse_loss(pa, ang)
            if w_equiv and equiv is not None:  # gauge-EQUIVARIANCE: predict z_b from z_a + Δangle (no GRL, keeps info)
                aa = torch.tensor([train_views[i]["angle"] for i in ia], dtype=torch.float32, device=DEV)
                ab = torch.tensor([train_views[i]["angle"] for i in ib], dtype=torch.float32, device=DEV)
                d = (ab - aa) * (math.pi / 180.0)             # SO(2) gauge: projection angle is a real angle
                feat = torch.cat([za, torch.stack([torch.sin(d), torch.cos(d)], 1)], 1)
                loss = loss + w_equiv * F.mse_loss(equiv(feat), zb.detach())  # sg target: InfoNCE negs prevent collapse
            opt.zero_grad(); loss.backward(); opt.step()

    # -------- eval --------
    enc.eval()
    ev = ds.load(cfg, "test")
    if smoke: ev = ev[:cfg.get("smoke_max_eval_pairs", 12) * 2] or ev
    with torch.no_grad():
        Z, Zc = [], []
        for s in range(0, len(ev), bs):
            ii = list(range(s, min(s + bs, len(ev))))
            z = enc(to_batch(ev, ii, "image", size))[0]
            Z.append(z.cpu().numpy())
            if equiv_ctr:  # canonicalize: transport every view to reference angle 0, then retrieve
                d = torch.tensor([-float(ev[i]["angle"]) for i in ii], dtype=torch.float32, device=DEV) * (math.pi / 180.0)
                zc = equiv(torch.cat([z, torch.stack([torch.sin(d), torch.cos(d)], 1)], 1))
                Zc.append(zc.cpu().numpy())
        Z = np.concatenate(Z) if Z else np.zeros((0, zdim))
        Zc = np.concatenate(Zc) if Zc else Z   # canonical embeddings for retrieval (raw Z kept for leakage)

    rows = []
    if task == "segmentation":
        with torch.no_grad():
            for s in range(0, len(ev), bs):
                ii = list(range(s, min(s + bs, len(ev))))
                d = dice_score(dec(enc(to_batch(ev, ii, "image", size))[0]), to_batch(ev, ii, "mask", size))
                for k, i in enumerate(ii):
                    rows.append({"case_id": ev[i]["cluster"], "dice": float(d[k])})
        sep = gauge_separability(Z, [v["gauge"] for v in ev])
        # collapse to per-case mean dice; attach separability to each
        by = {}
        for r in rows: by.setdefault(r["case_id"], []).append(r["dice"])
        return [{"case_id": c, "dice": float(np.mean(v)), "seq_separability": sep} for c, v in by.items()]
    else:
        study = [v["cluster"] for v in ev]; artery = [v.get("group", "") for v in ev]
        ang = [float(v.get("angle", 0.0)) for v in ev]
        if equiv_ctr and cfg.get("pairwise_transport_retrieval", False):
            hits = retrieval_top1_transport(Z, ang, study, artery, equiv)  # rigorous equivariant eval
        else:
            hits = retrieval_top1(Zc, study, artery)   # retrieval on canonicalized embeddings
        r2 = angle_r2(Z, ang)                       # leakage on raw embeddings (info-kept claim)
        by = {}
        for i, st in enumerate(study): by.setdefault(st, []).append(hits[i])
        return [{"study_id": st, "retrieval_top1": float(np.mean(h)), "angle_r2": r2} for st, h in by.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--override", nargs="*", default=[]); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for kv in a.override:
        k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except Exception: pass
        cfg[k] = v
    ds = load_dataset_module(a.dataset)
    task = "segmentation" if cfg.get("task") == "segmentation" else "embedding"
    out_rows = train(cfg, ds, task, a.smoke)
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    name = "per_case_metrics.jsonl" if task == "segmentation" else ("per_study_metrics.jsonl" if cfg.get("task") == "retrieval" else "per_patient_metrics.jsonl")
    with open(out_dir / name, "w") as f:
        for r in out_rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} rows -> {out_dir/name}")


if __name__ == "__main__":
    main()
