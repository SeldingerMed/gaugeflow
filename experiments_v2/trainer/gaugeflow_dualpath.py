#!/usr/bin/env python3
"""gaugeflow_dualpath — full-machinery GaugeFlow for STATIC discrete gauges.

Ports the transferable part of the canonical DIAS trainer
(`coin_obstopo/run_cfpath_train.py`) that the lite trainer lacked: a dual content/
style disentanglement. The lite trainer used ONE embedding, so its adversarial GRL
destroyed the content needed for the task (CardioSYNTAX retrieval -0.044, p~0). Here:

  encoder -> pooled
    z_topo  = topo_proj(pooled) [+ bottleneck]   # gauge-INVARIANT content; task uses this
    z_style = style_proj(pooled)                  # ABSORBS the gauge

Losses (faithful to run_cfpath_train.py's style machinery):
  task            : InfoNCE(z_topo_a, z_topo_b)  or  Dice+BCE(dec(z_topo), mask)
  gauge-consistency: w_state * MSE(z_topo_a, z_topo_b)        # paired views share content
  style-absorb    : w_absorb * CE(style_absorb_head(z_style), gauge)  # style path takes the gauge
  topo-invariance : w_grl * CE(style_clf(grad_reverse(z_topo)), gauge) # content made gauge-free
  style-prototype : w_proto * CE(prototype_logits(z_style), gauge)     # EMA style bank

DROPPED BY DESIGN (not omission): the commutator / operator / next-frame / transport
terms are world-model DYNAMICS terms. CardioSYNTAX retrieval and BraTS segmentation are
STATIC tasks with no temporal transition, so a transition operator to commute with does
not exist. Including them would be undefined, not merely heavy.

Same dataset contract, probes, gates, and per-cluster output as gaugeflow_lite.py — so
common/analyze.py + gates.json transfer unchanged. gauge_shuffle = negative control.
"""
from __future__ import annotations
import argparse, importlib.util, json, random
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
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


class DualPathNet(nn.Module):
    """Shared encoder + content(z_topo)/style(z_style) projections + style-prototype bank."""
    def __init__(self, ch=16, zdim=64, n_gauge=4, bottleneck=0, n_proto=8, temp=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, ch, 3, 2, 1), nn.GroupNorm(4, ch), nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.GroupNorm(4, ch * 2), nn.GELU(),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.GroupNorm(4, ch * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        feat = ch * 4
        self.topo_proj = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(True), nn.Linear(feat, zdim), nn.Tanh())
        if 0 < bottleneck < zdim:
            self.bottleneck = nn.Sequential(nn.Linear(zdim, bottleneck), nn.Tanh(), nn.Linear(bottleneck, zdim), nn.Tanh())
        else:
            self.bottleneck = nn.Identity()
        self.style_proj = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(True), nn.Linear(feat, zdim), nn.Tanh())
        self.style_absorb = nn.Sequential(nn.Linear(zdim, zdim), nn.ReLU(True), nn.Linear(zdim, n_gauge))
        self.style_clf = nn.Sequential(nn.Linear(zdim, zdim), nn.ReLU(True), nn.Linear(zdim, n_gauge))  # on GRL(z_topo)
        self.temp = temp
        self.register_buffer("prototypes", F.normalize(torch.randn(max(n_proto, n_gauge), zdim), dim=1))

    def encode(self, x):
        pooled = self.enc(x).flatten(1)
        z_topo = self.bottleneck(self.topo_proj(pooled))
        z_style = self.style_proj(pooled)
        return z_topo, z_style

    def proto_logits(self, z):
        return F.normalize(z, dim=1) @ F.normalize(self.prototypes, dim=1).t() / max(self.temp, 1e-6)

    @torch.no_grad()
    def update_prototypes(self, z_style, labels, momentum):
        zn = F.normalize(z_style, dim=1)
        for lab in labels.unique():
            m = labels == lab
            if not bool(m.any()):
                continue
            i = int(lab.item())
            if i >= self.prototypes.shape[0]:
                continue
            mean = F.normalize(zn[m].mean(0), dim=0)
            self.prototypes[i].mul_(momentum).add_((1 - momentum) * mean)
            self.prototypes[i].copy_(F.normalize(self.prototypes[i], dim=0))


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
        return F.interpolate(self.up(x), size=(self.out, self.out), mode="bilinear", align_corners=False)


# ---------------- losses / probes (shared with lite) ----------------
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


def angle_r2(Z, ang):
    Z = torch.tensor(Z, dtype=torch.float32); y = torch.tensor(ang, dtype=torch.float32)
    n = len(Z); idx = torch.randperm(n); tr, va = idx[:n // 2], idx[n // 2:]
    if len(va) < 2: return float("nan")
    Xtr = torch.cat([Z[tr], torch.ones(len(tr), 1)], 1)
    w = torch.linalg.lstsq(Xtr.T @ Xtr + 1e-2 * torch.eye(Xtr.shape[1]), Xtr.T @ y[tr]).solution
    Xva = torch.cat([Z[va], torch.ones(len(va), 1)], 1); pred = Xva @ w
    ss_res = ((y[va] - pred) ** 2).sum(); ss_tot = ((y[va] - y[va].mean()) ** 2).sum() + 1e-8
    return float((1 - ss_res / ss_tot).clamp(min=0))


def gauge_separability(Z, labels):
    classes = sorted(set(labels)); k = len(classes)
    if k < 2: return 0.0
    y = torch.tensor([classes.index(l) for l in labels]); Z = torch.tensor(Z, dtype=torch.float32)
    n = len(Z); idx = torch.randperm(n); tr, va = idx[:n // 2], idx[n // 2:]
    if len(va) < 2: return 0.0
    clf = nn.Linear(Z.shape[1], k); opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    for _ in range(200):
        opt.zero_grad(); F.cross_entropy(clf(Z[tr]), y[tr]).backward(); opt.step()
    acc = (clf(Z[va]).argmax(1) == y[va]).float().mean().item()
    return max(0.0, acc - 1.0 / k)


def retrieval_top1(Z, study, artery):
    Z = torch.tensor(Z, dtype=torch.float32); Zn = F.normalize(Z, dim=1); S = Zn @ Zn.T
    S.fill_diagonal_(-1e9); nn_idx = S.argmax(1).numpy()
    return [(study[i] == study[j] and artery[i] == artery[j]) for i, j in enumerate(nn_idx)]


# ---------------- data ----------------
def build_pairs(views, shuffle, rng):
    by_c = {}
    for i, v in enumerate(views):
        by_c.setdefault(v["cluster"], []).append(i)
    pairs = []
    for c, idxs in by_c.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if views[idxs[a]]["gauge"] != views[idxs[b]]["gauge"]:
                    pairs.append((idxs[a], idxs[b]))
    if shuffle and pairs:
        bs = [p[1] for p in pairs]; rng.shuffle(bs)
        pairs = [(p[0], bs[k]) for k, p in enumerate(pairs)]
    return pairs


def to_batch(views, idxs, key, size):
    arr = np.stack([views[i][key] for i in idxs]).astype("float32")
    t = torch.tensor(arr).unsqueeze(1)
    if t.shape[-1] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.to(DEV)


def gauge_vocab(views):
    return {g: i for i, g in enumerate(sorted({v["gauge"] for v in views}, key=str))}


def gauge_ids(views, idxs, vocab):
    return torch.tensor([vocab.get(views[i]["gauge"], 0) for i in idxs], device=DEV)


# ---------------- training ----------------
def train(cfg, ds, task, smoke):
    seed_all(int(cfg.get("seed", 0)))
    cfg["_smoke"] = smoke
    size = 64 if smoke else int(cfg.get("image_size", 128))
    epochs = int(cfg.get("smoke_epochs", 1)) if smoke else int(cfg.get("epochs", 24))
    zdim = 64
    train_views = ds.load(cfg, "train")
    vocab = gauge_vocab(train_views); n_gauge = max(2, len(vocab))

    net = DualPathNet(int(cfg.get("base_channels", 16)), zdim, n_gauge,
                      int(cfg.get("topo_bottleneck_dim", 0)),
                      int(cfg.get("num_style_prototypes", 8)),
                      float(cfg.get("style_prototype_temperature", 0.2))).to(DEV)
    params = list(net.parameters())
    dec = None
    if task == "segmentation":
        dec = SegDecoder(int(cfg.get("base_channels", 16)), zdim, size).to(DEV); params += list(dec.parameters())
    opt = torch.optim.Adam(params, lr=float(cfg.get("learning_rate", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-6)))

    rng = random.Random(int(cfg.get("seed", 0)))
    shuffle = bool(cfg.get("gauge_shuffle", False))
    gf = bool(cfg.get("gaugeflow_enabled", False))
    w_state = float(cfg.get("gaugeflow_state_consistency_loss_weight", 0.0)) if gf else 0.0
    w_absorb = float(cfg.get("style_absorb_loss_weight", 0.5)) if gf else 0.0
    w_grl = float(cfg.get("topo_style_grl_weight", 0.1)) if gf else 0.0
    w_proto = float(cfg.get("style_prototype_loss_weight", 0.1)) if gf else 0.0
    grl_lambda = float(cfg.get("grl_lambda", 1.0))
    proto_mom = float(cfg.get("style_prototype_momentum", 0.9))
    pairs = build_pairs(train_views, shuffle, rng)
    bs = int(cfg.get("batch_size", 16))

    for ep in range(epochs):
        rng.shuffle(pairs)
        plist = pairs[:cfg.get("smoke_max_train_pairs", 24)] if smoke else pairs
        for s in range(0, max(1, len(plist)), bs):
            chunk = plist[s:s + bs]
            if not chunk: continue
            ia = [p[0] for p in chunk]; ib = [p[1] for p in chunk]
            xa = to_batch(train_views, ia, "image", size); xb = to_batch(train_views, ib, "image", size)
            za_t, za_s = net.encode(xa); zb_t, zb_s = net.encode(xb)
            ga = gauge_ids(train_views, ia, vocab); gb = gauge_ids(train_views, ib, vocab)
            loss = torch.tensor(0.0, device=DEV)
            # ---- task (content path z_topo) ----
            if task == "segmentation":
                ma = to_batch(train_views, ia, "mask", size)
                loss = loss + dice_loss(dec(za_t), ma) + F.binary_cross_entropy_with_logits(dec(za_t), ma)
            else:
                zan, zbn = F.normalize(za_t, dim=1), F.normalize(zb_t, dim=1)
                logits = zan @ zbn.T / 0.1
                loss = loss + F.cross_entropy(logits, torch.arange(len(zan), device=DEV))
            # ---- gauge-consistency on content ----
            if w_state:
                loss = loss + w_state * F.mse_loss(za_t, zb_t)
            # ---- style path absorbs the gauge ----
            if w_absorb:
                loss = loss + w_absorb * 0.5 * (F.cross_entropy(net.style_absorb(za_s), ga) +
                                                F.cross_entropy(net.style_absorb(zb_s), gb))
            # ---- content made gauge-invariant via GRL ----
            if w_grl:
                loss = loss + w_grl * 0.5 * (F.cross_entropy(net.style_clf(GradReverse.apply(za_t, grl_lambda)), ga) +
                                             F.cross_entropy(net.style_clf(GradReverse.apply(zb_t, grl_lambda)), gb))
            # ---- style-prototype bank (EMA) ----
            if w_proto:
                loss = loss + w_proto * 0.5 * (F.cross_entropy(net.proto_logits(za_s), ga) +
                                               F.cross_entropy(net.proto_logits(zb_s), gb))
                net.update_prototypes(torch.cat([za_s, zb_s]).detach(), torch.cat([ga, gb]), proto_mom)
            opt.zero_grad(); loss.backward(); opt.step()

    # -------- eval (probe the CONTENT path z_topo) --------
    net.eval()
    ev = ds.load(cfg, "test")
    if smoke: ev = ev[:cfg.get("smoke_max_eval_pairs", 12) * 2] or ev
    with torch.no_grad():
        Z = []
        for s in range(0, len(ev), bs):
            xb = to_batch(ev, list(range(s, min(s + bs, len(ev)))), "image", size)
            Z.append(net.encode(xb)[0].cpu().numpy())
        Z = np.concatenate(Z) if Z else np.zeros((0, zdim))

    if task == "segmentation":
        rows = []
        with torch.no_grad():
            for s in range(0, len(ev), bs):
                ii = list(range(s, min(s + bs, len(ev))))
                d = dice_score(dec(net.encode(to_batch(ev, ii, "image", size))[0]), to_batch(ev, ii, "mask", size))
                for k, i in enumerate(ii):
                    rows.append({"case_id": ev[i]["cluster"], "dice": float(d[k])})
        sep = gauge_separability(Z, [v["gauge"] for v in ev])
        by = {}
        for r in rows: by.setdefault(r["case_id"], []).append(r["dice"])
        return [{"case_id": c, "dice": float(np.mean(v)), "seq_separability": sep} for c, v in by.items()]
    else:
        study = [v["cluster"] for v in ev]; artery = [v.get("group", "") for v in ev]
        ang = [float(v.get("angle", 0.0)) for v in ev]
        hits = retrieval_top1(Z, study, artery); r2 = angle_r2(Z, ang)
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
    rows = train(cfg, ds, task, a.smoke)
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    name = "per_case_metrics.jsonl" if task == "segmentation" else ("per_study_metrics.jsonl" if cfg.get("task") == "retrieval" else "per_patient_metrics.jsonl")
    with open(out_dir / name, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out_dir/name}")


if __name__ == "__main__":
    main()
