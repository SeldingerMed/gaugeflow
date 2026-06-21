#!/usr/bin/env python3
"""Shared rigor harness for experiments_v2 — modality-agnostic.

Turns per-arm evaluation rows into the statistics the external claim needs and that
the current ACDC/BraTS/CardioSYNTAX probes lack: cluster-bootstrap CIs, a permutation
null on the GaugeFlow-vs-baseline effect, the negative-control delta (shuffled gauge
must NOT help), and a pre-registered gate check.

Input: a long-format results.jsonl, one row per evaluation unit per arm per seed:
    {"arm": "gaugeflow|baseline|negctrl", "seed": 0, "cluster": "<seq/clip/case id>",
     "metric": 0.83, "leakage": 0.21}
- "metric" is the task metric (Dice, retrieval@1, or -MAE; see --direction).
- "leakage" is optional (gauge separability / leakage R²); omit if not applicable.

Gate spec (JSON via --gates), pre-registered before running, e.g.:
    {"metric_direction": "higher",            # higher|lower is better
     "min_delta_vs_baseline": 0.0,            # gaugeflow must beat baseline by >= this
     "require_delta_ci_excludes_0": true,
     "max_leakage": 0.035,                     # gaugeflow leakage must be <= this (optional)
     "max_negctrl_abs_delta": 0.01}            # shuffled-gauge effect must be ~0 (optional)

Usage:
    python analyze.py --results results.jsonl --gates gates.json --direction higher
    python analyze.py --demo        # synthetic self-check
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np


def _load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def _cluster_means(rows, arm):
    """Mean metric per cluster for one arm (pooled across seeds)."""
    by = {}
    for r in rows:
        if r["arm"] != arm:
            continue
        by.setdefault(r["cluster"], []).append(float(r["metric"]))
    return {c: float(np.mean(v)) for c, v in by.items()}


def _boot_ci(values, B, rng, agg=np.mean):
    """Cluster bootstrap: resample the list of per-cluster values with replacement."""
    vals = np.asarray(values, float)
    if len(vals) < 2:
        return float(agg(vals)), (float("nan"), float("nan"))
    stats = [agg(rng.choice(vals, len(vals), replace=True)) for _ in range(B)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(agg(vals)), (float(lo), float(hi))


def _perm_null(g, b, B, rng):
    """Permutation null on the paired (per-cluster) gaugeflow-baseline delta.
    Clusters present in both arms are paired; sign-flip each pair's delta."""
    common = sorted(set(g) & set(b))
    if len(common) < 2:
        return float("nan"), float("nan")
    d = np.array([g[c] - b[c] for c in common])
    obs = d.mean()
    null = []
    for _ in range(B):
        flip = rng.choice([-1, 1], len(d))
        null.append((d * flip).mean())
    null = np.array(null)
    # two-sided p
    p = (np.abs(null) >= abs(obs)).mean()
    return float(obs), float(p)


def analyze(rows, gates, direction="higher", B=5000, seed=0):
    rng = np.random.default_rng(seed)
    sign = 1.0 if direction == "higher" else -1.0
    out = {"direction": direction, "arms": {}}

    for arm in ("baseline", "gaugeflow", "negctrl"):
        cm = _cluster_means(rows, arm)
        if not cm:
            continue
        mean, ci = _boot_ci(list(cm.values()), B, rng)
        out["arms"][arm] = {"n_clusters": len(cm), "mean": mean, "ci95": ci}

    g, b = _cluster_means(rows, "gaugeflow"), _cluster_means(rows, "baseline")
    common = sorted(set(g) & set(b))
    deltas = [(g[c] - b[c]) * sign for c in common]
    dmean, dci = _boot_ci(deltas, B, rng)
    _, pval = _perm_null(g, b, B, rng)
    out["gaugeflow_minus_baseline"] = {"signed_delta": dmean, "ci95": dci, "perm_p": pval,
                                       "n_paired_clusters": len(common)}

    # negative control: shuffled-gauge arm should not beat baseline
    nc = _cluster_means(rows, "negctrl")
    if nc:
        cc = sorted(set(nc) & set(b))
        ncd = float(np.mean([(nc[c] - b[c]) * sign for c in cc])) if cc else float("nan")
        out["negctrl_minus_baseline"] = {"signed_delta": ncd, "n_paired_clusters": len(cc)}

    # leakage (optional): mean over gaugeflow rows that carry it
    leak = [float(r["leakage"]) for r in rows if r["arm"] == "gaugeflow" and "leakage" in r]
    if leak:
        out["gaugeflow_leakage_mean"] = float(np.mean(leak))

    # ---- gate check ----
    checks = {}
    g_ = gates
    if "min_delta_vs_baseline" in g_:
        checks["delta_meets_min"] = dmean >= g_["min_delta_vs_baseline"]
    if g_.get("require_delta_ci_excludes_0"):
        checks["delta_ci_excludes_0"] = dci[0] > 0
    if "max_leakage" in g_ and "gaugeflow_leakage_mean" in out:
        checks["leakage_under_cap"] = out["gaugeflow_leakage_mean"] <= g_["max_leakage"]
    if "max_negctrl_abs_delta" in g_ and "negctrl_minus_baseline" in out:
        checks["negctrl_is_null"] = abs(out["negctrl_minus_baseline"]["signed_delta"]) <= g_["max_negctrl_abs_delta"]
    out["gate_checks"] = checks
    out["PASS"] = bool(checks) and all(checks.values())
    return out


def _demo():
    """Synthetic self-check: a real gaugeflow effect (+0.04, clusters correlated),
    a null negative control. Harness must PASS and flag negctrl as null."""
    rng = np.random.default_rng(1)
    rows = []
    for c in range(40):
        base = rng.uniform(0.55, 0.8)            # per-cluster difficulty
        for s in range(3):                        # 3 seeds
            rows.append({"arm": "baseline", "seed": s, "cluster": f"c{c}", "metric": base + rng.normal(0, 0.01)})
            rows.append({"arm": "gaugeflow", "seed": s, "cluster": f"c{c}", "metric": base + 0.04 + rng.normal(0, 0.01), "leakage": 0.03})
            rows.append({"arm": "negctrl", "seed": s, "cluster": f"c{c}", "metric": base + rng.normal(0, 0.01)})
    gates = {"min_delta_vs_baseline": 0.0, "require_delta_ci_excludes_0": True,
             "max_leakage": 0.035, "max_negctrl_abs_delta": 0.01}
    res = analyze(rows, gates, direction="higher")
    print(json.dumps(res, indent=2))
    assert res["PASS"], "demo should PASS a real +0.04 effect"
    assert res["gaugeflow_minus_baseline"]["perm_p"] < 0.05, "real effect should be significant"
    assert res["gate_checks"]["negctrl_is_null"], "shuffled-gauge control should read as null"
    assert res["gaugeflow_minus_baseline"]["ci95"][0] > 0, "CI should exclude 0"
    print("\nself-check OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results"); ap.add_argument("--gates")
    ap.add_argument("--direction", default="higher", choices=["higher", "lower"])
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--out"); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return _demo()
    if not (a.results and a.gates):
        sys.exit("need --results and --gates (or --demo)")
    res = analyze(_load(a.results), json.load(open(a.gates)), a.direction, a.bootstrap)
    txt = json.dumps(res, indent=2)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt)
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
