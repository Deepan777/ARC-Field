"""Role-anomaly extension of ARC-Field (pre-registered in
00_project_management/decisions_log.md, item 26, before any run).

ARC-Field scores a flow by how unusual it is *given* its endpoint context
(conditional field). It cannot flag an endpoint whose context itself is
unusual. The extension adds a second, separately calibrated test: a masked
field fitted on the context alone (role anomaly). Each test is recalibrated
with the same policy at alpha/2; a flow alarms if either test alarms, so the
false-positive rate stays below alpha by the union bound, and its ranking
score is the larger of the two margins over the tests' own thresholds. No
weight is tuned.

Variants
  M0  conditional on role coordinates (ARC-Field as published)
  M1  M0 + role anomaly on role coordinates
  M2  M0 + role anomaly on RolX roles
  M3  conditional on role coordinates + RolX roles
  M4  conditional on roles + RolX, role anomaly on roles + RolX
      (designated method, "ARC-Field+R")
usage: python run_role_anomaly.py --protocol dev|chrono|hostdisjoint|nf [--seeds ...]
Output: results/role_anomaly/<protocol>/<variant>_s<seed>.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import DATA_DIR, PROJECT  # noqa: E402

VARIANTS = {"M0": ("roles", None), "M1": ("roles", "roles"), "M2": ("roles", "rolx"),
            "M3": ("roles+rolx", None), "M4": ("roles+rolx", "roles+rolx")}
POLS = ["static", "burnin_thr", "arc"]

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="chrono", choices=["dev", "chrono", "hostdisjoint", "nf"])
ap.add_argument("--seeds", type=int, nargs="+", default=None)
ap.add_argument("--variants", nargs="*", default=None)
ap.add_argument("--child", action="store_true")
args = ap.parse_args()
UNSW = args.protocol != "nf"
seeds = args.seeds or {"dev": [0, 1, 2], "chrono": [0, 1, 2, 3, 4], "hostdisjoint": [0, 1, 2, 3, 4],
                       "nf": [0, 1, 2]}[args.protocol]
variants = args.variants or list(VARIANTS)
OUTDIR = os.path.join(PROJECT, "results", "role_anomaly", args.protocol)
CACHE = os.path.join(PROJECT, "results", "context_baselines", "cache")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def fname(v, seed):
    return os.path.join(OUTDIR, f"{v}_s{seed}.json")


if not args.child:
    from jobs import dispatch
    jobs = [["--seeds", str(sd), "--variants", v] for sd in seeds for v in variants if not os.path.exists(fname(v, sd))]
    sys.exit(1 if dispatch(os.path.abspath(__file__), ["--protocol", args.protocol], jobs,
                           log_prefix="[dispatch] ") else 0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import _FORCE_CPU, af, cal, device, fit_arcfield, mt, resilient, save_json, RobustScaler, log_mask_for  # noqa: E402
import context_alt as ca  # noqa: E402

torch.set_num_threads(8)
tag = "unsw" if UNSW else "nf"
R_all = np.load(os.path.join(DATA_DIR, "ctxalt", f"{tag}_refex.npy"), mmap_mode="r")
if UNSW:
    import data_unsw as du
    corpus = du.load_corpus(verbose=False)
    feats = du.FEATURE_SETS["A"]
    fi = [du.ALL36.index(f) for f in feats]
    groups = du.groups_for(feats)
    heavy = du.HEAVY
else:
    import data_nf as dn
    corpus = dn.load_corpus(verbose=False)
    feats = dn.FEATURE_SETS["A"]
    fi = [dn.ALL.index(f) for f in feats]
    groups = dn.groups_for(feats)
    heavy = dn.HEAVY
kappa = max(1, round(len(feats) / 4))
role_names = list(corpus["ctx_names"])


def partitions(seed):
    if UNSW:
        fn = {"dev": du.development, "chrono": du.chronological, "hostdisjoint": du.host_disjoint}[args.protocol]
        P = fn(corpus, seed=seed)
        return {"train": P.train, "val": P.val, "cal": P.cal, "test": P.test}
    tr, va, ca_, tests = dn.cross_day(corpus, seed)
    parts = {"train": tr, "val": va, "cal": ca_}
    parts.update({f"day{d}": ix for d, ix in tests.items()})
    return parts


def scaled(sc, X, chunk=50000):
    out = np.empty((len(X), X.shape[1]), np.float32)
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = sc(np.asarray(X[a:a + chunk], np.float32))
    return out


def rolx_raw(seed, parts):
    """RolX context per partition; shares the context-baseline study's cache
    (same partitions and seeds give identical features)."""
    fn = os.path.join(CACHE, f"{args.protocol}_s{seed}_rolx.npz")
    if os.path.exists(fn):
        d = np.load(fn)
        if all(k in d.files for k in parts):
            return {k: d[k] for k in parts}
    rx = ca.RolX(seed=seed).fit(ca.rolx_fit_rows(np.asarray(R_all[parts["train"]], np.float32)))
    out = {k: ca.rolx_features(np.asarray(R_all[ix], np.float32), rx) for k, ix in parts.items()}
    np.savez(fn[:-4] + ".tmp.npz", **out)
    os.replace(fn[:-4] + ".tmp.npz", fn)
    json.dump({"k": int(rx.k), "dl": {str(k): float(v) for k, v in rx.dl.items()}}, open(fn[:-4] + ".json", "w"))
    return out


def context(spec, seed, parts):
    """Scaled context per partition and its observability groups (for the
    role-anomaly field): role coordinates split into source side,
    destination side and pair/protocol; RolX into source and destination."""
    blocks, gnames = [], []
    if "roles" in spec.split("+"):
        blocks.append({k: corpus["C"][ix] for k, ix in parts.items()})
        gnames.append(["src" if n.startswith("src_") else "dst" if n.startswith("dst_") else "pair"
                       for n in role_names])
    if "rolx" in spec.split("+"):
        raw = rolx_raw(seed, parts)
        blocks.append(raw)
        half = raw["train"].shape[1] // 2
        gnames.append(["rolx_src"] * half + ["rolx_dst"] * half)
    raw = {k: np.concatenate([b[k] for b in blocks], 1) for k in parts}
    sc = RobustScaler(None).fit(raw["train"])
    C = {k: scaled(sc, v) for k, v in raw.items()}
    names = sum(gnames, [])
    grp = [[i for i, g in enumerate(names) if g == u] for u in dict.fromkeys(names)]
    return C, grp


def evaluate(R1, k1, R2, k2, parts, alpha=0.05):
    """Single test (R2 None) or OR-fusion of two tests at alpha/2 each."""
    tests = [k for k in parts if k not in ("train", "val", "cal")]
    out = {}
    for pol in POLS:
        A, Y, L, per = [], [], [], {}
        for k in tests:
            ix = parts[k]
            host, y = corpus["src"][ix], ~corpus["is_normal"][ix]
            a = alpha if R2 is None else alpha / 2
            o1 = cal.run_policy(pol, cal.CalConfig(alpha=a, kappa=k1), R_val=R1["val"], R_cal=R1["cal"],
                                R_test=R1[k], host=host, benign_mask=~y)
            if R2 is None:
                score, alarm, ei = o1["score"], o1["alarm"], o1["eval_index"]
            else:
                o2 = cal.run_policy(pol, cal.CalConfig(alpha=a, kappa=k2), R_val=R2["val"], R_cal=R2["cal"],
                                    R_test=R2[k], host=host, benign_mask=~y)
                assert np.array_equal(o1["eval_index"], o2["eval_index"])
                ei = o1["eval_index"]
                alarm = o1["alarm"] | o2["alarm"]
                score = np.maximum(o1["score"] - o1["tau"], o2["score"] - o2["tau"])
            yy, lab = y[ei], corpus["label"][ix][ei]
            m = mt.all_metrics(score, alarm, yy, alpha)
            m["per_family"] = mt.per_family(alarm, lab)
            if R2 is not None:
                m["share_alarms_role_test_only"] = float(((o2["alarm"] & ~o1["alarm"]).sum())
                                                         / max(alarm.sum(), 1))
            per[k] = m
            A.append(alarm); Y.append(yy); L.append(lab)
        if UNSW:
            out[pol] = per["test"]
        else:
            pooled = mt.decision_metrics(np.concatenate(A), np.concatenate(Y))
            for key in ("auroc", "auprc", "tpr_at_1fpr"):
                pooled[key] = float(np.nanmean([v[key] for v in per.values()]))
            pooled["fpr_day_max"] = float(max(v["fpr"] for v in per.values()))
            pooled["per_family"] = mt.per_family(np.concatenate(A), np.concatenate(L))
            out[pol] = {"pooled": pooled, "per_day": per}
    return out


def run(v, seed, parts, M):
    cond, anom = VARIANTS[v]
    t0 = time.time()
    Cc, _ = context(cond, seed, parts)
    model, info = fit_arcfield(M["train"], Cc["train"], groups, seed, device())
    R1 = {k: af.residuals(model, M[k], Cc[k], groups, device()) for k in parts if k != "train"}
    res = {"variant": v, "conditional_context": cond, "anomaly_context": anom, "protocol": args.protocol,
           "seed": seed, "context_dim": int(Cc["train"].shape[1]), "train_seconds": info["train_seconds"]}
    R2, k2 = None, None
    if anom:
        Ca, ga = context(anom, seed, parts)
        Z = {k: np.zeros((len(x), 0), np.float32) for k, x in Ca.items()}
        model2, info2 = fit_arcfield(Ca["train"], Z["train"], ga, seed, device())
        R2 = {k: af.residuals(model2, Ca[k], Z[k], ga, device()) for k in parts if k != "train"}
        k2 = max(1, round(Ca["train"].shape[1] / 4))
        res.update(anomaly_dim=int(Ca["train"].shape[1]), anomaly_groups=len(ga),
                   anomaly_train_seconds=info2["train_seconds"])
    res["policies"] = evaluate(R1, kappa, R2, k2, parts)
    res["wall_seconds"] = time.time() - t0
    return res


for seed in seeds:
    parts = partitions(seed)
    Xs = RobustScaler(log_mask_for(feats, heavy)).fit(corpus["X"][parts["train"]][:, fi])
    M = {k: scaled(Xs, corpus["X"][ix][:, fi]) for k, ix in parts.items()}
    for v in variants:
        fn = fname(v, seed)
        if os.path.exists(fn):
            continue
        res = resilient(run, v, seed, parts, M)
        _FORCE_CPU["on"] = False
        save_json(res, fn)
        s, a = res["policies"]["static"], res["policies"]["arc"]
        s, a = s.get("pooled", s), a.get("pooled", a)
        print(f"{args.protocol} s{seed} {v} AUROC={s['auroc']:.4f} AUPRC={s['auprc']:.4f} "
              f"TPR@1%={s['tpr_at_1fpr']:.4f} | arc FPR={a['fpr']:.4f} MCC={a['mcc']:.4f} "
              f"({res['wall_seconds']:.0f}s)", flush=True)
