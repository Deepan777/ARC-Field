"""Context-baseline study: are the role coordinates better than the
established ways of describing an endpoint by its behaviour?

usage: python run_context_baselines.py --protocol chrono|hostdisjoint|nf [--seeds ...]

Every detector receives the same Tier A flow features plus one endpoint
context encoding:
  none        flow features only
  roles       ARC-Field's causal role coordinates (context.py)
  afterimage  Kitsune's AfterImage statistics, flow-level (context_alt.py)
  rolx        RolX role memberships of source and destination from causal
              ReFeX snapshot features; the NMF basis and the number of roles
              are fitted on the endpoints of the benign fitting flows only
  spectral    GraphINR-IDS's transductive spectral coordinates (UNSW only;
              ARC-Field only, as in the model ablation)
Context enters ARC-Field as the conditioning input of the field and every
other detector as extra input columns. "Kitsune" is KitNET on the AfterImage
statistics alone, i.e. the original Kitsune pipeline at flow level.
No hyper-parameter is tuned: ARC-Field and the baselines keep their frozen
configurations, the encodings their published defaults.
Output: results/context_baselines/<protocol>/<method>__<context>_s<seed>.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import DATA_DIR, PROJECT  # noqa: E402

BASES = ["MemAE", "Deep SVDD", "Dense Autoencoder", "Isolation Forest", "KitNET", "MCM"]
CTXS = ["none", "roles", "afterimage", "rolx"]
POLS = ["static", "burnin_thr", "arc"]

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="chrono", choices=["chrono", "hostdisjoint", "nf"])
ap.add_argument("--seeds", type=int, nargs="+", default=None)
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("--child", action="store_true")
args = ap.parse_args()
UNSW = args.protocol != "nf"
seeds = args.seeds or ([0, 1, 2, 3, 4] if UNSW else [0, 1, 2])
OUTDIR = os.path.join(PROJECT, "results", "context_baselines", args.protocol)
CACHE = os.path.join(PROJECT, "results", "context_baselines", "cache")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def plan():
    """(method, [contexts]) per job; one child process per (seed, method)."""
    p = [("ARC-Field", CTXS + (["spectral"] if UNSW else []))]
    p += [(b, CTXS) for b in BASES]
    p += [("Kitsune", ["afterimage-only"])]
    return [x for x in p if not args.methods or x[0] in args.methods]


def fname(meth, cx, seed):
    return os.path.join(OUTDIR, f"{meth.replace(' ', '_')}__{cx}_s{seed}.json")


if not args.child:
    from jobs import dispatch           # parent: no torch, only launches children
    jobs = [["--seeds", str(sd), "--methods", m] for sd in seeds for m, cxs in plan()
            if not all(os.path.exists(fname(m, c, sd)) for c in cxs)]
    sys.exit(1 if dispatch(os.path.abspath(__file__), ["--protocol", args.protocol], jobs,
                           log_prefix="[dispatch] ") else 0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import (resilient, _FORCE_CPU, af, device, evaluate_days, evaluate_policies,  # noqa: E402
                    fit_arcfield, save_json, to_unit, RobustScaler, log_mask_for)
import baselines as B0  # noqa: E402
import baselines_new as B1  # noqa: E402
import context_alt as ca  # noqa: E402

torch.set_num_threads(8)
ALT = os.path.join(DATA_DIR, "ctxalt")
tag = "unsw" if UNSW else "nf"
A_all = np.load(os.path.join(ALT, f"{tag}_afterimage.npy"), mmap_mode="r")
R_all = np.load(os.path.join(ALT, f"{tag}_refex.npy"), mmap_mode="r")
A_names = json.load(open(os.path.join(ALT, f"{tag}_ctxalt_names.json")))["afterimage"]
A_log = np.array(["_w_" in n for n in A_names])        # decayed weights are heavy-tailed

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


def partitions(seed):
    """{part: row indices}; NF test parts are day<d>."""
    if UNSW:
        P = {"chrono": du.chronological, "hostdisjoint": du.host_disjoint}[args.protocol](corpus, seed=seed)
        return {"train": P.train, "val": P.val, "cal": P.cal, "test": P.test}, P
    tr, va, ca_, tests = dn.cross_day(corpus, seed)
    parts = {"train": tr, "val": va, "cal": ca_}
    parts.update({f"day{d}": ix for d, ix in tests.items()})
    return parts, None


def rolx_parts(seed, parts):
    """RolX context per partition, cached per (protocol, seed)."""
    fn = os.path.join(CACHE, f"{args.protocol}_s{seed}_rolx.npz")
    if os.path.exists(fn):
        d = np.load(fn)
        return {k: d[k] for k in parts}, json.load(open(fn[:-4] + ".json"))
    t0 = time.time()
    rx = ca.RolX(seed=seed).fit(ca.rolx_fit_rows(np.asarray(R_all[parts["train"]], np.float32)))
    out = {k: ca.rolx_features(np.asarray(R_all[ix], np.float32), rx) for k, ix in parts.items()}
    meta = {"k": int(rx.k), "dl": {str(k): float(v) for k, v in rx.dl.items()}, "seconds": time.time() - t0}
    np.savez(fn[:-4] + ".tmp.npz", **out)
    os.replace(fn[:-4] + ".tmp.npz", fn)
    json.dump(meta, open(fn[:-4] + ".json", "w"))
    return out, meta


def spectral_parts(P, parts):
    import graphinr_ids_harness as rc
    ips = corpus["ips"]
    g = rc.RevGraph(spectral_k=32).fit(ips[corpus["src"][P.train]], ips[corpus["dst"][P.train]], verbose=False)

    def pe(idx):
        ps = g.eigenvectors[g.index(ips[corpus["src"][idx]])]
        pd_ = g.eigenvectors[g.index(ips[corpus["dst"][idx]])]
        return np.concatenate([ps, pd_, ps * pd_], 1).astype(np.float32)
    return {k: pe(ix) for k, ix in parts.items()}


def scaled(sc, X, chunk=50000):
    """sc(X) in row chunks: identical values, bounded peak memory (the scaler
    works in float64 internally)."""
    out = np.empty((len(X), X.shape[1]), np.float32)
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = sc(np.asarray(X[a:a + chunk], np.float32))
    return out


def context(cx, seed, parts, P):
    """Scaled context matrices per partition (scalers fitted on train)."""
    if cx == "none":
        return {k: np.zeros((len(ix), 0), np.float32) for k, ix in parts.items()}, {}
    if cx == "roles":
        raw = {k: corpus["C"][ix] for k, ix in parts.items()}
        sc, meta = RobustScaler(None), {}
    elif cx in ("afterimage", "afterimage-only"):
        raw = {k: A_all[ix] for k, ix in parts.items()}         # float32 rows from the memory map
        sc, meta = RobustScaler(A_log), {}
    elif cx == "rolx":
        raw, meta = rolx_parts(seed, parts)
        sc = RobustScaler(None)
    elif cx == "spectral":
        raw, meta = spectral_parts(P, parts), {}
        sc = RobustScaler(None)
    sc.fit(raw["train"])
    return {k: scaled(sc, v) for k, v in raw.items()}, meta


def run(meth, cx, seed, parts, P, M):
    C, cmeta = context(cx, seed, parts, P)
    tests = [k for k in parts if k not in ("train", "val", "cal")]
    res = {"method": meth, "context": cx, "protocol": args.protocol, "seed": seed,
           "context_dim": int(C["train"].shape[1]), "context_meta": cmeta}
    t0 = time.time()
    if meth == "ARC-Field":
        model, info = fit_arcfield(M["train"], C["train"], groups, seed, device())
        base = {k: af.residuals(model, M[k], C[k], groups, device()) for k in parts if k != "train"}
        res["train_seconds"] = info["train_seconds"]
        kind, kw = "matrix", {"kappa": kappa}
    else:
        if cx == "afterimage-only":
            U = {k: to_unit(C[k]) for k in parts}
        else:
            U = {k: to_unit(np.concatenate([M[k], C[k]], 1)) for k in parts}
        factory = B1.NEW_BASELINES["KitNET"] if meth == "Kitsune" else (B0.BASELINES.get(meth)
                                                                         or B1.NEW_BASELINES[meth])
        t1 = time.time()
        sf = factory(U["train"], seed)
        res["train_seconds"] = time.time() - t1
        base = {k: np.asarray(sf(U[k]), dtype=np.float64) for k in parts if k != "train"}
        kind, kw = "scalar", {}
    if UNSW:
        ctx = {"y": ~corpus["is_normal"][P.test], "labels": corpus["label"][P.test],
               "host": corpus["src"][P.test], "t": corpus["ltime"][P.test].astype(float),
               "time_resolved": args.protocol == "chrono"}
        res["policies"] = evaluate_policies(kind, {"val": base["val"], "cal": base["cal"], "test": base["test"]},
                                            POLS, ctx, kw)
    else:
        dctx = {int(k[3:]): {"y": ~corpus["is_normal"][parts[k]], "labels": corpus["label"][parts[k]],
                             "host": corpus["src"][parts[k]], "t": corpus["t"][parts[k]]} for k in tests}
        res["policies"] = evaluate_days(kind, base["val"], base["cal"], {int(k[3:]): base[k] for k in tests},
                                        dctx, POLS, kw)
    res["wall_seconds"] = time.time() - t0
    return res


for seed in seeds:
    parts, P = partitions(seed)
    Xs = RobustScaler(log_mask_for(feats, heavy)).fit(corpus["X"][parts["train"]][:, fi])
    M = {k: scaled(Xs, corpus["X"][ix][:, fi]) for k, ix in parts.items()}
    for meth, cxs in plan():
        for cx in cxs:
            fn = fname(meth, cx, seed)
            if os.path.exists(fn):
                continue
            res = resilient(run, meth, cx, seed, parts, P, M)
            _FORCE_CPU["on"] = False
            save_json(res, fn)
            s = res["policies"]["static"]
            s = s.get("pooled", s)
            a = res["policies"]["arc"]
            a = a.get("pooled", a)
            print(f"{args.protocol} s{seed} {meth:18s} {cx:16s} AUROC={s['auroc']:.4f} AUPRC={s['auprc']:.4f} "
                  f"TPR@1%={s['tpr_at_1fpr']:.4f} | arc FPR={a['fpr']:.4f} MCC={a['mcc']:.4f} "
                  f"({res['wall_seconds']:.0f}s)", flush=True)
