"""Protocol V: NF-CSE-CIC-IDS2018-v3, fit on the benign flows of the first
capture day, stream every later day separately (each recalibrated on its own
first window, as an operator would at the start of a working day).

Addresses and timestamps are available, so ARC-Field uses host-level role
coordinates, the per-host cap is active and Anomal-E can be run.
Metrics are reported per day and pooled over days (decisions concatenated;
ranking metrics averaged over days, because independently recalibrated days
are not on a common score scale).
Output: results/raw/nf_crossday_<featureset>/<method>_s<seed>.json
"""
import argparse
import os
import time

import numpy as np
import torch

from common import (resilient, _FORCE_CPU, MATRIX_POLICIES, SCALAR_POLICIES, NEW_BASELINES, REUSED_BASELINES,
                    RESULTS_RAW, af, cal, device, fit_arcfield, mt, save_json, to_unit, RobustScaler,
                    log_mask_for)
import data_nf as dn
import baselines as B0
import baselines_new as B1

ALL_METHODS = (["ARC-Field", "ARC-Field (no context)"] + REUSED_BASELINES + NEW_BASELINES)

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--featureset", default="A")
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("--no-save-scores", action="store_true")
ap.add_argument("--child", action="store_true")
args = ap.parse_args()
tag = f"nf_crossday_{args.featureset}"
outdir = os.path.join(RESULTS_RAW, tag)
scoredir = os.path.join(RESULTS_RAW, "scores", tag)


def fname(meth, seed):
    return os.path.join(outdir, f"{meth.replace(' ', '_').replace('(', '').replace(')', '')}_s{seed}.json")


if not args.child:
    import sys as _sys
    from common import dispatch
    dn.load_corpus(verbose=True)            # build the cache once, in the parent
    jobs = [["--seeds", str(sd), "--methods", m] for sd in args.seeds for m in (args.methods or ALL_METHODS)
            if not os.path.exists(fname(m, sd))]
    base = ["--featureset", args.featureset] + (["--no-save-scores"] if args.no_save_scores else [])
    _sys.exit(1 if dispatch(os.path.abspath(__file__), base, jobs, log_prefix="[dispatch] ") else 0)

corpus = dn.load_corpus(verbose=False)
feats = dn.FEATURE_SETS[args.featureset]
fi = [dn.ALL.index(f) for f in feats]
groups = dn.groups_for(feats)
kappa = max(1, round(len(feats) / 4))
os.makedirs(outdir, exist_ok=True)
os.makedirs(scoredir, exist_ok=True)
torch.set_num_threads(8)


def evaluate(kind, base_val, base_cal, tests, ctx, pols, cfg_kw):
    out = {}
    for pol in pols:
        kw = dict(cfg_kw)
        if kind == "matrix" and pol in ("burnin_full", "burnin_oracle"):
            kw["scoring"] = "raw_meansd"
        cfg = cal.CalConfig(**kw)
        A, Y, L, per_day = [], [], [], {}
        for d, T in tests.items():
            c = ctx[d]
            if kind == "matrix":
                o = cal.run_policy(pol, cfg, R_val=base_val, R_cal=base_cal, R_test=T, host=c["host"],
                                   benign_mask=~c["y"])
            else:
                o = cal.run_policy(pol, cfg, s_val=base_val, s_cal=base_cal, s_test=T, host=c["host"],
                                   benign_mask=~c["y"])
            ei = o["eval_index"]
            y, lab = c["y"][ei], c["labels"][ei]
            m = mt.all_metrics(o["score"], o["alarm"], y, cfg.alpha)
            m["per_family"] = mt.per_family(o["alarm"], lab)
            m["time_resolved"] = mt.time_resolved(o["alarm"], y, c["t"][ei])
            per_day[f"day{d}"] = m
            A.append(o["alarm"]); Y.append(y); L.append(lab)
        pooled = mt.decision_metrics(np.concatenate(A), np.concatenate(Y))
        for k in ("auroc", "auprc", "tpr_at_1fpr"):      # days without attacks give NaN
            pooled[k] = float(np.nanmean([v[k] for v in per_day.values()]))
        bm = [v["time_resolved"]["fpr_bin_max"] for v in per_day.values()]
        pooled["fpr_bin_max"] = float(max(b for b in bm if b is not None))
        pooled["fpr_day_max"] = float(max(v["fpr"] for v in per_day.values()))
        pooled["per_family"] = mt.per_family(np.concatenate(A), np.concatenate(L))
        out[pol] = {"pooled": pooled, "per_day": per_day}
    return out


for seed in args.seeds:
    tr, va, ca, tests = dn.cross_day(corpus, seed)
    X = corpus["X"][:, fi]
    Xs = RobustScaler(log_mask_for(feats, dn.HEAVY)).fit(X[tr])
    Cs = RobustScaler(None).fit(corpus["C"][tr])
    M = {"train": Xs(X[tr]), "val": Xs(X[va]), "cal": Xs(X[ca])}
    C = {"train": Cs(corpus["C"][tr]), "val": Cs(corpus["C"][va]), "cal": Cs(corpus["C"][ca])}
    Mt = {d: Xs(X[ix]) for d, ix in tests.items()}
    Ct = {d: Cs(corpus["C"][ix]) for d, ix in tests.items()}
    ctx = {d: {"y": ~corpus["is_normal"][ix], "labels": corpus["label"][ix], "host": corpus["src"][ix],
               "t": corpus["t"][ix]} for d, ix in tests.items()}
    svc = lambda ix: corpus["dst"][ix] * 70000 + corpus["dport"][ix].clip(0, 69999)  # noqa: E731
    print(f"\n=== {tag} seed {seed}: fit {len(tr):,}; " + ", ".join(
        f"day{d} {len(ix):,} ({ctx[d]['y'].mean():.3f})" for d, ix in tests.items()), flush=True)

    for meth in (args.methods or ALL_METHODS):
        fn = fname(meth, seed)
        if os.path.exists(fn):
            continue

        def one(meth=meth):
            t0 = time.time()
            res = {"method": meth, "seed": seed, "featureset": args.featureset, "features": feats,
                   "kappa": kappa, "n_train": len(tr),
                   "test_sizes": {f"day{d}": len(ix) for d, ix in tests.items()},
                   "test_attack_share": {f"day{d}": float(ctx[d]["y"].mean()) for d in tests}}
            if meth.startswith("ARC-Field"):
                use_c = "no context" not in meth
                Z = (lambda A_: A_) if use_c else (lambda A_: np.zeros((len(A_), 0), np.float32))
                model, info = fit_arcfield(M["train"], Z(C["train"]), groups, seed, device())
                Rv = af.residuals(model, M["val"], Z(C["val"]), groups, device())
                Rc = af.residuals(model, M["cal"], Z(C["cal"]), groups, device())
                Rt = {d: af.residuals(model, Mt[d], Z(Ct[d]), groups, device()) for d in tests}
                res.update(n_params=af.count_params(model), train_seconds=info["train_seconds"])
                res["policies"] = evaluate("matrix", Rv, Rc, Rt, ctx, MATRIX_POLICIES, {"kappa": kappa})
                scores = {"val": Rv, "cal": Rc, **{f"day{d}": v for d, v in Rt.items()}}
                kind = "matrix"
            elif meth == "GraphINR-IDS":
                # Published GraphINR-IDS (min-max scaling, its own mean/SD scorer);
                # spectral graph of the fitting flows (at most 10,000 nodes, the
                # published cap) and time coordinate from the flow end time, the
                # only timestamp kept in the corpus cache.
                import graphinr_baseline as gb
                parts = {"train": tr, "val": va, "cal": ca, **{f"day{d}": ix for d, ix in tests.items()}}
                g = gb.run_parts(X, corpus["src"], corpus["dst"], corpus["t"], corpus["is_normal"], parts,
                                 seed, device=device())
                Rv, Rc = g["val"], g["cal"]
                Rt = {d: g[f"day{d}"] for d in tests}
                res.update(n_params=g["n_params"], train_seconds=g["train_seconds"],
                           graph_nodes=g["graph_nodes"], oov_test_flows=g["oov_test_flows"])
                res["policies"] = evaluate("matrix", Rv, Rc, Rt, ctx, MATRIX_POLICIES,
                                           {"kappa": kappa, "scoring": "raw_meansd"})
                scores = {"val": Rv, "cal": Rc, **{f"day{d}": v for d, v in Rt.items()}}
                kind = "matrix"
            else:
                U = {k: to_unit(v) for k, v in M.items()}
                Ut = {d: to_unit(v) for d, v in Mt.items()}
                t1 = time.time()
                if meth == "Anomal-E":
                    ae = B1.AnomalE(seed).fit(U["train"], corpus["src"][tr], svc(tr))
                    res["train_seconds"] = time.time() - t1
                    sv = np.asarray(ae.score(U["val"], corpus["src"][va], svc(va)), float)
                    sc = np.asarray(ae.score(U["cal"], corpus["src"][ca], svc(ca)), float)
                    st = {d: np.asarray(ae.score(Ut[d], corpus["src"][ix], svc(ix)), float)
                          for d, ix in tests.items()}
                else:
                    factory = B0.BASELINES.get(meth) or B1.NEW_BASELINES[meth]
                    sf = factory(U["train"], seed)
                    res["train_seconds"] = time.time() - t1
                    sv, sc = np.asarray(sf(U["val"]), float), np.asarray(sf(U["cal"]), float)
                    st = {d: np.asarray(sf(Ut[d]), float) for d in tests}
                res["policies"] = evaluate("scalar", sv, sc, st, ctx, SCALAR_POLICIES, {})
                scores = {"val": sv, "cal": sc, **{f"day{d}": v for d, v in st.items()}}
                kind = "scalar"
            res["wall_seconds"] = time.time() - t0
            return res, scores, kind

        res, scores, kind = resilient(one)
        _FORCE_CPU["on"] = False
        if not args.no_save_scores:        # scores first, result file last (= done marker)
            npz = fn.replace(outdir, scoredir).replace(".json", ".npz")
            np.savez_compressed(npz[:-4] + ".tmp.npz",
                                **{k: (v.astype(np.float16) if kind == "matrix" else v.astype(np.float32))
                                   for k, v in scores.items()})
            os.replace(npz[:-4] + ".tmp.npz", npz)
        save_json(res, fn)
        a, s = res["policies"]["arc"]["pooled"], res["policies"]["static"]["pooled"]
        print(f"  {meth:28s} static AUROC={s['auroc']:.4f} AUPRC={s['auprc']:.4f} FPR={s['fpr']:.4f} | "
              f"arc FPR={a['fpr']:.4f} rec={a['recall']:.4f} MCC={a['mcc']:.4f} ({res['wall_seconds']:.0f}s)",
              flush=True)
