"""Main comparison on UNSW-NB15.

usage: python run_unsw.py --protocol chrono|hostdisjoint|random --seeds 0 1 2 3 4
                          [--featureset A] [--methods ...] [--skip-existing]

For each seed: builds the protocol, fits every detector on the same benign
partition with identical inputs, applies every calibration policy, and writes
results/raw/unsw_<protocol>_<featureset>/<method>_s<seed>.json. Base scores
(scalar) or residual matrices (float16) are cached under
results/raw/scores/... so that calibration experiments can be re-run without
refitting.
"""
import argparse
import os
import time

import numpy as np
import torch

from common import (resilient, free_gpu, _FORCE_CPU, CTX_VARIANTS, MATRIX_POLICIES, NEW_BASELINES, REUSED_BASELINES,
                    SCALAR_POLICIES, RESULTS_RAW, af, device, evaluate_policies,
                    fit_arcfield, save_json, to_unit, RobustScaler, log_mask_for)
import data_unsw as du
import baselines as B0            # eleven standard one-class baselines
import baselines_new as B1

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="chrono")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--featureset", default="A")
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("--skip-existing", action="store_true", default=True)
ap.add_argument("--recompute", action="store_true", help="ignore existing results")
ap.add_argument("--no-save-scores", action="store_true")
ap.add_argument("--child", action="store_true", help="internal: run in-process")
args = ap.parse_args()

if not args.child:
    # parent: one isolated subprocess per (seed, method) that is not done yet
    import sys as _sys
    from common import dispatch
    _all = (["ARC-Field"] + REUSED_BASELINES + NEW_BASELINES
            + [f"{m} +ctx" for m in CTX_VARIANTS])
    _outdir = os.path.join(RESULTS_RAW, f"unsw_{args.protocol}_{args.featureset}")
    jobs = []
    for sd in args.seeds:
        for m in (args.methods or _all):
            f = os.path.join(_outdir, f"{m.replace(' ', '_').replace('(', '').replace(')', '')}_s{sd}.json")
            if args.recompute or not os.path.exists(f):
                jobs.append(["--seeds", str(sd), "--methods", m])
    base = ["--protocol", args.protocol, "--featureset", args.featureset]
    if args.no_save_scores:
        base.append("--no-save-scores")
    failed = dispatch(os.path.abspath(__file__), base, jobs, log_prefix="[dispatch] ")
    _sys.exit(1 if failed else 0)

PROT = {"chrono": du.chronological, "hostdisjoint": du.host_disjoint, "random": du.random_split}
corpus = du.load_corpus()
feats = du.FEATURE_SETS[args.featureset]
fi = [du.ALL36.index(f) for f in feats]
groups = du.groups_for(feats)
kappa = max(1, round(len(feats) / 4))
tag = f"unsw_{args.protocol}_{args.featureset}"
outdir = os.path.join(RESULTS_RAW, tag)
scoredir = os.path.join(RESULTS_RAW, "scores", tag)
os.makedirs(outdir, exist_ok=True)
os.makedirs(scoredir, exist_ok=True)

all_methods = (["ARC-Field"] + REUSED_BASELINES + NEW_BASELINES
               + [f"{m} +ctx" for m in CTX_VARIANTS])
methods = args.methods or all_methods
torch.set_num_threads(8)

for seed in args.seeds:
    P = PROT[args.protocol](corpus, seed=seed)
    Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(corpus["X"][P.train][:, fi])
    Cs = RobustScaler(None).fit(corpus["C"][P.train])
    M = {k: Xs(corpus["X"][getattr(P, k)][:, fi]) for k in ("train", "val", "cal", "test")}
    C = {k: Cs(corpus["C"][getattr(P, k)]) for k in ("train", "val", "cal", "test")}
    ctx = {"y": ~corpus["is_normal"][P.test], "labels": corpus["label"][P.test],
           "host": corpus["src"][P.test], "t": corpus["ltime"][P.test].astype(float),
           "time_resolved": args.protocol == "chrono"}
    meta = {"protocol": P.name, "seed": seed, "featureset": args.featureset, "features": feats,
            "n_train": len(P.train), "n_val": len(P.val), "n_cal": len(P.cal),
            "n_test": len(P.test), "test_attack_frac": float(ctx["y"].mean()),
            "kappa": kappa, "protocol_meta": P.meta}
    print(f"\n=== {tag} seed {seed}: test {len(P.test):,} flows, "
          f"{ctx['y'].mean():.3f} attack ===", flush=True)

    for meth in methods:
        fn = os.path.join(outdir, f"{meth.replace(' ', '_').replace('(', '').replace(')', '')}_s{seed}.json")
        if args.skip_existing and not args.recompute and os.path.exists(fn):
            continue
        def one(meth=meth):
            t0 = time.time()
            res = {"method": meth, **meta}
            if meth == "ARC-Field":
                model, info = fit_arcfield(M["train"], C["train"], groups, seed, device())
                base = {k: af.residuals(model, M[k], C[k], groups, device()) for k in ("val", "cal", "test")}
                res.update(n_params=af.count_params(model), train_seconds=info["train_seconds"])
                res["policies"] = evaluate_policies("matrix", base, MATRIX_POLICIES, ctx, {"kappa": kappa})
                kind = "matrix"
            elif meth == "GraphINR-IDS":
                # published GraphINR-IDS, run unchanged (its own min-max scaling,
                # spectral and time coordinates, and mean/SD scorer)
                import graphinr_baseline as gb
                g = gb.run(corpus, P, fi, seed, device=device())
                base = {k: g[k] for k in ("val", "cal", "test")}
                res.update(n_params=g["n_params"], train_seconds=g["train_seconds"],
                           oov_test_flows=g["oov_test_flows"], graph_nodes=g["graph_nodes"])
                res["policies"] = evaluate_policies("matrix", base, MATRIX_POLICIES, ctx,
                                                    {"kappa": kappa, "scoring": "raw_meansd"})
                kind = "matrix"
            elif meth == "Anomal-E":
                # graph baseline: source hosts and destination service endpoints as nodes
                U = {k: to_unit(M[k]) for k in ("train", "val", "cal", "test")}
                svc = lambda idx: corpus["dst"][idx].astype(np.int64) * 70000 + corpus["dport"][idx].clip(0, 69999)
                t1 = time.time()
                ae = B1.AnomalE(seed).fit(U["train"], corpus["src"][P.train], svc(P.train))
                res["train_seconds"] = time.time() - t1
                base = {k: np.asarray(ae.score(U[k], corpus["src"][getattr(P, k)], svc(getattr(P, k))), dtype=np.float64)
                        for k in ("val", "cal", "test")}
                res["policies"] = evaluate_policies("scalar", base, SCALAR_POLICIES, ctx)
                kind = "scalar"
            else:
                name = meth.replace(" +ctx", "")
                use_ctx = meth.endswith("+ctx")
                U = {k: to_unit(np.concatenate([M[k], C[k]], 1) if use_ctx else M[k])
                     for k in ("train", "val", "cal", "test")}
                factory = B0.BASELINES.get(name) or B1.NEW_BASELINES[name]
                t1 = time.time()
                sf = factory(U["train"], seed)
                res["train_seconds"] = time.time() - t1
                base = {k: np.asarray(sf(U[k]), dtype=np.float64) for k in ("val", "cal", "test")}
                res["policies"] = evaluate_policies("scalar", base, SCALAR_POLICIES, ctx)
                kind = "scalar"

            res["wall_seconds"] = time.time() - t0
            return res, base, kind

        res, base, kind = resilient(one)
        _FORCE_CPU["on"] = False
        if not args.no_save_scores:        # scores first, result file last (= done marker)
            npz = fn.replace(outdir, scoredir).replace(".json", ".npz")
            np.savez_compressed(npz[:-4] + ".tmp.npz",
                                **{k: (v.astype(np.float16) if kind == "matrix" else v.astype(np.float32))
                                   for k, v in base.items()})
            os.replace(npz[:-4] + ".tmp.npz", npz)
        save_json(res, fn)
        s = res["policies"].get("arc", {})
        st = res["policies"]["static"]
        print(f"  {meth:30s} static F1={st['f1']:.4f} FPR={st['fpr']:.4f} AUROC={st['auroc']:.4f} | "
              f"arc F1={s.get('f1', float('nan')):.4f} FPR={s.get('fpr', float('nan')):.4f} "
              f"({res['wall_seconds']:.0f}s)", flush=True)
