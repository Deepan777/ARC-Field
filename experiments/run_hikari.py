"""Protocol VI: HIKARI-2021, genuinely encrypted (HTTPS) attacks, flow level.

usage: python run_hikari.py --setting random|shift [--seeds 0 1 2] [--featureset A]

The public flow table has no timestamps and stores each traffic category as
a block, so no causal stream exists: ARC-Field runs without endpoint context
(the masked field over Tier A flow features with the ARC recalibration),
and the test stream is in a seeded random order (data_hikari.split). Every
detector receives identical inputs and partitions, as in run_unsw.py.
Output: results/raw/hikari_<setting>_<featureset>/<method>_s<seed>.json
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import RESULTS_RAW  # noqa: E402

REUSED = ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Dense Autoencoder", "LSTM Autoencoder",
          "Deep SVDD", "DAGMM", "MemAE", "USAD", "Transformer Autoencoder", "Diffusion (DDPM) detector"]
NEW = ["KitNET", "MCM", "DIF", "Anomal-E"]
ALL_METHODS = ["ARC-Field"] + REUSED + NEW

ap = argparse.ArgumentParser()
ap.add_argument("--setting", default="shift", choices=["random", "shift"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--featureset", default="A")
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("--child", action="store_true")
args = ap.parse_args()
tag = f"hikari_{args.setting}_{args.featureset}"
outdir = os.path.join(RESULTS_RAW, tag)


def fname(meth, seed):
    return os.path.join(outdir, f"{meth.replace(' ', '_').replace('(', '').replace(')', '')}_s{seed}.json")


if not args.child:
    from jobs import dispatch
    jobs = [["--seeds", str(sd), "--methods", m] for sd in args.seeds for m in (args.methods or ALL_METHODS)
            if not os.path.exists(fname(m, sd))]
    sys.exit(1 if dispatch(os.path.abspath(__file__), ["--setting", args.setting, "--featureset", args.featureset],
                           jobs, log_prefix="[dispatch] ") else 0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import (resilient, _FORCE_CPU, MATRIX_POLICIES, SCALAR_POLICIES, af, device,  # noqa: E402
                    evaluate_policies, fit_arcfield, save_json, to_unit, RobustScaler, log_mask_for)
import data_hikari as dh  # noqa: E402
import baselines as B0  # noqa: E402
import baselines_new as B1  # noqa: E402

torch.set_num_threads(8)
os.makedirs(outdir, exist_ok=True)
corpus = dh.load_corpus(verbose=False)
feats = dh.FEATURE_SETS[args.featureset]
fi = [dh.ALL.index(f) for f in feats]
groups = dh.groups_for(feats)
kappa = max(1, round(len(feats) / 4))

for seed in args.seeds:
    tr, va, ca_, te = dh.split(corpus, args.setting, seed)
    parts = {"train": tr, "val": va, "cal": ca_, "test": te}
    Xs = RobustScaler(log_mask_for(feats, dh.HEAVY)).fit(corpus["X"][tr][:, fi])
    M = {k: Xs(corpus["X"][ix][:, fi]) for k, ix in parts.items()}
    ctx = {"y": ~corpus["is_normal"][te], "labels": corpus["label"][te], "host": corpus["src"][te],
           "t": np.arange(len(te), dtype=float), "time_resolved": False}
    meta = {"protocol": f"hikari_{args.setting}_s{seed}", "seed": seed, "featureset": args.featureset,
            "features": feats, "n_train": len(tr), "n_test": len(te), "kappa": kappa,
            "test_attack_frac": float(ctx["y"].mean()),
            "test_categories": {str(k): int(v) for k, v in zip(*np.unique(corpus["category"][te], return_counts=True))}}
    print(f"\n=== {tag} seed {seed}: test {len(te):,} flows, {ctx['y'].mean():.3f} attack", flush=True)
    for meth in (args.methods or ALL_METHODS):
        fn = fname(meth, seed)
        if os.path.exists(fn):
            continue

        def one(meth=meth):
            t0 = time.time()
            res = {"method": meth, **meta}
            if meth == "ARC-Field":
                Z = {k: np.zeros((len(v), 0), np.float32) for k, v in M.items()}
                model, info = fit_arcfield(M["train"], Z["train"], groups, seed, device())
                base = {k: af.residuals(model, M[k], Z[k], groups, device()) for k in ("val", "cal", "test")}
                res.update(n_params=af.count_params(model), train_seconds=info["train_seconds"])
                res["policies"] = evaluate_policies("matrix", base, MATRIX_POLICIES, ctx, {"kappa": kappa})
            elif meth == "GraphINR-IDS":
                # Published GraphINR-IDS (min-max scaling, its own mean/SD scorer).
                # Addresses exist, so the spectral coordinate is built from the
                # fitting flows; the table has no timestamps, so the time
                # coordinate is constant (inert).
                import graphinr_baseline as gb
                n = len(corpus["is_normal"])
                g = gb.run_parts(corpus["X"][:, fi], corpus["src"], corpus["dst"], np.zeros(n),
                                 corpus["is_normal"], parts, seed, device=device())
                base = {k: g[k] for k in ("val", "cal", "test")}
                res.update(n_params=g["n_params"], train_seconds=g["train_seconds"],
                           graph_nodes=g["graph_nodes"], oov_test_flows=g["oov_test_flows"])
                res["policies"] = evaluate_policies("matrix", base, MATRIX_POLICIES, ctx,
                                                    {"kappa": kappa, "scoring": "raw_meansd"})
            else:
                U = {k: to_unit(v) for k, v in M.items()}
                t1 = time.time()
                if meth == "Anomal-E":
                    svc = lambda ix: corpus["dst"][ix] * 70000 + corpus["dport"][ix].clip(0, 69999)  # noqa: E731
                    ae = B1.AnomalE(seed).fit(U["train"], corpus["src"][tr], svc(tr))
                    res["train_seconds"] = time.time() - t1
                    base = {k: np.asarray(ae.score(U[k], corpus["src"][parts[k]], svc(parts[k])), float)
                            for k in ("val", "cal", "test")}
                else:
                    factory = B0.BASELINES.get(meth) or B1.NEW_BASELINES[meth]
                    sf = factory(U["train"], seed)
                    res["train_seconds"] = time.time() - t1
                    base = {k: np.asarray(sf(U[k]), float) for k in ("val", "cal", "test")}
                res["policies"] = evaluate_policies("scalar", base, SCALAR_POLICIES, ctx)
            res["wall_seconds"] = time.time() - t0
            return res

        res = resilient(one)
        _FORCE_CPU["on"] = False
        save_json(res, fn)
        s, a = res["policies"]["static"], res["policies"]["arc"]
        print(f"  {meth:28s} static AUROC={s['auroc']:.4f} AUPRC={s['auprc']:.4f} TPR@1%={s['tpr_at_1fpr']:.4f} "
              f"FPR={s['fpr']:.4f} | arc FPR={a['fpr']:.4f} MCC={a['mcc']:.4f} ({res['wall_seconds']:.0f}s)",
              flush=True)
