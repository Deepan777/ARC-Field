"""Component ablation of the ARC-Field detector (retrained per variant).

usage: python run_ablation_model.py --protocol chrono|hostdisjoint --seeds 0 1 2 3 4

Variants
  full            ARC-Field as specified
  no_context      c removed (masked field over metadata only)
  src_context     only source-side role coordinates (+pair, proto)
  dst_context     only destination-side role coordinates (+pair, proto)
  unmasked        every feature visible; reconstruct all (identity admissible)
  heteroscedastic Gaussian head, beta-NLL, z^2 residuals (initial design)
  relu            SIREN replaced by a ReLU MLP of the same width/depth
  spectral_coord  role coordinates replaced by GraphINR-IDS spectral endpoint
                  coordinates (transductive; zero vector for unseen endpoints)
  abs_time        role coordinates + GraphINR-IDS absolute-time encoding TE(t)
Output: results/ablation/<protocol>/<variant>_s<seed>.json
"""
import argparse
import os
import time

import numpy as np

from common import (resilient, _FORCE_CPU, MATRIX_POLICIES, af, device, evaluate_policies, fit_arcfield,
                    save_json, RobustScaler, log_mask_for)
from paths import RESULTS_ABL
import data_unsw as du
import graphinr_ids_harness as rc

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="chrono")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--variants", nargs="*", default=None)
ap.add_argument("--child", action="store_true")
args = ap.parse_args()
if not args.child:
    import sys as _sys
    from common import dispatch
    _vs = args.variants or ["full", "no_context", "src_context", "dst_context", "unmasked",
                            "heteroscedastic", "relu", "spectral_coord", "abs_time"]
    _od = os.path.join(RESULTS_ABL, f"model_{args.protocol}")
    jobs = [["--seeds", str(sd), "--variants", v] for sd in args.seeds for v in _vs
            if not os.path.exists(os.path.join(_od, f"{v}_s{sd}.json"))]
    _sys.exit(1 if dispatch(os.path.abspath(__file__), ["--protocol", args.protocol], jobs,
                            log_prefix="[dispatch] ") else 0)

VARIANTS = ["full", "no_context", "src_context", "dst_context", "unmasked", "heteroscedastic",
            "relu", "spectral_coord", "abs_time"]
variants = args.variants or VARIANTS
PROT = {"chrono": du.chronological, "hostdisjoint": du.host_disjoint, "random": du.random_split}
corpus = du.load_corpus()
feats = du.FEATURE_SETS["A"]
fi = [du.ALL36.index(f) for f in feats]
groups = du.groups_for(feats)
kappa = max(1, round(len(feats) / 4))
names = list(corpus["ctx_names"])
src_cols = [i for i, n in enumerate(names) if not n.startswith("dst_")]
dst_cols = [i for i, n in enumerate(names) if not n.startswith("src_")]
outdir = os.path.join(RESULTS_ABL, f"model_{args.protocol}")
os.makedirs(outdir, exist_ok=True)
ips = corpus["ips"]


def spectral_pe(P):
    g = rc.RevGraph(spectral_k=32).fit(ips[corpus["src"][P.train]], ips[corpus["dst"][P.train]],
                                        verbose=False)

    def pe(idx):
        ps = g.eigenvectors[g.index(ips[corpus["src"][idx]])]
        pd_ = g.eigenvectors[g.index(ips[corpus["dst"][idx]])]
        return np.concatenate([ps, pd_, ps * pd_], 1).astype(np.float32)
    return pe


def abs_time_te(P):
    tn = rc.TimeNormaliser.fit(corpus["stime"][P.train].astype(np.float64))
    freqs = np.logspace(0, 4, 16)

    def te(idx):
        t = tn(corpus["stime"][idx].astype(np.float64)).astype(np.float64)
        ph = 2 * np.pi * np.outer(t, freqs)
        return np.concatenate([np.sin(ph), np.cos(ph)], 1).astype(np.float32)
    return te


for seed in args.seeds:
    P = PROT[args.protocol](corpus, seed=seed)
    Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(corpus["X"][P.train][:, fi])
    Cs = RobustScaler(None).fit(corpus["C"][P.train])
    parts = ("train", "val", "cal", "test")
    M = {k: Xs(corpus["X"][getattr(P, k)][:, fi]) for k in parts}
    C = {k: Cs(corpus["C"][getattr(P, k)]) for k in parts}
    ctx = {"y": ~corpus["is_normal"][P.test], "labels": corpus["label"][P.test],
           "host": corpus["src"][P.test], "t": corpus["ltime"][P.test].astype(float),
           "time_resolved": args.protocol == "chrono"}
    for var in variants:
        fn = os.path.join(outdir, f"{var}_s{seed}.json")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        over, CC, masked = {}, C, True
        if var == "no_context":
            CC = {k: np.zeros((len(M[k]), 0), np.float32) for k in parts}
        elif var == "src_context":
            CC = {k: C[k][:, src_cols] for k in parts}
        elif var == "dst_context":
            CC = {k: C[k][:, dst_cols] for k in parts}
        elif var == "unmasked":
            over["masked"] = False
            masked = False
        elif var == "heteroscedastic":
            over["hetero"] = True
        elif var == "relu":
            over["activation"] = "relu"
        elif var == "spectral_coord":
            pe = spectral_pe(P)
            CC = {k: pe(getattr(P, k)) for k in parts}
        elif var == "abs_time":
            te = abs_time_te(P)
            CC = {k: np.concatenate([C[k], te(getattr(P, k))], 1) for k in parts}
        def fit_score():
            model, info = fit_arcfield(M["train"], CC["train"], groups, seed, device(), **over)
            return model, info, {k: af.residuals(model, M[k], CC[k], groups, device(), masked=masked)
                                 for k in ("val", "cal", "test")}
        model, info, base = resilient(fit_score)
        _FORCE_CPU["on"] = False
        res = {"variant": var, "protocol": P.name, "seed": seed, "n_params": af.count_params(model),
               "train_seconds": info["train_seconds"], "context_dim": int(CC["train"].shape[1]),
               "policies": evaluate_policies("matrix", base, ["static", "burnin_thr", "arc"], ctx,
                                             {"kappa": kappa})}
        res["wall_seconds"] = time.time() - t0
        save_json(res, fn)
        a, s = res["policies"]["arc"], res["policies"]["static"]
        print(f"{args.protocol} s{seed} {var:16s} AUROC={s['auroc']:.4f} AUPRC={s['auprc']:.4f} "
              f"| arc F1={a['f1']:.4f} FPR={a['fpr']:.4f} MCC={a['mcc']:.4f}", flush=True)
