"""Calibrator ablation and sensitivity, from cached detector outputs (no
refitting). Uses results/raw/scores/unsw_<protocol>_A/<method>_s<seed>.npz.

Parts
  ablation     ARC calibrator with one component removed or replaced
  sensitivity  trim constant c, burn-in size B, target alpha (realised vs
               target FPR), host-cap share q
Output: results/ablation/calibration_<protocol>.json
"""
import argparse
import os

import numpy as np

from common import RESULTS_RAW, cal, mt, save_json
from paths import RESULTS_ABL
import data_unsw as du

ap = argparse.ArgumentParser()
ap.add_argument("--protocol", default="chrono")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--methods", nargs="*", default=["ARC-Field", "MemAE", "Dense_Autoencoder", "KitNET", "Deep_SVDD", "DIF", "Isolation_Forest"])
args = ap.parse_args()
PROT = {"chrono": du.chronological, "hostdisjoint": du.host_disjoint, "random": du.random_split}
corpus = du.load_corpus()
sdir = os.path.join(RESULTS_RAW, "scores", f"unsw_{args.protocol}_A")
MATRIX = {"ARC-Field"}

ABLATIONS = {
    "ARC (full)": {},
    "- host cap": {"use_quota": False},
    "- trimming (plain quantile)": {"warm": "quantile"},
    "tail transfer instead of trimming": {"warm": "tail"},
    "mean/SD instead of median/MAD": {"robust": False},
    "- restandardisation": {"restandardise": False},
    "+ periodic recalibration (20k)": {"recal_every": 20000},
    "+ continuous RM tracking": {"eta_tau": 0.0005},
    "+ streaming median/MAD tracking": {"eta_loc": 0.0002, "eta_scale": 0.0002},
}
SENS = {"trim_c": [1.0, 1.25, 1.5, 2.0, 3.0], "burnin": [1000, 2000, 5000, 10000],
        "alpha": [0.01, 0.02, 0.05, 0.1], "quota": [0.05, 0.1, 0.2, 0.5, 1.0]}

OUT = os.path.join(RESULTS_ABL, f"calibration_{args.protocol}.json")
if os.path.exists(OUT):
    print("already done:", OUT)
    raise SystemExit(0)
out = {"ablation": {}, "sensitivity": {}}
for seed in args.seeds:
    P = PROT[args.protocol](corpus, seed=seed)
    y = ~corpus["is_normal"][P.test]
    host = corpus["src"][P.test]
    for meth in args.methods:
        f = os.path.join(sdir, f"{meth}_s{seed}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        mat = meth in MATRIX
        kw0 = {"kappa": 6}

        def run(extra):
            cfg = cal.CalConfig(**{**kw0, **extra})
            if mat:
                o = cal.run_policy("arc", cfg, R_val=d["val"].astype(np.float32),
                                   R_cal=d["cal"].astype(np.float32),
                                   R_test=d["test"].astype(np.float32), host=host)
            else:
                o = cal.run_policy("arc", cfg, s_val=d["val"], s_cal=d["cal"], s_test=d["test"], host=host)
            yy = y[o["eval_index"]]
            m = mt.all_metrics(o["score"], o["alarm"], yy, cfg.alpha)
            return {k: m[k] for k in ("f1", "fpr", "recall", "precision", "mcc", "auroc", "auprc",
                                      "fpr_calibration_error")}

        for name, extra in ABLATIONS.items():
            if not mat and name in ("mean/SD instead of median/MAD", "- restandardisation",
                                    "+ streaming median/MAD tracking"):
                continue
            out["ablation"].setdefault(meth, {}).setdefault(name, []).append(run(extra))
        for par, vals in SENS.items():
            for v in vals:
                out["sensitivity"].setdefault(meth, {}).setdefault(par, {}).setdefault(str(v), []).append(
                    run({par: v}))
        print(f"s{seed} {meth} done", flush=True)

save_json(out, OUT)
