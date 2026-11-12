"""Post hoc diagnostic of the trimmed-quantile assumption (no refitting).

The ARC threshold trims the recalibration window at g = med + c k MAD and
takes the quantile of the kept scores at level (1 - alpha) / (1 - beta), where
beta is the share of *calibration-period* benign scores beyond the analogous
trim point. This transfers the shape of the benign tail, in median/MAD units,
from the calibration period to the window. If the window's benign tail is
heavier, more than beta of its benign scores lie beyond g and the threshold is
too low, whatever the contamination.

For every recalibration window of Protocol II (UNSW-NB15) and Protocol V
(NF-CSE-CIC-IDS2018-v3) this script reports, for ARC-Field:
  beta_cal      clean trimmed share assumed by the rule (label-free),
  beta_win      trimmed share observed in the window (label-free),
  beta_win_ben  trimmed share among the window's benign records (labels; analysis only),
and the realised FPR/recall of the frozen rule, of larger trim constants and of
the untrimmed quantile (the '- trimming' ablation).
Output: results/processed/tail_transfer.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import RESULTS_PROC, RESULTS_RAW  # noqa: E402
import calibration as cal  # noqa: E402

VARIANTS = {"arc_c1.5": {}, "arc_c2": {"trim_c": 2.0}, "arc_c3": {"trim_c": 3.0},
            "arc_untrimmed": {"warm": "quantile"}}


def diagnose(R_val, R_cal, R_test, host, y, kappa):
    cfg = cal.CalConfig(kappa=kappa)
    tf = cal._log_u
    loc, scl = cal._loc_scale(tf(R_val), True)
    sc = lambda R, lo, sl: cal.topk_mean((tf(R) - lo) / sl, kappa)  # noqa: E731
    T = cal.ScoreTransform.fit(sc(R_cal, loc, scl))
    l_cal = T(sc(R_cal, loc, scl))
    tau = cal._quantile(l_cal, 1 - cfg.alpha)
    med_c = float(np.median(l_cal))
    mad_c = float(np.median(np.abs(l_cal - med_c)) * 1.4826) or 1e-6
    k_tail = (tau - med_c) / mad_c
    beta_cal = float(np.mean(l_cal > med_c + cfg.trim_c * k_tail * mad_c))
    B = cfg.burnin
    cap = max(1, int(cfg.quota * B))
    seen, keep = {}, np.zeros(B, bool)
    for j, h in enumerate(host[:B]):
        seen[int(h)] = seen.get(int(h), 0) + 1
        keep[j] = seen[int(h)] <= cap
    Xw, yw = R_test[:B][keep], y[:B][keep]
    lo_w, sl_w = cal._loc_scale(tf(Xw), True)
    lw = T(sc(Xw, lo_w, sl_w))
    med_b = float(np.median(lw))
    mad_b = float(np.median(np.abs(lw - med_b)) * 1.4826) or 1e-6
    g = med_b + cfg.trim_c * k_tail * mad_b
    out = {"beta_cal": beta_cal, "beta_win": float(np.mean(lw > g)),
           "beta_win_ben": float(np.mean(lw[~yw] > g)) if (~yw).any() else None,
           "window_attack_share": float(yw.mean()), "level": (1 - cfg.alpha) / (1 - beta_cal)}
    for name, kw in VARIANTS.items():
        o = cal.run_policy("arc", cal.CalConfig(kappa=kappa, **kw), R_val=R_val, R_cal=R_cal,
                           R_test=R_test, host=host, benign_mask=~y)
        ye = y[o["eval_index"]]
        out[name] = {"fpr": float(o["alarm"][~ye].mean()),
                     "recall": float(o["alarm"][ye].mean()) if ye.any() else None}
    o = cal.run_policy("burnin_thr", cal.CalConfig(kappa=kappa), R_val=R_val, R_cal=R_cal, R_test=R_test,
                       host=host, benign_mask=~y)
    ye = y[o["eval_index"]]
    out["plain_quantile"] = {"fpr": float(o["alarm"][~ye].mean()),
                             "recall": float(o["alarm"][ye].mean()) if ye.any() else None}
    return out


res = {}
# ---- Protocol V -----------------------------------------------------------
import data_nf as dn  # noqa: E402
nf = dn.load_corpus(verbose=False)
kappa_nf = max(1, round(len(dn.FEATURE_SETS["A"]) / 4))
for f in sorted(glob.glob(os.path.join(RESULTS_RAW, "scores", "nf_crossday_A", "ARC-Field_s*.npz"))):
    seed = int(f.rsplit("_s", 1)[1][:-4])
    d = np.load(f)
    _, _, _, tests = dn.cross_day(nf, seed)
    for day, ix in tests.items():
        r = diagnose(d["val"].astype(np.float32), d["cal"].astype(np.float32), d[f"day{day}"].astype(np.float32),
                     nf["src"][ix], ~nf["is_normal"][ix], kappa_nf)
        res.setdefault("nf_crossday_A", []).append({"seed": seed, "window": f"day{day}", **r})
    print("Protocol V seed", seed, "done", flush=True)
del nf

# ---- Protocol II ----------------------------------------------------------
import data_unsw as du  # noqa: E402
un = du.load_corpus(verbose=False)
for f in sorted(glob.glob(os.path.join(RESULTS_RAW, "scores", "unsw_chrono_A", "ARC-Field_s*.npz"))):
    seed = int(f.rsplit("_s", 1)[1][:-4])
    d = np.load(f)
    P = du.chronological(un, seed=seed)
    r = diagnose(d["val"].astype(np.float32), d["cal"].astype(np.float32), d["test"].astype(np.float32),
                 un["src"][P.test], ~un["is_normal"][P.test], 6)
    res.setdefault("unsw_chrono_A", []).append({"seed": seed, "window": "campaign2", **r})
    print("Protocol II seed", seed, "done", flush=True)

summary = {}
for ds, rows in res.items():
    s = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
         for k in ("beta_cal", "beta_win", "beta_win_ben", "window_attack_share")}
    for v in list(VARIANTS) + ["plain_quantile"]:
        s[v + "_fpr"] = float(np.mean([r[v]["fpr"] for r in rows]))
        rec = [r[v]["recall"] for r in rows if r[v]["recall"] is not None]
        s[v + "_recall"] = float(np.mean(rec)) if rec else None
    summary[ds] = s
    print(ds, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}))
os.makedirs(RESULTS_PROC, exist_ok=True)
with open(os.path.join(RESULTS_PROC, "tail_transfer.json"), "w") as fh:
    json.dump({"rows": res, "summary": summary}, fh, indent=1)
