"""Two analyses on cached Protocol II outputs (no refitting).

1. Score stationarity ("calibratability"). For each detector and seed, the
   benign test scores are split into half-hour bins; for each bin we compute
   the Kolmogorov-Smirnov distance between its benign-score distribution and
   that of the benign records of the recalibration window. The mean distance
   is related to the realised FPR dispersion under the ARC policy.
2. Attribution for ARC-Field. Using the per-feature standardised residuals of
   the ARC policy's statistics (median/MAD of the capped window), the share of
   each observability group in the top-kappa score is averaged per attack
   family and for benign false alarms. The correlation between the per-bin
   benign FPR under the static threshold and the per-bin attack share tests
   whether attack bursts raise false alarms on concurrent benign records.
Output: results/processed/stationarity_attribution.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import RESULTS_PROC, RESULTS_RAW  # noqa: E402
import calibration as cal  # noqa: E402
import data_unsw as du  # noqa: E402

corpus = du.load_corpus()
sdir = os.path.join(RESULTS_RAW, "scores", "unsw_chrono_A")
B = cal.CalConfig().burnin
feats = du.FEATURE_SETS["A"]
groups = du.groups_for(feats)
gnames = du.group_names_for(feats)
out = {"stationarity": {}, "attribution": {}, "collateral": {}}

for f in sorted(glob.glob(os.path.join(sdir, "*_s*.npz"))):
    base = os.path.basename(f)[:-4]
    meth, seed = base.rsplit("_s", 1)
    seed = int(seed)
    P = du.chronological(corpus, seed=seed)
    y = ~corpus["is_normal"][P.test]
    t = corpus["ltime"][P.test].astype(float)
    d = np.load(f)
    mat = d["test"].ndim == 2
    cfg = cal.CalConfig(kappa=6, scoring="log_robust")
    kw = dict(R_val=d["val"].astype(np.float32), R_cal=d["cal"].astype(np.float32),
              R_test=d["test"].astype(np.float32)) if mat else dict(s_val=d["val"], s_cal=d["cal"], s_test=d["test"])
    o = cal.run_policy("arc", cfg, host=corpus["src"][P.test], **kw)
    sc = np.full(len(y), np.nan)
    sc[o["eval_index"]] = o["score"]
    # reference: benign scores of the first half hour after the recalibration
    # window, under the same recalibrated statistics and threshold
    bins = ((t - t.min()) // 1800).astype(int)
    valid = ~np.isnan(sc)
    first = bins[valid].min()
    ref = sc[valid & ~y & (bins == first)]
    ks = []
    for b in np.unique(bins[valid]):
        m = valid & ~y & (bins == b)
        if m.sum() >= 200 and b != first:
            ks.append(stats.ks_2samp(sc[m], ref).statistic)
    fb = [(o["alarm"][(~y[o["eval_index"]]) & (bins[o["eval_index"]] == b)]).mean()
          for b in np.unique(bins[o["eval_index"]])
          if ((~y[o["eval_index"]]) & (bins[o["eval_index"]] == b)).sum() >= 200]
    out["stationarity"].setdefault(meth, []).append(
        {"seed": seed, "mean_ks": float(np.mean(ks)), "fpr_bin_sd": float(np.std(fb)),
         "fpr_bin_max": float(np.max(fb))})

    if meth == "ARC-Field":
        # attribution with the recalibrated robust statistics
        L = np.log(d["test"].astype(np.float64) + 1e-6)
        Lw = L[:B]
        med = np.median(Lw, 0)
        mad = np.median(np.abs(Lw - med), 0) * 1.4826 + 1e-3
        Z = (L - med) / mad
        k = 6
        top = np.argsort(-Z, 1)[:, :k]
        share = np.zeros((len(Z), len(groups)))
        for gi, g in enumerate(groups):
            share[:, gi] = np.isin(top, g).sum(1) / k
        lab = corpus["label"][P.test]
        ei = o["eval_index"]
        alarm = np.zeros(len(y), bool)
        alarm[ei] = o["alarm"]
        att = {}
        for fam in sorted(set(lab[y])):
            m = (lab == fam) & alarm
            if m.sum() >= 20:
                att[fam] = dict(zip(gnames, share[m].mean(0).round(4).tolist()))
        fa = ~y & alarm
        att["benign false alarms"] = dict(zip(gnames, share[fa].mean(0).round(4).tolist()))
        out["attribution"][f"s{seed}"] = att
        # collateral: static-threshold benign FPR per bin vs attack share per bin
        o_st = cal.run_policy("static", cal.CalConfig(kappa=6), host=corpus["src"][P.test], **kw)
        a_st = np.zeros(len(y), bool)
        a_st[o_st["eval_index"]] = o_st["alarm"]
        rows = []
        for b in np.unique(bins):
            m = bins == b
            if (~y & m).sum() >= 200:
                rows.append((y[m].mean(), a_st[~y & m].mean(), alarm[~y & m].mean()))
        rows = np.array(rows)
        r_st = stats.spearmanr(rows[:, 0], rows[:, 1])
        r_ar = stats.spearmanr(rows[:, 0], rows[:, 2])
        out["collateral"][f"s{seed}"] = {"spearman_static": [float(r_st.correlation), float(r_st.pvalue)],
                                         "spearman_arc": [float(r_ar.correlation), float(r_ar.pvalue)],
                                         "bins": rows.round(4).tolist()}
    print(meth, seed, "done", flush=True)

summ = {m: {k: float(np.mean([r[k] for r in v])) for k in ("mean_ks", "fpr_bin_sd", "fpr_bin_max")}
        for m, v in out["stationarity"].items()}
xs = [summ[m]["mean_ks"] for m in summ]
ys = [summ[m]["fpr_bin_sd"] for m in summ]
if len(xs) >= 4:
    rho = stats.spearmanr(xs, ys)
    out["stationarity_vs_fpr_dispersion"] = {"spearman": float(rho.correlation), "p": float(rho.pvalue),
                                             "n_detectors": len(xs)}
out["stationarity_summary"] = summ
json.dump(out, open(os.path.join(RESULTS_PROC, "stationarity_attribution.json"), "w"), indent=1)
print(json.dumps(out.get("stationarity_vs_fpr_dispersion"), indent=1))
