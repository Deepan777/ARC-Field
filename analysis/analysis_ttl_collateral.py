"""Why does recalibration fail once the TTL features are removed?

Protocol II, seed 0, ARC-Field. Tier A residuals come from the cached main
run; the no-TTL field is refitted here with the frozen hyper-parameters (its
residuals are cached in results/raw/scores/diag/ and the reported feature-set
results are not touched). For both feature sets:

1. the per-half-hour benign FPR under the ARC policy against the attack share
   of the half hour (Spearman), i.e. collateral false alarms;
2. for benign records, the per-feature log-residual standardised by the
   median/MAD of the first B stream records, averaged separately over
   high-attack (> 25 %) and low-attack (< 10 %) half hours; the features whose
   benign residuals rise most during attack bursts are the carriers of the
   collateral effect;
3. the share of each observability group among the top-kappa features of
   benign false alarms in high-attack half hours.
Output: results/processed/ttl_collateral.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd  # before torch: pyarrow after torch crashes on Windows
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
from common import fit_arcfield, device, RobustScaler, log_mask_for, save_json  # noqa: E402
import arcfield as af  # noqa: E402
import calibration as cal  # noqa: E402
import data_unsw as du  # noqa: E402
from paths import RESULTS_PROC, RESULTS_RAW  # noqa: E402

SEED = 0
corpus = du.load_corpus()
P = du.chronological(corpus, seed=SEED)
y = ~corpus["is_normal"][P.test]
t = corpus["ltime"][P.test].astype(float)
host = corpus["src"][P.test]
bins = ((t - t.min()) // 1800).astype(int)
B = cal.CalConfig().burnin
diag_dir = os.path.join(RESULTS_RAW, "scores", "diag")
os.makedirs(diag_dir, exist_ok=True)
out = {}

for fs in ("A", "A_noTTL"):
    feats = du.FEATURE_SETS[fs]
    groups = du.groups_for(feats)
    gnames = du.group_names_for(feats)
    kappa = max(1, round(len(feats) / 4))
    if fs == "A":
        d = np.load(os.path.join(RESULTS_RAW, "scores", "unsw_chrono_A", f"ARC-Field_s{SEED}.npz"))
        base = {k: d[k].astype(np.float32) for k in ("val", "cal", "test")}
    else:
        cache = os.path.join(diag_dir, f"unsw_chrono_{fs}_ARC-Field_s{SEED}.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            base = {k: d[k].astype(np.float32) for k in ("val", "cal", "test")}
        else:
            fi = [du.ALL36.index(f) for f in feats]
            Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(corpus["X"][P.train][:, fi])
            Cs = RobustScaler(None).fit(corpus["C"][P.train])
            M = {k: Xs(corpus["X"][getattr(P, k)][:, fi]) for k in ("train", "val", "cal", "test")}
            C = {k: Cs(corpus["C"][getattr(P, k)]) for k in ("train", "val", "cal", "test")}
            model, _ = fit_arcfield(M["train"], C["train"], groups, SEED, device())
            base = {k: af.residuals(model, M[k], C[k], groups, device()) for k in ("val", "cal", "test")}
            np.savez_compressed(cache, **{k: v.astype(np.float16) for k, v in base.items()})

    o = cal.run_policy("arc", cal.CalConfig(kappa=kappa), R_val=base["val"], R_cal=base["cal"],
                       R_test=base["test"], host=host, benign_mask=~y)
    ei = o["eval_index"]
    alarm = np.zeros(len(y), bool)
    alarm[ei] = o["alarm"]
    evald = np.zeros(len(y), bool)
    evald[ei] = True

    # 1. collateral: per-bin benign FPR vs attack share
    rows = []
    for b in np.unique(bins[evald]):
        m = evald & (bins == b)
        nb = (m & ~y).sum()
        if nb < 500:
            continue
        rows.append((b, float(y[m].mean()), float(alarm[m & ~y].mean())))
    share = np.array([r[1] for r in rows])
    fpr = np.array([r[2] for r in rows])
    rho, p = stats.spearmanr(share, fpr)

    # 2. per-feature benign residual shift, high- vs low-attack bins
    L = np.log(np.maximum(base["test"].astype(np.float64), 0) + 1e-6)
    ref = L[:B]
    med = np.median(ref, 0)
    mad = np.median(np.abs(ref - med), 0) * 1.4826 + 1e-9
    Z = (L - med) / mad
    hi_bins = {r[0] for r in rows if r[1] > 0.25}
    lo_bins = {r[0] for r in rows if r[1] < 0.10}
    hi = evald & ~y & np.isin(bins, list(hi_bins))
    lo = evald & ~y & np.isin(bins, list(lo_bins))
    shift = Z[hi].mean(0) - Z[lo].mean(0)
    order = np.argsort(-shift)
    feat_shift = [{"feature": feats[j], "group": gnames[[j in g for g in groups].index(True)],
                   "benign_z_high": float(Z[hi, j].mean()), "benign_z_low": float(Z[lo, j].mean()),
                   "shift": float(shift[j])} for j in order]

    # 3. group attribution of benign false alarms in high-attack bins
    fa = hi & alarm
    top = np.argsort(-Z[fa], axis=1)[:, :kappa]
    gid = np.zeros(len(feats), int)
    for gi, g in enumerate(groups):
        gid[g] = gi
    counts = np.bincount(gid[top].ravel(), minlength=len(groups)) / max(top.size, 1)

    out[fs] = {"kappa": kappa, "arc_fpr": float(alarm[evald & ~y].mean()),
               "bins": [{"bin": int(b), "attack_share": s, "benign_fpr": f} for b, s, f in rows],
               "collateral_spearman": [float(rho), float(p)],
               "fpr_high_attack_bins": float(alarm[hi].mean()), "fpr_low_attack_bins": float(alarm[lo].mean()),
               "n_high_bins": len(hi_bins), "n_low_bins": len(lo_bins),
               "feature_shift": feat_shift,
               "false_alarm_group_share_high_bins": dict(zip(gnames, map(float, counts)))}
    print(f"{fs}: arc FPR {out[fs]['arc_fpr']:.4f}  collateral rho={rho:.2f} (p={p:.1e})  "
          f"FPR high/low-attack bins {out[fs]['fpr_high_attack_bins']:.4f}/{out[fs]['fpr_low_attack_bins']:.4f}")
    for r in feat_shift[:6]:
        print(f"   {r['feature']:18s} {r['group']:10s} shift {r['shift']:+.2f} (high {r['benign_z_high']:+.2f}, "
              f"low {r['benign_z_low']:+.2f})")
    print("   false-alarm group share (high bins):", out[fs]["false_alarm_group_share_high_bins"])

# 4. how nearly the source TTL alone separates the classes (full corpus)

from paths import DATA_DIR  # noqa: E402
df = pd.read_parquet(os.path.join(DATA_DIR, "unsw_full.parquet"), columns=["sttl", "Label"])
att = df["Label"].to_numpy() == 1
top = df["sttl"][att].value_counts().index[:2].tolist()
out["ttl_separation"] = {"attack_top_sttl": [int(v) for v in top],
                         "attack_share_in_top": float(df["sttl"][att].isin(top).mean()),
                         "benign_share_in_top": float(df["sttl"][~att].isin(top).mean())}
print("TTL separation:", out["ttl_separation"])

save_json(out, os.path.join(RESULTS_PROC, "ttl_collateral.json"))
