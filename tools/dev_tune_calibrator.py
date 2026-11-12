"""Select calibrator hyper-parameters on the development protocol only.

Selection rule (fixed in advance): among configurations whose realised FPR is
within [alpha/2, alpha + 0.02] on both development seeds, pick the one with the
highest mean MCC. Output: logs/dev/calibrator_selection.json
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import LOG_DIR  # noqa: E402
import calibration as cal  # noqa: E402
import metrics as mt  # noqa: E402

D = [np.load(os.path.join(LOG_DIR, "dev", f"dev_resid_s{s}.npz"), allow_pickle=True) for s in (0, 1)]


def evaluate(policy, cfg):
    out = []
    for d in D:
        o = cal.run_policy(policy, cfg, R_val=d["val"], R_cal=d["cal"], R_test=d["test"],
                           host=d["host"], benign_mask=~d["y"])
        y = d["y"][o["eval_index"]]
        out.append(mt.all_metrics(o["score"], o["alarm"], y, cfg.alpha))
    return out


rows = []
base = cal.CalConfig()
for pol in ["static", "burnin_thr", "burnin_full", "burnin_oracle", "ema"]:
    r = evaluate(pol, base)
    rows.append({"policy": pol, "cfg": {}, "mcc": np.mean([x["mcc"] for x in r]),
                 "fpr": [x["fpr"] for x in r], "recall": [x["recall"] for x in r],
                 "f1": np.mean([x["f1"] for x in r]), "auroc": np.mean([x["auroc"] for x in r])})
    print(rows[-1]["policy"], round(rows[-1]["mcc"], 4), np.round(rows[-1]["fpr"], 4),
          np.round(rows[-1]["recall"], 4), round(rows[-1]["auroc"], 4))

configs = []
for warm in ["trimmed", "quantile", "tail"]:
    for trim_c in ([1.25, 1.5, 2.0] if warm == "trimmed" else [1.5]):
        for burnin in [2000, 5000]:
            for recal in [0, 20000, 50000]:
                configs.append(dict(warm=warm, trim_c=trim_c, burnin=burnin, recal_every=recal))
# continuous gated Robbins-Monro tracking and streaming trackers, for the record
for eta_tau in [0.0005, 0.002]:
    for eta_loc in [0.0, 0.0002]:
        configs.append(dict(warm="trimmed", trim_c=1.5, burnin=5000, recal_every=0,
                            eta_tau=eta_tau, eta_loc=eta_loc, eta_scale=eta_loc))
for kw in configs:
    cfg = cal.CalConfig(**kw)
    r = evaluate("arc", cfg)
    rows.append({"policy": "arc", "cfg": kw, "mcc": np.mean([x["mcc"] for x in r]),
                 "fpr": [x["fpr"] for x in r], "recall": [x["recall"] for x in r],
                 "f1": np.mean([x["f1"] for x in r]), "auroc": np.mean([x["auroc"] for x in r])})
    print("arc", kw, round(rows[-1]["mcc"], 4), np.round(rows[-1]["fpr"], 4),
          np.round(rows[-1]["recall"], 4), round(rows[-1]["auroc"], 4), flush=True)

alpha = base.alpha
ok = [r for r in rows if r["policy"] == "arc" and all(alpha / 2 <= f <= alpha + 0.02 for f in r["fpr"])]
best = max(ok, key=lambda r: r["mcc"]) if ok else max(
    [r for r in rows if r["policy"] == "arc"], key=lambda r: r["mcc"])
out = {"selection_rule": "FPR in [alpha/2, alpha+0.02] on both dev seeds, max mean MCC",
       "selected": best, "n_feasible": len(ok), "all": rows}
with open(os.path.join(LOG_DIR, "dev", "calibrator_selection.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print("SELECTED", best["cfg"], best["mcc"], best["fpr"], best["recall"])
