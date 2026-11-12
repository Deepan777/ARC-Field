"""Per-family detection at a matched benign false-positive rate.

Operating points produced by different detectors have different realised FPRs,
which confounds per-family comparisons. Here every detector's threshold is set
on its own benign test scores to a common FPR (1 % and 0.1 %), so the per-family
detection rates compare ranking ability family by family. Scores are those of
the static policy (reference-period statistics).
Output: results/processed/family_matched_fpr.json
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
import data_unsw as du  # noqa: E402

corpus = du.load_corpus(verbose=False)
PROT = {"chrono": du.chronological, "hostdisjoint": du.host_disjoint}
out = {}
for prot, fn in PROT.items():
    sdir = os.path.join(RESULTS_RAW, "scores", f"unsw_{prot}_A")
    for f in sorted(glob.glob(os.path.join(sdir, "*_s*.npz"))):
        meth, seed = os.path.basename(f)[:-4].rsplit("_s", 1)
        if "+ctx" in meth:
            continue
        seed = int(seed)
        P = fn(corpus, seed=seed)
        y = ~corpus["is_normal"][P.test]
        lab = corpus["label"][P.test]
        d = np.load(f)
        if d["test"].ndim == 2:
            cfg = cal.CalConfig(kappa=6, scoring="log_robust")
            o = cal.run_policy("static", cfg, R_val=d["val"].astype(np.float32), R_cal=d["cal"].astype(np.float32),
                               R_test=d["test"].astype(np.float32))
        else:
            o = cal.run_policy("static", cal.CalConfig(), s_val=d["val"], s_cal=d["cal"], s_test=d["test"])
        s, yy, ll = o["score"], y[o["eval_index"]], lab[o["eval_index"]]
        for fpr in (0.01, 0.001):
            thr = np.quantile(s[~yy], 1 - fpr)
            row = out.setdefault(prot, {}).setdefault(f"{fpr}", {}).setdefault(meth, {})
            for fam in sorted(set(ll[yy])):
                m = ll == fam
                row.setdefault(fam, []).append(float((s[m] > thr).mean()))
            row.setdefault("ALL", []).append(float((s[yy] > thr).mean()))
        print(prot, meth, seed, flush=True)
# share of each family among the attack records of the Protocol II stream (seed 0)
P = du.chronological(corpus, seed=0)
lab = corpus["label"][P.test]
att = lab[lab != "Normal"]
out["family_share_chrono"] = {f: float((att == f).mean()) for f in sorted(set(att))}
json.dump(out, open(os.path.join(RESULTS_PROC, "family_matched_fpr.json"), "w"), indent=1)
