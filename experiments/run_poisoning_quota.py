"""Supplementary: does a tighter per-host cap limit single-host poisoning?

The frozen configuration uses q = 0.2, chosen a priori so that no benign host
of the testbed is capped; RQ4 shows it never binds at the tested injection
shares. This study re-runs replay and chaff injections into the Protocol II
recalibration window for ARC-Field with q in {0.02, 0.05, 0.1, 0.2}, from one
host and from 20 hosts. It is reported as a supplementary sensitivity analysis;
the frozen configuration is not changed.
Output: results/raw/poisoning/poisoning_quota.json
"""
import json
import os

import numpy as np

from common import RESULTS_RAW, cal, mt, save_json
import data_unsw as du

OUT = os.path.join(RESULTS_RAW, "poisoning", "poisoning_quota.json")
if os.path.exists(OUT):
    print("already done")
    raise SystemExit(0)
corpus = du.load_corpus()
sdir = os.path.join(RESULTS_RAW, "scores", "unsw_chrono_A")
B = cal.CalConfig().burnin
NEW_HOST = 10_000_000
res = []
for seed in range(5):
    P = du.chronological(corpus, seed=seed)
    y = ~corpus["is_normal"][P.test]
    host = corpus["src"][P.test]
    d = np.load(os.path.join(sdir, f"ARC-Field_s{seed}.npz"))
    V = d["test"].astype(np.float32)
    rng = np.random.default_rng(2000 + seed)
    later = np.arange(len(y)) >= len(y) // 2
    atk, ben = np.where(y & later)[0], np.where(~y & later)[0]
    for kind in ("replay", "chaff"):
        for rate in (0.05, 0.1, 0.2):
            n = int(rate * B)
            a = rng.choice(atk, n)
            if kind == "replay":
                Xi = V[a].copy()
            else:
                b = rng.choice(ben, n)
                lam = rng.uniform(0, 1, n)[:, None]
                Xi = np.exp((1 - lam) * np.log(V[b] + 1e-6) + lam * np.log(V[a] + 1e-6)).astype(np.float32)
            pos = np.sort(rng.integers(0, B, n))
            for spread in ("single", "distributed"):
                hi = np.full(n, NEW_HOST) if spread == "single" else NEW_HOST + rng.integers(0, 20, n)
                for q in (0.02, 0.05, 0.1, 0.2):
                    cfg = cal.CalConfig(kappa=6, quota=q)
                    o = cal.run_policy("arc", cfg, R_val=d["val"].astype(np.float32), R_cal=d["cal"].astype(np.float32),
                                       R_test=V, host=host, inject={"R": Xi, "host": hi, "pos": pos})
                    m = mt.all_metrics(o["score"], o["alarm"], y[o["eval_index"]], cfg.alpha)
                    res.append({"seed": seed, "kind": kind, "rate": rate, "spread": spread, "quota": q,
                                **{k: m[k] for k in ("recall", "fpr", "mcc", "auroc")}})
    # clean window for each quota
    for q in (0.02, 0.05, 0.1, 0.2):
        cfg = cal.CalConfig(kappa=6, quota=q)
        o = cal.run_policy("arc", cfg, R_val=d["val"].astype(np.float32), R_cal=d["cal"].astype(np.float32),
                           R_test=V, host=host)
        m = mt.all_metrics(o["score"], o["alarm"], y[o["eval_index"]], cfg.alpha)
        res.append({"seed": seed, "kind": "none", "rate": 0.0, "spread": "single", "quota": q,
                    **{k: m[k] for k in ("recall", "fpr", "mcc", "auroc")}})
    print("seed", seed, "done", flush=True)
save_json({"results": res}, OUT)
