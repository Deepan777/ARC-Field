"""Poisoning of label-free recalibration (Protocol II, cached detector outputs).

Threat model: during the burn-in window the adversary controls one source
endpoint (or, in the 'distributed' variant, 20 endpoints) and injects records.
The recalibration step cannot distinguish them from deployment traffic.
Injected records are never evaluated; detection is measured on the genuine
records that follow the burn-in.

Injection types, expressed in each detector's output space
  replay   copies of genuine attack records (the adversary replays its traffic)
  chaff    log-space interpolation between a benign and an attack record,
           lambda ~ U(0,1): records spread over the band near the threshold
  blind    one record per injection whose residual is extreme (x1e6) on the
           kappa features that best separate attacks, benign elsewhere
           (feature blinding; for scalar detectors an extreme score)
Output: results/raw/poisoning/poisoning_results.json
"""
import argparse
import json
import os

import numpy as np

from common import RESULTS_RAW, cal, mt, save_json
import data_unsw as du

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--methods", nargs="*", default=["ARC-Field", "MemAE", "Dense_Autoencoder", "KitNET", "DIF"])
args = ap.parse_args()
corpus = du.load_corpus()
sdir = os.path.join(RESULTS_RAW, "scores", "unsw_chrono_A")
MATRIX = {"ARC-Field"}
B = cal.CalConfig().burnin
RATES = [0.0, 0.002, 0.01, 0.02, 0.05, 0.1, 0.2]
BLIND_N = [1, 2, 5, 10, 20, 50]
NEW_HOST = 10_000_000

OUT = os.path.join(RESULTS_RAW, "poisoning", "poisoning_results.json")
out = {"config": {"burnin": B, "rates": RATES, "blind_n": BLIND_N}, "results": [], "done": []}
if os.path.exists(OUT):
    out = json.load(open(OUT))
    out.setdefault("done", [])
for seed in args.seeds:
    P = du.chronological(corpus, seed=seed)
    y = ~corpus["is_normal"][P.test]
    host = corpus["src"][P.test]
    rng = np.random.default_rng(1000 + seed)
    later = np.arange(len(y)) >= len(y) // 2
    atk_pool = np.where(y & later)[0]
    ben_pool = np.where(~y & later)[0]
    for meth in args.methods:
        f = os.path.join(sdir, f"{meth}_s{seed}.npz")
        if not os.path.exists(f) or f"{meth}_s{seed}" in out["done"]:
            continue
        d = np.load(f)
        mat = meth in MATRIX
        V = d["test"].astype(np.float32) if mat else d["test"].astype(np.float64)
        kw0 = {"kappa": 6}
        # features that best separate attacks (what an informed adversary blinds)
        if mat:
            Lv = np.log(d["val"].astype(np.float64) + 1e-6)
            mu, sd = Lv.mean(0), Lv.std(0) + 1e-9
            sep = ((np.log(V[atk_pool[:5000]] + 1e-6) - mu) / sd).mean(0)
            top = np.argsort(-sep)[:6]

        def make(kind, n):
            a = rng.choice(atk_pool, n)
            if kind == "replay":
                return V[a].copy()
            if kind == "chaff":
                b = rng.choice(ben_pool, n)
                lam = rng.uniform(0, 1, n)
                if mat:
                    lam = lam[:, None]
                    return np.exp((1 - lam) * np.log(V[b] + 1e-6) + lam * np.log(V[a] + 1e-6)).astype(np.float32)
                return (1 - lam) * V[b] + lam * V[a]
            if kind == "blind":
                b = rng.choice(ben_pool, n)
                if mat:
                    Z = V[b].copy()
                    Z[:, top] = V.max() * 1e6
                    return Z
                return np.full(n, np.abs(V).max() * 1e6)
            raise ValueError(kind)

        settings = [("replay", r, "single") for r in RATES] + \
                   [("chaff", r, "single") for r in RATES[1:]] + \
                   [("replay", r, "distributed") for r in (0.1, 0.2)] + \
                   [("chaff", r, "distributed") for r in (0.1, 0.2)] + \
                   [("blind", nb / B, "single") for nb in BLIND_N]
        for kind, rate, spread in settings:
            n = int(round(rate * B))
            inj = None
            if n > 0:
                Xi = make(kind, n)
                hosts_i = (np.full(n, NEW_HOST) if spread == "single"
                           else NEW_HOST + rng.integers(0, 20, n))
                inj = {("R" if mat else "s"): Xi, "host": hosts_i,
                       "pos": np.sort(rng.integers(0, B, n))}
            pols = ["burnin_full", "burnin_thr", "arc"] if mat else ["burnin_thr", "arc"]
            for pol in pols:
                kw = dict(kw0)
                if pol == "burnin_full":
                    kw["scoring"] = "raw_meansd"
                cfg = cal.CalConfig(**kw)
                if mat:
                    o = cal.run_policy(pol, cfg, R_val=d["val"].astype(np.float32),
                                       R_cal=d["cal"].astype(np.float32), R_test=V, host=host, inject=inj)
                else:
                    o = cal.run_policy(pol, cfg, s_val=d["val"], s_cal=d["cal"], s_test=V, host=host,
                                       inject=inj)
                yy = y[o["eval_index"]]
                m = mt.all_metrics(o["score"], o["alarm"], yy, cfg.alpha)
                out["results"].append({"seed": seed, "method": meth, "kind": kind, "rate": rate,
                                       "n_injected": n, "spread": spread, "policy": pol,
                                       **{k: m[k] for k in ("f1", "recall", "fpr", "precision", "mcc",
                                                            "auroc", "auprc")}})
        out["done"].append(f"{meth}_s{seed}")
        save_json(out, OUT)
        print(f"s{seed} {meth} done", flush=True)
