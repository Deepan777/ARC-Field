"""Synthetic end-to-end smoke test of the release (no datasets needed, CPU,
under a minute): causal context, ARC-Field fit/residuals, every calibration
policy, metrics, the GraphINR-IDS spectral comparator and one standard baseline."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "third_party"))

import arcfield as af  # noqa: E402
import baselines as B0  # noqa: E402
import calibration as cal  # noqa: E402
import metrics as M  # noqa: E402
import graphinr_ids_harness as gh  # noqa: E402

rng = np.random.default_rng(0)
n, d, dc = 12000, 12, 29
X = rng.normal(size=(n, d)).astype(np.float32)
X[:, 6:] = X[:, :6] * 0.8 + 0.2 * X[:, 6:]
C = rng.uniform(-1, 1, size=(n, dc)).astype(np.float32)
y = np.zeros(n, dtype=bool)
y[8000::7] = True
X[y] += 2.5
host = rng.integers(0, 40, size=n)
groups = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9)), list(range(9, 12))]

tr, st = np.arange(0, 5000), np.arange(5000, n)
model = af.ARCField(d, dc, hidden=64, layers=3, omega_0=30.0, hetero=False)
af.train_field(model, X[tr], C[tr], groups, epochs=2, lr=2e-4, batch=512, seed=0, device="cpu")
R_val = af.residuals(model, X[3000:4000], C[3000:4000], groups, device="cpu")
R_cal = af.residuals(model, X[4000:5000], C[4000:5000], groups, device="cpu")
R_str = af.residuals(model, X[st], C[st], groups, device="cpu")
print("params", af.count_params(model), "residuals", R_str.shape)

cfg = cal.CalConfig(burnin=2000)
for pol in ("static", "burnin_thr", "burnin_full", "ema", "arc"):
    out = cal.run_policy(pol, cfg, R_val=R_val, R_cal=R_cal, R_test=R_str,
                         host=host[st], benign_mask=~y[st])
    m = M.all_metrics(out["score"], out["alarm"], y[st][out["eval_index"]], cfg.alpha)
    print(f"{pol:12s} AUROC {m['auroc']:.3f} FPR {m['fpr']:.3f}")

ips = np.array([f"10.0.0.{i}" for i in range(40)])
g = gh.RevGraph(spectral_k=4).fit(ips[host[tr]], ips[(host[tr] + 1) % 40], verbose=False)
print("spectral comparator fitted")

score_fn = B0.BASELINES["Isolation Forest"](X[tr], 0)
print("Isolation Forest AUROC %.3f" % M.ranking_metrics(score_fn(X[st]), y[st])["auroc"])
print("SMOKE TEST PASSED")
