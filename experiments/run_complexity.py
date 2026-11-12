"""Computational cost of ARC-Field.

Measures (i) parameters and analytic multiply-accumulates per record,
(ii) scoring throughput on GPU and CPU, (iii) throughput of the role
coordinates (vectorised batch and flow-at-a-time streaming reference, with the
peak streaming state size), (iv) cost of one robust recalibration.
Output: results/raw/complexity/complexity.json
"""
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch

from common import RESULTS_RAW, af, cal, save_json, RobustScaler, log_mask_for, ARC_HP
import context as cx
import data_unsw as du

if os.path.exists(os.path.join(RESULTS_RAW, "complexity", "complexity.json")):
    print("already done")
    raise SystemExit(0)
corpus = du.load_corpus()
feats = du.FEATURE_SETS["A"]
fi = [du.ALL36.index(f) for f in feats]
groups = du.groups_for(feats)
d, dc, h, L, K = len(feats), corpus["C"].shape[1], ARC_HP["hidden"], ARC_HP["layers"], len(groups)
out = {}

# ---- ARC-Field -------------------------------------------------------------
model = af.ARCField(d, dc, hidden=h, layers=L, hetero=False)
n_in = dc + 2 * d
macs = K * (n_in * h + (L - 1) * h * h + h * d)
out["arcfield"] = {"params": af.count_params(model), "macs_per_record": int(macs),
                   "forward_passes_per_record": K}
X = np.random.randn(100000, d).astype(np.float32)
C = np.random.randn(100000, dc).astype(np.float32)
for dev in ("cuda", "cpu"):
    if dev == "cuda" and not torch.cuda.is_available():
        continue
    torch.set_num_threads(8)
    af.residuals(model, X[:8192], C[:8192], groups, dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    af.residuals(model, X, C, groups, dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    out["arcfield"][f"records_per_s_{dev}"] = 100000 / (time.time() - t0)

# ---- role coordinates -------------------------------------------------------
col = {n: i for i, n in enumerate(du.ALL36)}
n = len(corpus["src"])
t0 = time.time()
cx.unsw_context(src=corpus["src"], dst=corpus["dst"], dport=corpus["dport"], t=corpus["ltime"].astype(float),
                sbytes=corpus["X"][:, col["sbytes"]].astype(float), dbytes=corpus["X"][:, col["dbytes"]].astype(float),
                dur=corpus["X"][:, col["dur"]].astype(float), proto=corpus["proto"])
out["context"] = {"vectorised_records_per_s": n / (time.time() - t0), "n_records": n}
# streaming reference: per-endpoint deques for the two windows, both roles
N = 200000
src, dst, t = corpus["src"][:N], corpus["dst"][:N], corpus["ltime"][:N].astype(float)
qs = {W: (defaultdict(deque), defaultdict(deque)) for W in (10.0, 300.0)}
peak = 0
t0 = time.time()
for i in range(N):
    tot = 0
    for W, (qo, qi) in qs.items():
        for q in (qo[src[i]], qi[dst[i]]):
            while q and q[0] <= t[i] - W:
                q.popleft()
        qo[src[i]].append(t[i])
        qi[dst[i]].append(t[i])
    if i % 5000 == 0:
        peak = max(peak, sum(len(q) for W in qs for m in qs[W] for q in m.values()))
out["context"]["streaming_records_per_s_python"] = N / (time.time() - t0)
out["context"]["streaming_peak_state_entries"] = int(peak)

# ---- recalibration cost -----------------------------------------------------
R = np.abs(np.random.randn(20000, d)).astype(np.float32) ** 2
t0 = time.time()
for _ in range(10):
    cal.run_policy("arc", cal.CalConfig(), R_val=R[:5000], R_cal=R[5000:10000], R_test=R,
                   host=np.random.randint(0, 50, len(R)))
out["calibration"] = {"arc_policy_seconds_per_20k_stream": (time.time() - t0) / 10}
save_json(out, os.path.join(RESULTS_RAW, "complexity", "complexity.json"))
print(out)
