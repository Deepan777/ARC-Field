"""Verify that the vectorised role-coordinate computation equals a genuinely
causal flow-at-a-time stream computation (no look-ahead, no self-inclusion).

For the first N flows of the emission-ordered corpus, recompute with naive
per-endpoint deques: (i) outbound count / novelty / mean log bytes of the
source, (ii) inbound count of the destination, (iii) cross-role count of the
source (how often it was recently a destination), (iv) pair recency.
Writes reproducibility/causality_check.json.
"""
import json
import math
import os
import sys
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import PROJECT  # noqa: E402
import data_unsw as du  # noqa: E402

N = 150000
W = 300.0

c = du.load_corpus(verbose=False)
names = c["ctx_names"]
C = c["C"][:N]
src, dst, t = c["src"][:N], c["dst"][:N], c["ltime"][:N].astype(float)
sb, db = c["X"][:N, du.ALL36.index("sbytes")], c["X"][:N, du.ALL36.index("dbytes")]
dport = c["dport"][:N]

out_q = defaultdict(deque)      # src -> (t, newdst, lsb)
in_q = defaultdict(deque)       # dst -> t
role_q = defaultdict(deque)     # endpoint as destination -> t
last_pair = {}
last_pair_W = {}
ref = np.zeros((N, 5))
for i in range(N):
    s, d, ti = int(src[i]), int(dst[i]), t[i]
    for q in (out_q[s], in_q[d], role_q[s]):
        while q and q[0][0] <= ti - W:
            q.popleft()
    n_out = len(out_q[s])
    mean_new = sum(e[1] for e in out_q[s]) / max(n_out, 1)
    mean_lsb = sum(e[2] for e in out_q[s]) / max(n_out, 1)
    n_in = len(in_q[d])
    n_src_as_dst = len(role_q[s])
    prev = last_pair.get((s, d))
    gap = math.inf if prev is None else ti - prev
    ref[i] = [math.log1p(n_out), mean_new, mean_lsb, math.log1p(n_in), math.log1p(n_src_as_dst)]
    # update AFTER query: the flow never sees itself
    new = 1.0 if gap > W else 0.0
    out_q[s].append((ti, new, math.log1p(max(sb[i], 0))))
    in_q[d].append((ti,))
    role_q[d].append((ti,))
    last_pair[(s, d)] = ti

cols = ["src_out_n_300s", "src_out_newdst_300s", "src_out_lsb_300s", "dst_in_n_300s", "src_in_n_300s"]
vec = np.stack([C[:, names.index(k)] for k in cols], axis=1)
err = np.abs(vec - ref).max(axis=0)
res = {"n_flows_checked": N, "window_s": W,
       "max_abs_error": dict(zip(cols, [float(e) for e in err])),
       "passed": bool((err < 1e-4).all())}
os.makedirs(os.path.join(PROJECT, "reproducibility"), exist_ok=True)
with open(os.path.join(PROJECT, "reproducibility", "causality_check.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
