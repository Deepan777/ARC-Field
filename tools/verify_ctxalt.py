"""Check the alternative context encodings (src/context_alt.py) against naive
reference implementations, and check that they are causal.

  1. AfterImage: every statistic equals a brute-force weighted sum over the
     earlier flows of the same key.
  2. ReFeX: base, egonet and recursive features of a snapshot equal a
     networkx / dense-matrix computation.
  3. Causality: perturbing flows at or after a row leaves the encodings of
     earlier rows unchanged (AfterImage also the row itself, whose own bytes
     must not enter its context); ReFeX rows whose snapshot ends before the
     perturbation are unchanged.
Exit code 0 when every check passes.
"""
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import context_alt as ca  # noqa: E402

rng = np.random.default_rng(7)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---- synthetic stream -------------------------------------------------------
n = 2500
t = np.sort(rng.uniform(0, 900, n)).round(1)            # ties on purpose
src = rng.integers(0, 15, n)
dst = rng.integers(0, 25, n)
dport = rng.choice([22, 53, 80, 443], n)
sb = rng.lognormal(6, 2, n).round()
db = np.where(rng.random(n) < 0.2, 0, rng.lognormal(7, 2, n).round())
lams = (5.0, 0.1, 0.01)

A, names = ca.afterimage(src, dst, dport, t, sb, db, lambdas=lams)
x, y = np.log1p(sb), np.log1p(db)

# brute force
ref = np.zeros_like(A, dtype=np.float64)
pair = src * 1000 + dst
svc = pair * 100000 + dport
prev_t = {}
gap = np.full(n, np.nan)
for i in range(n):
    if pair[i] in prev_t:
        gap[i] = t[i] - prev_t[pair[i]]
    prev_t[pair[i]] = t[i]
jit = np.log1p(np.nan_to_num(gap))
for i in range(n):
    c = 0
    for lam in lams:
        def stats1(mask, v, incl_self=False):
            m = mask.copy()
            m[i:] = False
            if incl_self:
                m[i] = mask[i]
            wts = 2.0 ** (-lam * (t[i] - t[m]))
            w = wts.sum()
            if w <= 1e-12:
                return [w, 0, 0]
            mu = (wts * v[m]).sum() / w
            return [w, mu, np.sqrt(max((wts * v[m] ** 2).sum() / w - mu * mu, 0))]

        def stats2(mask):
            m = mask.copy()
            m[i:] = False
            wts = 2.0 ** (-lam * (t[i] - t[m]))
            w = wts.sum()
            if w <= 1e-12:
                return [w] + [0] * 6
            mx, my = (wts * x[m]).sum() / w, (wts * y[m]).sum() / w
            vx = (wts * x[m] ** 2).sum() / w - mx * mx
            vy = (wts * y[m] ** 2).sum() / w - my * my
            vx = 0.0 if vx < ca.REL_VAR_TOL * (1 + mx * mx) else vx
            vy = 0.0 if vy < ca.REL_VAR_TOL * (1 + my * my) else vy
            cov = (wts * x[m] * y[m]).sum() / w - mx * my
            both = vx > 0 and vy > 0
            return [w, mx, np.sqrt(vx), np.sqrt(mx * mx + my * my), np.sqrt(vx + vy),
                    cov if both else 0.0, float(np.clip(cov / np.sqrt(vx * vy), -1, 1)) if both else 0.0]
        ref[i, c:c + 3] = stats1(src == src[i], x); c += 3
        ref[i, c:c + 7] = stats2(pair == pair[i]); c += 7
        ref[i, c:c + 3] = stats1((pair == pair[i]) & ~np.isnan(gap), jit, incl_self=True); c += 3
        ref[i, c:c + 7] = stats2(svc == svc[i]); c += 7
err = np.abs(A - ref) / (1 + np.abs(ref))
check("AfterImage equals brute force", err.max() < 1e-4, f"(max rel err {err.max():.2e}, {A.shape[1]} cols)")

# causality: perturb bytes from row r on
r = n // 2
sb2, db2 = sb.copy(), db.copy()
sb2[r:] = rng.lognormal(9, 1, n - r)
db2[r:] = rng.lognormal(9, 1, n - r)
A2, _ = ca.afterimage(src, dst, dport, t, sb2, db2, lambdas=lams)
check("AfterImage: rows <= r unchanged by bytes of rows >= r", np.array_equal(A[: r + 1], A2[: r + 1]))

# want-subset equals full
want = np.sort(rng.choice(n, 300, replace=False))
Aw, _ = ca.afterimage(src, dst, dport, t, sb, db, want=want, lambdas=lams)
check("AfterImage: subset output equals full output", np.array_equal(Aw, A[want]))

# ---- ReFeX snapshot vs networkx -------------------------------------------
m = 40
us = rng.integers(0, m, 400)
vs = rng.integers(0, m, 400)
vs[:60] = 0                                              # a hub
F = ca.refex_snapshot(us, vs, m)
keep = us != vs
G = nx.MultiDiGraph()
G.add_nodes_from(range(m))
G.add_edges_from(zip(us[keep], vs[keep]))
D = nx.DiGraph(G)
Ug = nx.Graph(D)
tri = nx.triangles(Ug)
base = np.zeros((m, 6))
for v in range(m):
    ego = set(Ug[v]) | {v}
    internal = Ug.subgraph(ego).number_of_edges()
    boundary = sum(1 for a in ego for b in Ug[a] if b not in ego)
    base[v] = [D.out_degree(v), D.in_degree(v), G.out_degree(v), G.in_degree(v), internal, boundary]
check("ReFeX base and egonet features equal networkx",
      np.allclose(F[:, :6], np.log1p(base), atol=1e-5),
      f"(triangles total {sum(tri.values()) // 3})")
Adj = nx.to_numpy_array(Ug, nodelist=range(m))
deg = np.maximum(Adj.sum(1), 1)[:, None]
R = base
for _ in range(2):
    S = Adj @ R
    R = np.hstack([R, S, S / deg])
check("ReFeX recursive features equal dense computation", np.allclose(F, np.log1p(R), rtol=1e-5, atol=1e-4),
      f"({F.shape[1]} features)")

# ---- ReFeX context causality --------------------------------------------------
tt = np.sort(rng.uniform(0, 1800, 3000))
s_ = rng.integers(0, 30, 3000)
d_ = rng.integers(0, 60, 3000)
C1 = ca.refex_context(s_, d_, tt)
T = 1000.0
d2 = d_.copy()
d2[tt >= T] = rng.integers(60, 90, int((tt >= T).sum()))
C2 = ca.refex_context(s_, d2, tt)
snap_end = tt[0] + np.floor((tt - tt[0]) / ca.ROLX_STEP) * ca.ROLX_STEP
before = tt < T
straddle = (tt >= T) & (snap_end <= T)        # own destination changed, snapshot did not
src_cols = list(range(ca.N_REFEX)) + [2 * ca.N_REFEX]
check("ReFeX context: rows before a change are unchanged",
      np.array_equal(C1[before], C2[before]), f"({int(before.sum())} rows)")
check("ReFeX context: source side unchanged when the snapshot ends before the change",
      np.array_equal(C1[straddle][:, src_cols], C2[straddle][:, src_cols]), f"({int(straddle.sum())} rows)")
check("ReFeX context: later rows do change", not np.array_equal(C1[snap_end > T], C2[snap_end > T]))
want = np.sort(rng.choice(3000, 400, replace=False))
check("ReFeX context: subset output equals full output",
      np.array_equal(ca.refex_context(s_, d_, tt, want=want), C1[want]))

# ---- RolX ---------------------------------------------------------------------
rx = ca.RolX(k_max=6).fit(ca.rolx_fit_rows(C1.astype(np.float32)))
Z = ca.rolx_features(C1.astype(np.float32), rx)
check("RolX: memberships finite, k chosen", np.isfinite(Z).all() and 2 <= rx.k <= 6, f"(k={rx.k})")

print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
