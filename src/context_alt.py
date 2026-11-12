"""Alternative endpoint-context encodings for the context-baseline study.

The role coordinates of context.py are compared with the two established ways
of describing an endpoint by its behaviour rather than by its identity:

* AfterImage (Mirsky et al., NDSS 2018; the feature extractor of Kitsune):
  damped incremental statistics per source host, per host-to-host channel
  (with the 2-D statistics of the two directions), of the channel's
  inter-arrival jitter, and per service channel (source, destination,
  destination port), each at the five Kitsune decay rates. Adapted from
  packets to flow records: the value of a record is its log byte count in
  each direction, and the socket level becomes the service channel because
  ephemeral source ports make socket-level state degenerate for flows.
* ReFeX / RolX (Henderson et al., KDD 2011 / KDD 2012): structural roles of
  endpoints in the communication graph. ReFeX gives every node of a snapshot
  graph local and egonet features and two rounds of recursive neighbour sums
  and means. RolX factorises them with non-negative matrix factorisation into
  role memberships, with the number of roles chosen by a description-length
  criterion. Snapshots are rebuilt every `step` seconds from the flows of the
  preceding `W` seconds.

Both keep the causal contract of context.py: the encoding attached to flow i
depends only on flows emitted strictly before it (AfterImage: query before
update, except the inter-arrival gap that ends at flow i, as in the role
coordinates' pair recency; ReFeX: snapshots end strictly before the flow's
own timestamp). No hyper-parameter is tuned: the decay rates are Kitsune's
defaults, the snapshot window equals the longer role-coordinate window, and
the number of roles is chosen by RolX's own criterion on the fitting rows.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numba
import numpy as np
from scipy import sparse

from context import pair_key, previous_gap

KITSUNE_LAMBDAS = (5.0, 3.0, 1.0, 0.1, 0.01)     # 1/s, Kitsune defaults
ROLX_WINDOW = 300.0                               # s, = longer role window
ROLX_STEP = 60.0                                  # s, snapshot stride
N_REFEX = 54                                      # 6 base -> 18 -> 54
REL_VAR_TOL = 1e-6                                # relative variance floor (2-D stats)


def _dense(key: np.ndarray) -> Tuple[np.ndarray, int]:
    u, inv = np.unique(key, return_inverse=True)
    return inv.astype(np.int64), len(u)


# ---------------------------------------------------------------------------
# AfterImage
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _damped_1d(key, t, x, valid, lam, nkeys, update_first, slot, n_out):
    out = np.zeros((n_out, 3), np.float32)
    w = np.zeros(nkeys)
    ls = np.zeros(nkeys)
    ss = np.zeros(nkeys)
    tl = np.zeros(nkeys)
    seen = np.zeros(nkeys, np.bool_)
    for i in range(key.shape[0]):
        k = key[i]
        if seen[k]:
            d = 2.0 ** (-lam * (t[i] - tl[k]))
            w[k] *= d
            ls[k] *= d
            ss[k] *= d
        tl[k] = t[i]
        seen[k] = True
        if update_first and valid[i]:
            w[k] += 1.0
            ls[k] += x[i]
            ss[k] += x[i] * x[i]
        j = slot[i]
        if j >= 0:
            ww = w[k]
            out[j, 0] = ww
            if ww > 1e-12:
                mu = ls[k] / ww
                var = ss[k] / ww - mu * mu
                out[j, 1] = mu
                out[j, 2] = np.sqrt(var) if var > 0.0 else 0.0
        if (not update_first) and valid[i]:
            w[k] += 1.0
            ls[k] += x[i]
            ss[k] += x[i] * x[i]
    return out


@numba.njit(cache=True)
def _damped_2d(key, t, x, y, lam, nkeys, slot, n_out):
    out = np.zeros((n_out, 7), np.float32)
    w = np.zeros(nkeys)
    lx = np.zeros(nkeys)
    sx = np.zeros(nkeys)
    ly = np.zeros(nkeys)
    sy = np.zeros(nkeys)
    sxy = np.zeros(nkeys)
    tl = np.zeros(nkeys)
    seen = np.zeros(nkeys, np.bool_)
    for i in range(key.shape[0]):
        k = key[i]
        if seen[k]:
            d = 2.0 ** (-lam * (t[i] - tl[k]))
            w[k] *= d
            lx[k] *= d
            sx[k] *= d
            ly[k] *= d
            sy[k] *= d
            sxy[k] *= d
        tl[k] = t[i]
        seen[k] = True
        j = slot[i]
        if j >= 0:
            ww = w[k]
            out[j, 0] = ww
            if ww > 1e-12:
                mx = lx[k] / ww
                my = ly[k] / ww
                vx = sx[k] / ww - mx * mx
                vy = sy[k] / ww - my * my
                # E[x^2] - mu^2 cancels catastrophically for a near-constant
                # stream (e.g. no response bytes); treat such variances as 0
                if vx < REL_VAR_TOL * (1.0 + mx * mx):
                    vx = 0.0
                if vy < REL_VAR_TOL * (1.0 + my * my):
                    vy = 0.0
                cov = sxy[k] / ww - mx * my
                out[j, 1] = mx
                out[j, 2] = np.sqrt(vx)
                out[j, 3] = np.sqrt(mx * mx + my * my)        # magnitude
                out[j, 4] = np.sqrt(vx + vy)                  # radius
                if vx > 0.0 and vy > 0.0:
                    out[j, 5] = cov                           # covariance
                    out[j, 6] = min(max(cov / np.sqrt(vx * vy), -1.0), 1.0)   # correlation
        w[k] += 1.0
        lx[k] += x[i]
        sx[k] += x[i] * x[i]
        ly[k] += y[i]
        sy[k] += y[i] * y[i]
        sxy[k] += x[i] * y[i]
    return out


def afterimage(src: np.ndarray, dst: np.ndarray, dport: np.ndarray, t: np.ndarray,
               sbytes: np.ndarray, dbytes: np.ndarray, want: Optional[np.ndarray] = None,
               lambdas=KITSUNE_LAMBDAS, out: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """AfterImage statistics for a capture-ordered stream (t non-decreasing).

    `want` (optional, sorted row indices) limits the output to those rows; the
    state is still updated by every row. Returns (n_out, 20 * len(lambdas)),
    written into `out` (e.g. a memory map) when given.
    """
    n = len(t)
    slot = np.full(n, -1, np.int64)
    if want is None:
        slot[:] = np.arange(n)
        n_out = n
    else:
        slot[want] = np.arange(len(want))
        n_out = len(want)
    t = t.astype(np.float64)
    x = np.log1p(np.maximum(sbytes, 0)).astype(np.float64)
    y = np.log1p(np.maximum(dbytes, 0)).astype(np.float64)
    kh, nh = _dense(src.astype(np.int64))
    pk = pair_key(src, dst)
    kc, nc = _dense(pk)
    ks, ns = _dense(pair_key(kc, dport.astype(np.int64)))
    gap = previous_gap(pk, t)
    valid = np.isfinite(gap)
    jit = np.where(valid, np.log1p(np.where(valid, gap, 0.0)), 0.0)
    ones = np.ones(n, np.bool_)
    if out is None:
        out = np.zeros((n_out, 20 * len(lambdas)), np.float32)
    names, c = [], 0
    for lam in lambdas:
        tag = f"l{lam:g}"
        for block, nm in (
                (lambda: _damped_1d(kh, t, x, ones, lam, nh, False, slot, n_out),
                 [f"H_{s}_{tag}" for s in ("w", "mu", "sd")]),
                (lambda: _damped_2d(kc, t, x, y, lam, nc, slot, n_out),
                 [f"HH_{s}_{tag}" for s in ("w", "mu", "sd", "mag", "rad", "cov", "pcc")]),
                (lambda: _damped_1d(kc, t, jit, valid, lam, nc, True, slot, n_out),
                 [f"HHjit_{s}_{tag}" for s in ("w", "mu", "sd")]),
                (lambda: _damped_2d(ks, t, x, y, lam, ns, slot, n_out),
                 [f"HpHp_{s}_{tag}" for s in ("w", "mu", "sd", "mag", "rad", "cov", "pcc")])):
            out[:, c:c + len(nm)] = block()
            names += nm
            c += len(nm)
    return out, names


# ---------------------------------------------------------------------------
# ReFeX features of snapshot graphs
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _triangles(indptr, indices, n):
    """Per-node triangle counts of an undirected graph whose CSR keeps, for
    each node, only its neighbours of higher (degree, id) rank (forward
    algorithm, O(m^1.5))."""
    tri = np.zeros(n)
    mark = np.full(n, -1, np.int64)
    for u in range(n):
        for p in range(indptr[u], indptr[u + 1]):
            mark[indices[p]] = u
        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            for q in range(indptr[v], indptr[v + 1]):
                w = indices[q]
                if mark[w] == u:
                    tri[u] += 1.0
                    tri[v] += 1.0
                    tri[w] += 1.0
    return tri


def refex_snapshot(us: np.ndarray, vs: np.ndarray, m: int) -> np.ndarray:
    """ReFeX features (m, 54) of a directed multigraph given by flow edges."""
    keep = us != vs
    us, vs = us[keep], vs[keep]
    Wd = sparse.csr_matrix((np.ones(len(us)), (us, vs)), shape=(m, m))
    B = Wd.copy()
    B.data[:] = 1.0
    outdeg = np.asarray(B.sum(1)).ravel()
    indeg = np.asarray(B.sum(0)).ravel()
    outw = np.asarray(Wd.sum(1)).ravel()
    inw = np.asarray(Wd.sum(0)).ravel()
    U = (B + B.T).tocsr()
    U.data[:] = 1.0
    deg = np.asarray(U.sum(1)).ravel()
    rank = np.empty(m, np.int64)
    rank[np.lexsort((np.arange(m), deg))] = np.arange(m)
    coo = U.tocoo()
    fw = rank[coo.col] > rank[coo.row]
    O = sparse.csr_matrix((np.ones(int(fw.sum())), (coo.row[fw], coo.col[fw])), shape=(m, m))
    O.sort_indices()
    tri = _triangles(O.indptr.astype(np.int64), O.indices.astype(np.int64), m)
    internal = deg + tri
    boundary = np.maximum(U @ deg - deg - 2.0 * tri, 0.0)
    F = np.stack([outdeg, indeg, outw, inw, internal, boundary], axis=1)
    den = np.maximum(deg, 1.0)[:, None]
    for _ in range(2):
        S = U @ F
        F = np.hstack([F, S, S / den])
    return np.log1p(np.maximum(F, 0.0)).astype(np.float32)


def refex_context(src: np.ndarray, dst: np.ndarray, t: np.ndarray, want: Optional[np.ndarray] = None,
                  W: float = ROLX_WINDOW, step: float = ROLX_STEP,
                  verbose: bool = False, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-flow ReFeX vectors of the source and destination endpoints in the
    latest snapshot that ends strictly before the flow, plus presence flags.

    Snapshot k ends at s_k = t[0] + k * step and contains the flows with
    s_k - W <= t < s_k; a flow with s_k <= t < s_{k+1} uses snapshot k.
    Returns float16 (n_out, 2 * 54 + 2): [ReFeX(src), ReFeX(dst), src_present,
    dst_present]; absent endpoints get zeros.
    """
    n = len(t)
    t = t.astype(np.float64)
    rows = np.arange(n) if want is None else np.asarray(want)
    if out is None:
        out = np.zeros((len(rows), 2 * N_REFEX + 2), np.float16)
    snap = np.floor((t[rows] - t[0]) / step).astype(np.int64)
    ks, starts = np.unique(snap, return_index=True)
    ends = np.append(starts[1:], len(rows))
    for i, (k, a, b) in enumerate(zip(ks, starts, ends)):
        s = t[0] + k * step
        lo = np.searchsorted(t, s - W, side="left")
        hi = np.searchsorted(t, s, side="left")
        if hi <= lo:
            continue
        nodes, inv = np.unique(np.concatenate([src[lo:hi], dst[lo:hi]]), return_inverse=True)
        F = refex_snapshot(inv[: hi - lo], inv[hi - lo:], len(nodes))
        for side, col0, flag in ((src, 0, 2 * N_REFEX), (dst, N_REFEX, 2 * N_REFEX + 1)):
            q = side[rows[a:b]]
            pos = np.minimum(np.searchsorted(nodes, q), len(nodes) - 1)
            found = nodes[pos] == q
            blk = np.zeros((b - a, N_REFEX), np.float32)
            blk[found] = F[pos[found]]
            out[a:b, col0:col0 + N_REFEX] = blk
            out[a:b, flag] = found
        if verbose and i % 500 == 0:
            print(f"    refex snapshot {i}/{len(ks)} ({len(nodes)} nodes, {hi - lo} flows)", flush=True)
    return out


# ---------------------------------------------------------------------------
# RolX: role memberships from ReFeX vectors (fitted on the fitting rows only)
# ---------------------------------------------------------------------------

class RolX:
    """NMF role extraction. `fit` takes ReFeX rows of the endpoints of the
    benign fitting flows (present endpoints only); the number of roles k is
    the minimiser of a description length, 0.5 k (n + f) log2(n f) bits for
    the factors plus 0.5 n f log2(RSS / (n f)) bits for the residual, as in
    RolX's MDL model selection with Gaussian residual coding."""

    def __init__(self, k_max: int = 12, seed: int = 0, n_select: int = 20000, n_fit: int = 100000):
        self.k_max, self.seed, self.n_select, self.n_fit = k_max, seed, n_select, n_fit

    def _prep(self, F):
        return np.minimum(F.astype(np.float64) / self.scale, 10.0)

    def fit(self, F: np.ndarray) -> "RolX":
        from sklearn.decomposition import NMF
        rng = np.random.default_rng(self.seed)
        F = F.astype(np.float64)
        self.scale = np.percentile(F, 99, axis=0) + 1e-6
        V = self._prep(F)
        sel = V[rng.choice(len(V), size=min(self.n_select, len(V)), replace=False)]
        n, f = sel.shape
        self.dl = {}
        for k in range(2, self.k_max + 1):
            nmf = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=self.seed)
            Wk = nmf.fit_transform(sel)
            rss = float(((sel - Wk @ nmf.components_) ** 2).sum())
            self.dl[k] = 0.5 * k * (n + f) * np.log2(n * f) + 0.5 * n * f * np.log2(rss / (n * f) + 1e-12)
        self.k = min(self.dl, key=self.dl.get)
        fit_rows = V[rng.choice(len(V), size=min(self.n_fit, len(V)), replace=False)]
        self.nmf = NMF(n_components=self.k, init="nndsvda", max_iter=400, random_state=self.seed).fit(fit_rows)
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        """(n, k + 1): normalised role memberships and log total membership."""
        Wm = self.nmf.transform(self._prep(F))
        mag = Wm.sum(1, keepdims=True)
        mem = np.where(mag > 0, Wm / np.maximum(mag, 1e-12), 0.0)
        return np.hstack([mem, np.log1p(mag)]).astype(np.float32)


def rolx_features(R: np.ndarray, model: RolX) -> np.ndarray:
    """Per-flow RolX context from stored ReFeX rows R (n, 110)."""
    Fs, Fd = R[:, :N_REFEX], R[:, N_REFEX:2 * N_REFEX]
    ps, pd_ = R[:, 2 * N_REFEX:2 * N_REFEX + 1], R[:, 2 * N_REFEX + 1:]
    return np.hstack([model.transform(Fs), ps, model.transform(Fd), pd_]).astype(np.float32)


def rolx_fit_rows(R: np.ndarray) -> np.ndarray:
    """ReFeX rows of the present endpoints (source and destination sides)."""
    Fs = R[R[:, 2 * N_REFEX] > 0, :N_REFEX]
    Fd = R[R[:, 2 * N_REFEX + 1] > 0, N_REFEX:2 * N_REFEX]
    return np.vstack([Fs, Fd]).astype(np.float32)
