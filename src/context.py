"""Causal, identity-free endpoint context ("role coordinates").

Every quantity computed here for a flow i depends only on flows emitted
*strictly before* i in the capture order and inside a trailing time window
(t_i - W, t_i]. Endpoint identifiers are used only as grouping keys: the output
is a vector of counts, rates and novelty fractions, never an identifier or an
embedding looked up by identifier. Two endpoints that behave identically
therefore receive identical coordinates, and an endpoint never seen before
receives a well-defined coordinate from its first flow onwards.

The vectorised implementation below is verified against a flow-at-a-time
streaming reference (`StreamingSketch`) in `code/tools/verify_causality.py`.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _group_offsets(keys: np.ndarray, t: np.ndarray, pad: float):
    """Sort rows by (key, position) and build a composite, strictly separated
    time axis so that one global `searchsorted` handles every group at once.

    `t` must already be non-decreasing in row (capture) order.
    """
    order = np.lexsort((np.arange(len(keys)), keys))       # key, then position
    k_sorted = keys[order]
    _, k_rank = np.unique(k_sorted, return_inverse=True)
    span = float(t.max() - t.min()) + pad + 1.0
    comp = k_rank.astype(np.float64) * span + (t[order] - t.min())
    return order, comp


def window_sums(keys: np.ndarray, t: np.ndarray, W: float,
                values: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """For each row i: count and sums of `values` over earlier rows j < i with
    keys[j] == keys[i] and t[j] > t[i] - W.

    Returns (count, [sum_v for v in values]) aligned with the input rows.
    """
    n = len(keys)
    order, comp = _group_offsets(keys, t, W)
    lo = np.searchsorted(comp, comp - W, side="right")
    idx = np.arange(n)
    # Rows lo..idx-1 are the earlier rows of the same key inside the window.
    cnt_sorted = (idx - lo).astype(np.float64)
    out_cnt = np.empty(n)
    out_cnt[order] = cnt_sorted
    sums = []
    for v in values:
        vs = v[order].astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(vs)])
        s_sorted = cs[idx] - cs[lo]
        o = np.empty(n)
        o[order] = s_sorted
        sums.append(o)
    return out_cnt, sums


def previous_gap(keys: np.ndarray, t: np.ndarray) -> np.ndarray:
    """t[i] - t[j*] where j* is the most recent earlier row with the same key;
    +inf if there is none."""
    n = len(keys)
    order = np.lexsort((np.arange(n), keys))
    ks, ts = keys[order], t[order]
    gap_sorted = np.full(n, np.inf)
    same = ks[1:] == ks[:-1]
    gap_sorted[1:][same] = ts[1:][same] - ts[:-1][same]
    out = np.empty(n)
    out[order] = gap_sorted
    return out


def pair_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Collision-free integer key for an ordered pair of integer codes."""
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    return a * (int(b.max()) + 1) + b


# ---------------------------------------------------------------------------
# UNSW-NB15: host-level role coordinates
# ---------------------------------------------------------------------------

UNSW_CONTEXT_WINDOWS = (10.0, 300.0)   # seconds: burst scale and session scale
PAIR_GAP_CAP = 3600.0


def unsw_context(src: np.ndarray, dst: np.ndarray, dport: np.ndarray,
                 t: np.ndarray, sbytes: np.ndarray, dbytes: np.ndarray,
                 dur: np.ndarray, proto: np.ndarray,
                 windows: Iterable[float] = UNSW_CONTEXT_WINDOWS
                 ) -> Tuple[np.ndarray, List[str]]:
    """Role coordinates for every flow of a capture-ordered UNSW-NB15 stream.

    Inputs are integer-coded endpoints (`src`, `dst`), destination port, the
    emission time `t` (non-decreasing) and the per-flow observables used to
    summarise behaviour. Only encryption-observable quantities are used: byte
    and packet totals, duration, whether the responder answered at all, and
    endpoint/port novelty.
    """
    ls = np.log1p(np.maximum(sbytes, 0))
    ld = np.log1p(np.maximum(dbytes, 0))
    lt = np.log1p(np.maximum(dur, 0))
    noresp = (dbytes <= 0).astype(np.float64)

    k_sd = pair_key(src, dst)
    k_sp = pair_key(src, dport)
    k_ds = pair_key(dst, src)
    gap_sd = previous_gap(k_sd, t)
    gap_sp = previous_gap(k_sp, t)

    cols, names = [], []
    for W in windows:
        tag = f"{int(W)}s"
        # Outbound behaviour of the source endpoint.
        new_dst = (gap_sd > W).astype(np.float64)
        new_port = (gap_sp > W).astype(np.float64)
        n, (s_nd, s_np, s_nr, s_ls, s_ld, s_lt) = window_sums(
            src, t, W, [new_dst, new_port, noresp, ls, ld, lt])
        den = np.maximum(n, 1.0)
        cols += [np.log1p(n), s_nd / den, s_np / den, s_nr / den,
                 s_ls / den, s_ld / den, s_lt / den]
        names += [f"src_out_n_{tag}", f"src_out_newdst_{tag}", f"src_out_newport_{tag}",
                  f"src_out_noresp_{tag}", f"src_out_lsb_{tag}", f"src_out_ldb_{tag}",
                  f"src_out_ldur_{tag}"]
        # Inbound behaviour of the destination endpoint.
        new_src = (previous_gap(k_ds, t) > W).astype(np.float64)
        n2, (s_ns, s_nr2, s_ls2) = window_sums(dst, t, W, [new_src, noresp, ls])
        den2 = np.maximum(n2, 1.0)
        cols += [np.log1p(n2), s_ns / den2, s_nr2 / den2, s_ls2 / den2]
        names += [f"dst_in_n_{tag}", f"dst_in_newsrc_{tag}", f"dst_in_noresp_{tag}",
                  f"dst_in_lsb_{tag}"]
    # Cross roles at the longest window: how often the source is itself
    # contacted, and how often the destination itself initiates.
    Wl = max(windows)
    n_src_as_dst = _cross_count(src, dst, t, Wl)
    n_dst_as_src = _cross_count(dst, src, t, Wl)
    cols += [np.log1p(n_src_as_dst), np.log1p(n_dst_as_src)]
    names += [f"src_in_n_{int(Wl)}s", f"dst_out_n_{int(Wl)}s"]
    # Pair recency (shift-invariant relative time) and pair intensity.
    n_pair, _ = window_sums(k_sd, t, Wl, [])
    cols += [np.log1p(np.minimum(gap_sd, PAIR_GAP_CAP)), np.log1p(n_pair)]
    names += ["pair_gap_log", f"pair_n_{int(Wl)}s"]
    # Transport protocol (observable in the outer IP header).
    cols += [(proto == 0).astype(np.float64), (proto == 1).astype(np.float64),
             (proto >= 2).astype(np.float64)]
    names += ["proto_tcp", "proto_udp", "proto_other"]
    C = np.stack(cols, axis=1).astype(np.float32)
    return C, names


def _cross_count(query_key: np.ndarray, event_key: np.ndarray, t: np.ndarray,
                 W: float) -> np.ndarray:
    """For row i: number of earlier rows j with event_key[j] == query_key[i]
    and t[j] > t[i] - W, i.e. how often endpoint query_key[i] appeared in the
    *other* role recently. Implemented by merging the two role streams."""
    n = len(t)
    # Stream of events (role occurrences) and stream of queries, merged in
    # capture order; queries sort after events at the same position so an
    # event at row j < i is counted but the query row's own event is not.
    keys = np.concatenate([event_key, query_key]).astype(np.int64)
    tt = np.concatenate([t, t])
    pos = np.concatenate([np.arange(n) * 2, np.arange(n) * 2 - 1]).astype(np.int64)
    # query at row i gets position 2i-1 < event position 2i, so its own event
    # (row i) is excluded while every event at j < i (position 2j < 2i-1) is in.
    is_event = np.concatenate([np.ones(n), np.zeros(n)])
    order_pos = np.argsort(pos, kind="stable")
    keys, tt, is_event = keys[order_pos], tt[order_pos], is_event[order_pos]
    orig = order_pos  # index into concatenated arrays
    order, comp = _group_offsets(keys, tt, W)
    ev_sorted = is_event[order]
    cs = np.concatenate([[0.0], np.cumsum(ev_sorted)])
    lo = np.searchsorted(comp, comp - W, side="right")
    idx = np.arange(len(keys))
    val_sorted = cs[idx] - cs[lo]
    val = np.empty(len(keys))
    val[order] = val_sorted
    # map back to query rows
    out = np.empty(n)
    q_mask = orig >= n
    out[orig[q_mask] - n] = val[q_mask]
    return out


# ---------------------------------------------------------------------------
# CIC-IDS2017 (MachineLearningCVE release): service-level context
# ---------------------------------------------------------------------------

CIC_CONTEXT_WINDOWS = (200.0, 5000.0)   # rows; this release has no timestamps


def cic_context(dport: np.ndarray, pos: np.ndarray, fwd_bytes: np.ndarray,
                bwd_pkts: np.ndarray, dur: np.ndarray,
                windows: Iterable[float] = CIC_CONTEXT_WINDOWS
                ) -> Tuple[np.ndarray, List[str]]:
    """Context for a release without endpoint addresses or timestamps.

    The only endpoint attribute available is the destination port, so the
    destination endpoint is represented at service granularity, and the row
    index inside each capture day serves as the clock. Global stream context
    (rate of previously unseen ports) is added because it is the only way a
    per-flow detector on this release can register scanning.
    """
    lf = np.log1p(np.maximum(fwd_bytes, 0))
    lt = np.log1p(np.maximum(dur, 0))
    noresp = (bwd_pkts <= 0).astype(np.float64)
    glob = np.zeros(len(dport), dtype=np.int64)
    gap_p = previous_gap(dport.astype(np.int64), pos)
    cols, names = [], []
    for W in windows:
        tag = f"{int(W)}r"
        n, (s_nr, s_lf, s_lt) = window_sums(dport.astype(np.int64), pos, W, [noresp, lf, lt])
        den = np.maximum(n, 1.0)
        cols += [np.log1p(n), s_nr / den, s_lf / den, s_lt / den]
        names += [f"svc_n_{tag}", f"svc_noresp_{tag}", f"svc_lfb_{tag}", f"svc_ldur_{tag}"]
        newp = (gap_p > W).astype(np.float64)
        ng, (g_np, g_nr) = window_sums(glob, pos, W, [newp, noresp])
        deng = np.maximum(ng, 1.0)
        cols += [g_np / deng, g_nr / deng]
        names += [f"glob_newport_{tag}", f"glob_noresp_{tag}"]
    cols += [np.log1p(np.minimum(gap_p, 1e5))]
    names += ["svc_gap_log"]
    return np.stack(cols, axis=1).astype(np.float32), names


# ---------------------------------------------------------------------------
# Streaming reference implementation (used only to verify causality)
# ---------------------------------------------------------------------------

class StreamingSketch:
    """Flow-at-a-time computation of the outbound source sketch for one window.

    Deliberately naive: per-endpoint deques of (time, values). Used to check
    that the vectorised `window_sums` reproduces a genuinely causal stream
    computation exactly.
    """

    def __init__(self, W: float):
        self.W = W
        self.q: Dict[int, deque] = defaultdict(deque)

    def query_then_update(self, key: int, t: float, vals: Tuple[float, ...]):
        q = self.q[key]
        while q and q[0][0] <= t - self.W:
            q.popleft()
        cnt = len(q)
        sums = [sum(v[1][k] for v in q) for k in range(len(vals))]
        q.append((t, vals))
        return cnt, sums
