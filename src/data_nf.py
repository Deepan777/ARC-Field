"""NF-CSE-CIC-IDS2018-v3 (Luay et al., 2025): NetFlow v3 corpus, feature
tiers, role context and the cross-day protocol (Protocol V).

The corpus has 20.1 M flow records with source/destination addresses, ports,
flow start/end timestamps (ms) and 53 NetFlow features, captured over ten
working days on an emulated enterprise network of about 450 hosts. Unlike the
CIC-IDS2017 release, addresses and timestamps are present, so ARC-Field uses
the same host-level role coordinates as on UNSW-NB15 and the per-host cap of
the recalibration is active.

Construction (two streaming passes over the CSV inside the authors' zip):
  1. addresses, ports, protocol, bytes, duration, TTL and labels of every flow;
     emission order (flow end, flow start, file position); capture days are
     segments separated by more than three idle hours; role coordinates are
     computed causally over the complete, unthinned stream of each day;
  2. the NetFlow features of the rows that any protocol seed can use.
"""
from __future__ import annotations

import os
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd

from paths import DATA_DIR, NF_ZIP
import context as ctx

# ---------------------------------------------------------------------------
# Feature tiers (same observability logic as UNSW-NB15)
#   A  outer IP header + packet sizes and timing (survives TLS, QUIC, tunnels)
#   B  needs a readable TCP header
#   C  needs payload inspection (DNS/FTP fields, nDPI L7 protocol): excluded
# ---------------------------------------------------------------------------
VOLUME = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
          "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
          "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
          "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
          "NUM_PKTS_1024_TO_1514_BYTES"]
TIMING = ["FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
          "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
          "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV"]
HEADER = ["MIN_TTL", "MAX_TTL", "ICMP_TYPE", "ICMP_IPV4_TYPE"]
TIER_B = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
          "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
          "RETRANSMITTED_OUT_PKTS"]
TTL_FEATURES = ["MIN_TTL", "MAX_TTL"]
TIER_A = VOLUME + TIMING + HEADER
ALL = TIER_A + TIER_B
FEATURE_SETS = {
    "A": TIER_A,                                             # 32
    "A_noTTL": [f for f in TIER_A if f not in TTL_FEATURES],  # 30
    "AB": ALL,                                               # 41
}
GROUPS = {"volume": VOLUME, "timing": TIMING, "header": HEADER, "tcp": TIER_B}
HEAVY = set(VOLUME + TIMING + ["TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "RETRANSMITTED_IN_BYTES",
                              "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS"])

PASS1 = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
         "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS", "MIN_TTL",
         "Label", "Attack"]
DAY_GAP_S = 3 * 3600.0
PER_DAY_TEST = 100_000
N_POOL = 200_000
N_CAL = 5000
N_SEEDS = 3          # test offsets stored for seeds 0..N_SEEDS-1


def groups_for(features: List[str]) -> List[List[int]]:
    out = []
    for members in GROUPS.values():
        idx = [features.index(f) for f in members if f in features]
        if idx:
            out.append(idx)
    return out


def group_names_for(features: List[str]) -> List[str]:
    return [g for g, m in GROUPS.items() if any(f in features for f in m)]


def _member(z: zipfile.ZipFile) -> str:
    return next(n for n in z.namelist() if n.endswith("data/NF-CICIDS2018-v3.csv"))


def _chunks(cols, chunksize=2_000_000):
    z = zipfile.ZipFile(NF_ZIP)
    with z.open(_member(z)) as f:
        yield from pd.read_csv(f, usecols=cols, chunksize=chunksize, low_memory=False)


def _pass1(verbose=True):
    ip_index: Dict[str, int] = {}
    att_index: Dict[str, int] = {}
    parts = {k: [] for k in ("t0", "t1", "src", "dst", "dport", "proto", "ib", "ob", "dur", "ttl",
                             "lab", "att")}
    n = 0
    for ch in _chunks(PASS1):
        for col in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
            for u in pd.unique(ch[col]):
                if u not in ip_index:
                    ip_index[u] = len(ip_index)
        for u in pd.unique(ch["Attack"]):
            if u not in att_index:
                att_index[u] = len(att_index)
        parts["t0"].append(ch["FLOW_START_MILLISECONDS"].to_numpy(np.int64))
        parts["t1"].append(ch["FLOW_END_MILLISECONDS"].to_numpy(np.int64))
        parts["src"].append(ch["IPV4_SRC_ADDR"].map(ip_index).to_numpy(np.int32))
        parts["dst"].append(ch["IPV4_DST_ADDR"].map(ip_index).to_numpy(np.int32))
        parts["dport"].append(ch["L4_DST_PORT"].to_numpy(np.int32))
        parts["proto"].append(ch["PROTOCOL"].to_numpy(np.int16))
        parts["ib"].append(ch["IN_BYTES"].to_numpy(np.float64))
        parts["ob"].append(ch["OUT_BYTES"].to_numpy(np.float64))
        parts["dur"].append(ch["FLOW_DURATION_MILLISECONDS"].to_numpy(np.float64) / 1000.0)
        parts["ttl"].append(ch["MIN_TTL"].to_numpy(np.int16))
        parts["lab"].append(ch["Label"].to_numpy(np.int8))
        parts["att"].append(ch["Attack"].map(att_index).to_numpy(np.int16))
        n += len(ch)
        if verbose:
            print(f"  pass 1: {n:,} rows", flush=True)
    d = {k: np.concatenate(v) for k, v in parts.items()}
    ips = np.array(sorted(ip_index, key=ip_index.get))
    attacks = np.array(sorted(att_index, key=att_index.get))
    return d, ips, attacks


def build_corpus(verbose: bool = True) -> Dict:
    d, ips, attacks = _pass1(verbose)
    n = len(d["t1"])
    pos = np.arange(n)
    order = np.lexsort((pos, d["t0"], d["t1"]))                # emission order
    d = {k: v[order] for k, v in d.items()}
    t = d["t1"].astype(np.float64) / 1000.0
    gaps = np.diff(t)
    cuts = np.where(gaps > DAY_GAP_S)[0] + 1
    bounds = np.concatenate([[0], cuts, [n]])
    day = np.zeros(n, np.int16)
    for i in range(len(bounds) - 1):
        day[bounds[i]:bounds[i + 1]] = i
    proto = np.where(d["proto"] == 6, 0, np.where(d["proto"] == 17, 1, 2)).astype(np.int64)
    is_normal = d["lab"] == 0

    # rows any seed can use: fitting pool and calibration block of day 0,
    # and the systematic test subsamples (offsets 0..N_SEEDS-1) of later days
    rng = np.random.default_rng(20180214)
    ben0 = np.where((day == 0) & is_normal)[0]
    cal = ben0[-N_CAL:]
    pool = ben0[:-N_CAL]
    pool = np.sort(rng.choice(pool, size=min(N_POOL, len(pool)), replace=False))
    role = np.full(n, -1, np.int8)                  # -1 unused, 0 pool, 1 cal, 2 test
    role[pool] = 0
    role[cal] = 1
    rank = np.zeros(n, np.int64)
    stride = np.ones(int(day.max()) + 1, np.int64)
    for i in range(1, len(bounds) - 1):
        idx = np.arange(bounds[i], bounds[i + 1])
        stride[i] = max(1, len(idx) // PER_DAY_TEST)
        rank[idx] = np.arange(len(idx))
        keep = idx[(rank[idx] % stride[i]) < N_SEEDS]
        role[keep] = 2
    sel = np.where(role >= 0)[0]                     # emission order

    # role coordinates, causally over the full (unthinned) stream of each day;
    # only the selected rows are kept
    C_sel, cnames = None, None
    for i in range(len(bounds) - 1):
        s = slice(bounds[i], bounds[i + 1])
        Ci, cnames = ctx.unsw_context(src=d["src"][s].astype(np.int64), dst=d["dst"][s].astype(np.int64),
                                      dport=d["dport"][s].astype(np.int64), t=t[s], sbytes=d["ib"][s],
                                      dbytes=d["ob"][s], dur=d["dur"][s], proto=proto[s])
        if C_sel is None:
            C_sel = np.zeros((len(sel), Ci.shape[1]), np.float32)
        in_day = (sel >= bounds[i]) & (sel < bounds[i + 1])
        C_sel[in_day] = Ci[sel[in_day] - bounds[i]]
        del Ci
        if verbose:
            print(f"  context day {i}: {bounds[i + 1] - bounds[i]:,} flows", flush=True)

    # pass 2: features of the selected rows (map original file positions)
    orig_sel = order[sel]
    want = np.zeros(n, bool)
    want[orig_sel] = True
    where_in_sel = np.full(n, -1, np.int64)
    where_in_sel[orig_sel] = np.arange(len(sel))
    X = np.zeros((len(sel), len(ALL)), np.float32)
    start = 0
    for ch in _chunks(ALL):
        m = want[start:start + len(ch)]
        if m.any():
            rows = where_in_sel[start:start + len(ch)][m]
            # usecols returns file order; reorder to ALL explicitly
            X[rows] = ch[ALL].to_numpy(np.float64)[m].clip(-1e12, 1e12).astype(np.float32)
        start += len(ch)
        if verbose:
            print(f"  pass 2: {start:,} rows", flush=True)

    attacks = np.array(["Normal" if a.strip().lower() == "benign" else a for a in attacks])
    out = {"X": X, "feat_names": np.array(ALL), "C": C_sel, "ctx_names": np.array(cnames),
           "src": d["src"][sel].astype(np.int64), "dst": d["dst"][sel].astype(np.int64),
           "dport": d["dport"][sel].astype(np.int64), "t": t[sel], "day": day[sel],
           "role": role[sel], "rank": rank[sel], "stride": stride,
           "label": attacks[d["att"][sel]].astype(str), "is_normal": is_normal[sel],
           "ips": ips, "day_start": t[bounds[:-1]], "day_n": np.diff(bounds)}
    # corpus profile (full stream, before selection)
    prof = {"n_flows": int(n), "n_ips": int(len(ips)), "attack_share": float(1 - is_normal.mean()),
            "days": []}
    for i in range(len(bounds) - 1):
        s = slice(bounds[i], bounds[i + 1])
        att = attacks[d["att"][s]]
        vc = pd.Series(att[~is_normal[s]]).value_counts().to_dict()
        prof["days"].append({"day": i, "start_utc": str(pd.to_datetime(t[bounds[i]], unit="s")),
                             "end_utc": str(pd.to_datetime(t[bounds[i + 1] - 1], unit="s")),
                             "n": int(bounds[i + 1] - bounds[i]),
                             "attack_share": float(1 - is_normal[s].mean()),
                             "n_src": int(len(np.unique(d["src"][s]))),
                             "attacks": {str(k): int(v) for k, v in vc.items()},
                             "stride": int(stride[i])})
    ttl = d["ttl"]
    top = pd.Series(ttl[~is_normal]).value_counts().index[:2].tolist()
    prof["ttl_separation"] = {"attack_top_min_ttl": [int(v) for v in top],
                              "attack_share_in_top": float(np.isin(ttl[~is_normal], top).mean()),
                              "benign_share_in_top": float(np.isin(ttl[is_normal], top).mean())}
    return out, prof


def load_corpus(verbose: bool = True) -> Dict:
    cache = os.path.join(DATA_DIR, "nf_cicids2018_arcfield_corpus.npz")
    if not os.path.exists(cache):
        import json
        out, prof = build_corpus(verbose)
        np.savez_compressed(cache, **out)
        with open(os.path.join(DATA_DIR, "nf_cicids2018_profile.json"), "w") as f:
            json.dump(prof, f, indent=1)
        if verbose:
            print(f"[DATA] NF-CICIDS2018-v3 corpus built: {len(out['X']):,} selected rows")
    d = np.load(cache, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["ctx_names"] = list(out["ctx_names"])
    out["feat_names"] = list(out["feat_names"])
    if verbose:
        print(f"[DATA] NF-CICIDS2018-v3 corpus from cache: {len(out['X']):,} rows, "
              f"{len(out['day_n'])} days")
    return out


def cross_day(corpus: Dict, seed: int, n_train: int = 64000, n_val: int = 5000):
    """Protocol V. Fit on benign flows of the first capture day; stream every
    later day separately in emission order, systematically thinned to about
    100k flows (offset set by the seed). Returns (train, val, cal, {day: idx})."""
    if seed >= N_SEEDS:
        raise ValueError(f"test offsets are stored for seeds 0..{N_SEEDS - 1}")
    rng = np.random.default_rng(seed)
    pool = np.where(corpus["role"] == 0)[0].copy()
    rng.shuffle(pool)
    train, val = np.sort(pool[:n_train]), np.sort(pool[n_train:n_train + n_val])
    cal = np.where(corpus["role"] == 1)[0]
    tests = {}
    for d in range(1, len(corpus["day_n"])):
        s = corpus["stride"][d]
        m = (corpus["role"] == 2) & (corpus["day"] == d) & (corpus["rank"] % s == seed % s)
        tests[d] = np.where(m)[0]
    return train, val, cal, tests


if __name__ == "__main__":
    import json
    c = load_corpus()
    print(json.dumps(json.load(open(os.path.join(DATA_DIR, "nf_cicids2018_profile.json"))), indent=1)[:4000])
