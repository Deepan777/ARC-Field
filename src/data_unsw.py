"""UNSW-NB15 corpus, feature sets, context and evaluation protocols."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from paths import DATA_DIR
import context as ctx

# ---------------------------------------------------------------------------
# Feature sets. Tiers follow the observability audit of GraphINR-IDS:
#   A  outer IP header + packet timing only (survives TLS, QUIC and tunnels)
#   B  needs a readable TCP header        (survives TLS over TCP only)
#   C  needs plaintext payload             (survives no encryption)
# ---------------------------------------------------------------------------

TIER_B = ["tcprtt", "synack", "ackdat", "swin", "dwin", "sloss", "dloss"]
TIER_C = ["trans_depth", "res_bdy_len", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"]
ALL36 = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "Sload",
    "Dload", "Spkts", "Dpkts", "swin", "dwin", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm",
]
TIER_A = [f for f in ALL36 if f not in TIER_B + TIER_C]
TTL_FEATURES = ["sttl", "dttl", "ct_state_ttl"]
TIER_A_NOTTL = [f for f in TIER_A if f not in TTL_FEATURES]

FEATURE_SETS = {
    "A": TIER_A,                  # default: encryption-observable (24)
    "A_noTTL": TIER_A_NOTTL,      # shortcut audit: TTL-derived features removed (21)
    "AB": [f for f in ALL36 if f not in TIER_C],   # TLS over TCP (31)
    "ABC": ALL36,                 # full benchmark set (36)
}

# Observability groups used by the masked conditional field. Each group is a
# physically coherent view of the flow; the field predicts each view from the
# others plus context.
GROUPS = {
    "volume": ["sbytes", "dbytes", "Spkts", "Dpkts", "smeansz", "dmeansz", "Sload", "Dload"],
    "timing": ["dur", "Sjit", "Djit", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat"],
    "header": ["sttl", "dttl", "ct_state_ttl", "is_sm_ips_ports", "swin", "dwin", "sloss", "dloss"],
    "connection": ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                   "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"],
    "application": ["trans_depth", "res_bdy_len", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"],
}

# Features whose raw scale spans orders of magnitude; log-compressed.
HEAVY = {"dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts", "smeansz",
         "dmeansz", "Sjit", "Djit", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
         "res_bdy_len", "sloss", "dloss", "swin", "dwin", "trans_depth"}


def groups_for(features: List[str]) -> List[List[int]]:
    out = []
    for g, members in GROUPS.items():
        idx = [features.index(f) for f in members if f in features]
        if idx:
            out.append(idx)
    covered = sorted(i for g in out for i in g)
    assert covered == list(range(len(features))), "every feature must be in a group"
    return out


def group_names_for(features: List[str]) -> List[str]:
    return [g for g, m in GROUPS.items() if any(f in features for f in m)]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_corpus(verbose: bool = True) -> Dict:
    """Full UNSW-NB15 (2.54 M flows) in emission order, with role context.

    Emission order is (Ltime, Stime, original position): a flow record exists
    only once the flow has ended, so the context of a record may use every
    record emitted before it and nothing after it.
    """
    cache = os.path.join(DATA_DIR, "unsw_arcfield_corpus.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        out = {k: d[k] for k in d.files}
        out["ctx_names"] = list(out["ctx_names"])
        out["feat_names"] = list(out["feat_names"])
        if verbose:
            print(f"[DATA] UNSW corpus from cache: {len(out['label']):,} flows")
        return out

    pq = os.path.join(DATA_DIR, "unsw_full.parquet")
    if not os.path.exists(pq):
        raise FileNotFoundError("run code/tools/profile_unsw.py first")
    df = pd.read_parquet(pq)
    df["_pos"] = np.arange(len(df))
    df = df.sort_values(["Ltime", "Stime", "_pos"], kind="stable").reset_index(drop=True)

    ips, codes = np.unique(np.concatenate([df["srcip"].to_numpy(), df["dstip"].to_numpy()]),
                           return_inverse=True)
    src = codes[: len(df)].astype(np.int64)
    dst = codes[len(df):].astype(np.int64)
    proto_map = {"tcp": 0, "udp": 1}
    proto = np.array([proto_map.get(p, 2) for p in df["proto"].to_numpy()], dtype=np.int64)
    t = df["Ltime"].to_numpy().astype(np.float64)

    C, cnames = ctx.unsw_context(
        src=src, dst=dst, dport=df["dsport"].to_numpy().astype(np.int64), t=t,
        sbytes=df["sbytes"].to_numpy(np.float64), dbytes=df["dbytes"].to_numpy(np.float64),
        dur=df["dur"].to_numpy(np.float64), proto=proto)

    X = df[ALL36].to_numpy(np.float64).astype(np.float32)
    out = {
        "X": X, "feat_names": np.array(ALL36), "C": C, "ctx_names": np.array(cnames),
        "src": src, "dst": dst, "ips": ips, "dport": df["dsport"].to_numpy().astype(np.int64),
        "proto": proto, "stime": df["Stime"].to_numpy().astype(np.int64),
        "ltime": df["Ltime"].to_numpy().astype(np.int64),
        "label": df["attack_cat"].to_numpy().astype(str),
        "is_normal": df["is_normal"].to_numpy().astype(bool),
    }
    np.savez_compressed(cache, **out)
    out["ctx_names"] = cnames
    out["feat_names"] = list(ALL36)
    if verbose:
        print(f"[DATA] UNSW corpus built: {len(X):,} flows, context dim {C.shape[1]}")
    return out


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

@dataclass
class Protocol:
    name: str
    train: np.ndarray          # benign, fitting
    val: np.ndarray            # benign, residual statistics
    cal: np.ndarray            # benign, threshold
    test: np.ndarray           # stream, in emission order (benign + attack)
    meta: Dict = field(default_factory=dict)
    ip_remap: Optional[Dict[int, int]] = None   # test-time endpoint renumbering


def session_cut(corpus: Dict) -> int:
    t = np.unique(corpus["stime"])
    return int(t[int(np.argmax(np.diff(t)))])


def chronological(corpus: Dict, seed: int, n_train: int = 64000, n_val: int = 5000,
                  n_cal: int = 5000, test_stride: int = 4) -> Protocol:
    """Protocol II. Fit on benign traffic of campaign 1 (22 Jan 2015); stream
    campaign 2 (18 Feb 2015, +26 days) in emission order.

    Fitting partitions are drawn at random from the benign flows of campaign 1
    (seeded) and are mutually disjoint; `cal` is the last benign block of the
    campaign so that the threshold is estimated on the most recent data.
    The test stream is a systematic 1-in-`test_stride` subsample of campaign 2
    (offset set by the seed), which preserves emission order and hour-level
    composition while keeping every baseline tractable.
    """
    rng = np.random.default_rng(seed)
    cut = session_cut(corpus)
    s1 = corpus["stime"] <= cut
    ben1 = np.where(s1 & corpus["is_normal"])[0]
    cal = ben1[-n_cal:]
    pool = ben1[:-n_cal].copy()
    rng.shuffle(pool)
    train = np.sort(pool[:n_train])
    val = np.sort(pool[n_train:n_train + n_val])
    s2 = np.where(~s1)[0]
    test = s2[(seed % test_stride)::test_stride]
    return Protocol(f"unsw_chrono_s{seed}", train, val, cal, test,
                    meta={"cut": cut, "seed": seed, "stride": test_stride})


CLIENTS = [f"59.166.0.{i}" for i in range(10)]


def host_disjoint(corpus: Dict, seed: int, n_train: int = 64000, n_val: int = 5000,
                  n_cal: int = 5000, n_test: int = 120000) -> Protocol:
    """Protocol III (inductive). The ten benign client hosts 59.166.0.0-9 are
    split 5/5 by the seed. Fitting uses campaign-1 benign flows that do not
    touch a held-out client. The test stream is every other campaign-1 flow
    that touches a held-out client or is an attack, kept in emission order and
    systematically thinned to about `n_test` flows, so its composition over
    time is the natural one. Time period and network are unchanged; only
    endpoint identity is novel.
    """
    rng = np.random.default_rng(seed)
    ips = list(corpus["ips"])
    code = {ip: i for i, ip in enumerate(ips)}
    clients = np.array([code[c] for c in CLIENTS if c in code])
    held = rng.choice(clients, size=len(clients) // 2, replace=False)
    cut = session_cut(corpus)
    s1 = corpus["stime"] <= cut
    touches_held = np.isin(corpus["src"], held) | np.isin(corpus["dst"], held)
    ben = np.where(s1 & corpus["is_normal"] & ~touches_held)[0]
    rng.shuffle(ben)
    need = n_train + n_val + n_cal
    train, val, cal = (np.sort(ben[:n_train]), np.sort(ben[n_train:n_train + n_val]),
                       np.sort(ben[n_train + n_val:need]))
    pool = np.where(s1 & (touches_held | ~corpus["is_normal"]))[0]
    stride = max(1, len(pool) // n_test)
    test = pool[(seed % stride)::stride]
    return Protocol(f"unsw_hostdisjoint_s{seed}", train, val, cal, test,
                    meta={"seed": seed, "held_out_clients": [ips[h] for h in held],
                          "stride": stride})


def random_split(corpus: Dict, seed: int, n_train: int = 64000, n_val: int = 5000,
                 n_cal: int = 5000, n_test: int = 120000) -> Protocol:
    """Protocol I (in-distribution reference): campaign 1 only. Benign fitting
    partitions are drawn at random; the test stream is every remaining
    campaign-1 flow, in emission order, systematically thinned."""
    rng = np.random.default_rng(seed)
    cut = session_cut(corpus)
    s1 = np.where(corpus["stime"] <= cut)[0]
    ben = s1[corpus["is_normal"][s1]].copy()
    rng.shuffle(ben)
    need = n_train + n_val + n_cal
    used = np.zeros(len(corpus["stime"]), dtype=bool)
    used[ben[:need]] = True
    pool = s1[~used[s1]]
    stride = max(1, len(pool) // n_test)
    return Protocol(f"unsw_random_s{seed}", np.sort(ben[:n_train]),
                    np.sort(ben[n_train:n_train + n_val]),
                    np.sort(ben[n_train + n_val:need]),
                    pool[(seed % stride)::stride], meta={"seed": seed, "stride": stride})


def development(corpus: Dict, seed: int = 0, n_train: int = 64000, n_val: int = 5000,
                n_cal: int = 5000, test_stride: int = 1, head_hours: float = 2.0) -> Protocol:
    """Protocol D (hyper-parameter selection only; never reported as a result).

    All campaign-1 attacks occur in its first two hours. Benign flows emitted
    after those hours are used for fitting, and the first two hours (8-15 %
    attack traffic per half hour) are streamed. Every calibrator and model
    hyper-parameter is chosen on this protocol and frozen before Protocols
    I-IV are run, so no reported number is tuned on its own test stream.
    """
    rng = np.random.default_rng(seed)
    cut = session_cut(corpus)
    s1 = np.where(corpus["stime"] <= cut)[0]
    t = corpus["ltime"][s1]
    split = t.min() + head_hours * 3600
    head = s1[t < split]
    tail = s1[t >= split]
    ben = tail[corpus["is_normal"][tail]]
    cal = ben[-n_cal:]
    pool = ben[:-n_cal].copy()
    rng.shuffle(pool)
    return Protocol(f"unsw_dev_s{seed}", np.sort(pool[:n_train]),
                    np.sort(pool[n_train:n_train + n_val]), cal,
                    head[(seed % test_stride)::test_stride], meta={"seed": seed})

if __name__ == "__main__":
    c = load_corpus()
    print(c["C"].shape, len(c["ctx_names"]))
    print(dict(zip(c["ctx_names"], np.round(c["C"][:200000].mean(0), 3))))
