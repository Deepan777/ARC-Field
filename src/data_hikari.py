"""HIKARI-2021 (Ferriyan et al., Applied Sciences 11(17):7868, 2021): encrypted
application-layer attacks over HTTPS (browser brute force, XML-RPC brute
force, CMS vulnerability probing) and a cryptominer found in real background
traffic, plus real background traffic and synthetic benign browsing. Flow
table ALLFLOWMETER_HIKARI2021.csv of the authors' Zenodo release (record
5199540, v1.3.5, CC BY 4.0, MD5 d7d9e277fe4a66cb00764d7f91a810dd).

The public flow table has no timestamps and stores each traffic category as
a separate block (its per-file index restarts at every block). No causal
stream can be reconstructed from it, so endpoint context (role coordinates,
AfterImage, ReFeX) cannot be computed and recalibration windows have no
natural order. Protocol VI therefore evaluates flow-level detection on
genuinely encrypted traffic, with test streams in a seeded random order:
  VI-a  random: fit on benign flows of both benign sources, test on the rest
  VI-b  source shift: fit on the real background traffic only, test on the
        remaining background, the synthetic benign browsing and the attacks
Feature tiers follow the same observability logic as UNSW-NB15 and NetFlow:
Tier A needs only packet sizes, counts and timing (survives TLS, QUIC and
tunnels); Tier B adds TCP header fields (flags, header lengths, windows).
"""
from __future__ import annotations

import os
import zipfile
from typing import Dict, List

import numpy as np

from paths import DATA_DIR, PROJECT

HIKARI_ZIP = os.environ.get("ARC_HIKARI_ZIP", os.path.join(PROJECT, "data_raw", "HIKARI-2021",
                                                           "ALLFLOWMETER_HIKARI2021.csv.zip"))
_ST = ("min", "max", "tot", "avg", "std")
VOLUME = (["fwd_pkts_tot", "bwd_pkts_tot", "fwd_data_pkts_tot", "bwd_data_pkts_tot", "down_up_ratio"]
          + [f"{d}_pkts_payload.{s}" for d in ("fwd", "bwd", "flow") for s in _ST]
          + ["fwd_subflow_pkts", "bwd_subflow_pkts", "fwd_subflow_bytes", "bwd_subflow_bytes",
             "fwd_bulk_bytes", "bwd_bulk_bytes", "fwd_bulk_packets", "bwd_bulk_packets",
             "fwd_bulk_rate", "bwd_bulk_rate"])
TIMING = (["flow_duration", "fwd_pkts_per_sec", "bwd_pkts_per_sec", "flow_pkts_per_sec",
           "payload_bytes_per_second"]
          + [f"{d}_iat.{s}" for d in ("fwd", "bwd", "flow") for s in _ST]
          + [f"{a}.{s}" for a in ("active", "idle") for s in _ST])
TIER_B = (["fwd_header_size_tot", "fwd_header_size_min", "fwd_header_size_max",
           "bwd_header_size_tot", "bwd_header_size_min", "bwd_header_size_max"]
          + [f"flow_{f}_flag_count" for f in ("FIN", "SYN", "RST", "ACK", "CWR", "ECE")]
          + ["fwd_PSH_flag_count", "bwd_PSH_flag_count", "fwd_URG_flag_count", "bwd_URG_flag_count",
             "fwd_init_window_size", "bwd_init_window_size", "fwd_last_window_size"])
TIER_A = VOLUME + TIMING
ALL = TIER_A + TIER_B
FEATURE_SETS = {"A": TIER_A, "AB": ALL}
GROUPS = {"volume": VOLUME, "timing": TIMING, "tcp": TIER_B}
HEAVY = set(VOLUME + TIMING + ["fwd_header_size_tot", "bwd_header_size_tot"])
BENIGN = ("Background", "Benign")


def groups_for(features: List[str]) -> List[List[int]]:
    out = []
    for members in GROUPS.values():
        idx = [features.index(f) for f in members if f in features]
        if idx:
            out.append(idx)
    return out


def build_corpus(verbose: bool = True) -> Dict:
    import pandas as pd
    z = zipfile.ZipFile(HIKARI_ZIP)
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, usecols=["Unnamed: 0", "originh", "originp", "responh", "responp",
                                     "traffic_category", "Label"] + ALL, low_memory=False)
    u = df["Unnamed: 0"].to_numpy()
    block = np.zeros(len(df), np.int16)
    block[np.where(np.diff(u) < 0)[0] + 1] = 1
    block = np.cumsum(block).astype(np.int16)
    # fixed-width unicode, never object arrays: the cache is read with
    # allow_pickle=False
    ips, codes = np.unique(np.concatenate([df["originh"].to_numpy().astype("U"),
                                           df["responh"].to_numpy().astype("U")]), return_inverse=True)
    X = df[ALL].to_numpy(np.float64).clip(-1e12, 1e12).astype(np.float32)
    cat = df["traffic_category"].to_numpy().astype("U")
    out = {"X": X, "feat_names": np.array(ALL), "src": codes[:len(df)].astype(np.int64),
           "dst": codes[len(df):].astype(np.int64), "dport": df["responp"].to_numpy(np.int64),
           "ips": ips, "label": np.where(np.isin(cat, BENIGN), "Normal", cat).astype(str),
           "category": cat, "is_normal": (df["Label"].to_numpy() == 0), "block": block}
    assert np.array_equal(out["is_normal"], np.isin(cat, BENIGN)), "Label and traffic_category disagree"
    if verbose:
        print(f"[DATA] HIKARI-2021: {len(df):,} flows, {len(ips):,} addresses, {int(block.max()) + 1} blocks; "
              + ", ".join(f"{k} {v:,}" for k, v in pd.Series(cat).value_counts().items()), flush=True)
    return out


def load_corpus(verbose: bool = True) -> Dict:
    cache = os.path.join(DATA_DIR, "hikari2021_corpus.npz")
    if not os.path.exists(cache):
        out = build_corpus(verbose)
        np.savez_compressed(cache[:-4] + ".tmp.npz", **out)
        os.replace(cache[:-4] + ".tmp.npz", cache)
    d = np.load(cache, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["feat_names"] = list(out["feat_names"])
    if verbose:
        print(f"[DATA] HIKARI-2021 corpus from cache: {len(out['X']):,} flows", flush=True)
    return out


def split(corpus: Dict, setting: str, seed: int, n_train: int = 64000, n_val: int = 5000,
          n_cal: int = 5000, n_test: int = 120000):
    """Protocol VI-a ('random') or VI-b ('shift'). Returns (train, val, cal,
    test); the test stream is in a seeded random order and thinned to about
    n_test flows (the categories keep their natural proportions)."""
    rng = np.random.default_rng(seed)
    cat = corpus["category"]
    fit_src = np.isin(cat, BENIGN) if setting == "random" else (cat == "Background")
    pool = np.where(fit_src)[0]
    rng.shuffle(pool)
    need = n_train + n_val + n_cal
    train, val, cal = np.sort(pool[:n_train]), np.sort(pool[n_train:n_train + n_val]), pool[n_train + n_val:need]
    used = np.zeros(len(cat), bool)
    used[pool[:need]] = True
    rest = np.where(~used)[0]
    rng.shuffle(rest)
    stride = max(1, len(rest) // n_test)
    test = rest[::stride]
    return train, val, np.sort(cal), test
