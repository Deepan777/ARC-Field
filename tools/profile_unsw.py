"""Build the full UNSW-NB15 flow corpus (all 2.54M records, all four files)
and profile its topology and temporal structure.

The GraphINR-IDS pipeline read only the first 150k rows of each file and kept
IP addresses but not ports, protocol, service or the flow end time. The
extension needs all of these to compute causal endpoint context, so the corpus
is rebuilt here. Output: data_processed/unsw_full.parquet (+ a JSON profile).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import UNSW_RAW_DIR, DATA_DIR  # noqa: E402

RAW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label",
]
STR_COLS = {"srcip", "dstip", "proto", "state", "service", "attack_cat",
            "sport", "dsport"}


def canon_label(v) -> str:
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "normal", "benign", "0"}:
        return "Normal"
    s = " ".join(s.split())
    return {"Backdoors": "Backdoor"}.get(s, s)


def to_port(s: pd.Series) -> np.ndarray:
    # A handful of UNSW ports are hex strings ("0x000b") or '-'.
    def conv(x):
        x = str(x).strip()
        if x.startswith("0x"):
            try:
                return int(x, 16)
            except ValueError:
                return -1
        try:
            return int(float(x))
        except ValueError:
            return -1
    return np.array([conv(x) for x in s.to_numpy()], dtype=np.int32)


def build() -> pd.DataFrame:
    out = os.path.join(DATA_DIR, "unsw_full.parquet")
    if os.path.exists(out):
        return pd.read_parquet(out)
    parts = []
    for i in range(1, 5):
        p = os.path.join(UNSW_RAW_DIR, f"UNSW-NB15_{i}.csv")
        df = pd.read_csv(p, header=None, names=RAW_COLS, encoding="latin-1",
                         low_memory=False, dtype=str)
        if str(df.iloc[0]["srcip"]).strip() == "srcip":
            df = df.iloc[1:].reset_index(drop=True)
        df["srcip"] = df["srcip"].str.strip().str.lstrip("﻿").str.lstrip("ï»¿")
        df["dstip"] = df["dstip"].str.strip()
        df["file"] = i
        parts.append(df)
        print(f"loaded file {i}: {len(df):,}")
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["sport"] = to_port(df["sport"])
    df["dsport"] = to_port(df["dsport"])
    for c in RAW_COLS:
        if c in STR_COLS:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["proto", "state", "service"]:
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["attack_cat"] = [canon_label(v) for v in df["attack_cat"].to_numpy()]
    df["is_normal"] = df["attack_cat"] == "Normal"
    df["Stime"] = df["Stime"].astype(np.int64)
    df["Ltime"] = df["Ltime"].astype(np.int64)
    # Stable capture order: start time, then file order as tie-breaker.
    df["row_in_file"] = df.groupby("file").cumcount()
    df = df.sort_values(["Stime", "file", "row_in_file"], kind="stable").reset_index(drop=True)
    df.to_parquet(out, index=False)
    return df


def profile(df: pd.DataFrame) -> dict:
    t = df["Stime"].to_numpy()
    order = np.unique(t)
    gaps = np.diff(order)
    cut = int(order[int(np.argmax(gaps))])
    s1 = df[df["Stime"] <= cut]
    s2 = df[df["Stime"] > cut]
    ips1 = set(s1["srcip"]).union(s1["dstip"])
    ips2 = set(s2["srcip"]).union(s2["dstip"])
    ben1 = s1[s1["is_normal"]]
    ben2 = s2[s2["is_normal"]]
    bips1 = set(ben1["srcip"]).union(ben1["dstip"])
    bips2 = set(ben2["srcip"]).union(ben2["dstip"])
    prof = {
        "n_flows": int(len(df)),
        "n_benign": int(df["is_normal"].sum()),
        "n_attack": int((~df["is_normal"]).sum()),
        "attack_families": df.loc[~df["is_normal"], "attack_cat"].value_counts().to_dict(),
        "n_ips": int(len(set(df["srcip"]).union(df["dstip"]))),
        "n_src_ips": int(df["srcip"].nunique()),
        "n_dst_ips": int(df["dstip"].nunique()),
        "n_services": int(df["service"].nunique()),
        "n_dst_ports": int(df["dsport"].nunique()),
        "session_cut_epoch": cut,
        "session_gap_hours": float(gaps.max() / 3600.0),
        "session1": {"n": int(len(s1)), "benign": int(s1["is_normal"].sum()),
                     "t0": int(s1["Stime"].min()), "t1": int(s1["Stime"].max()),
                     "hours": float((s1["Stime"].max() - s1["Stime"].min()) / 3600),
                     "n_ips": len(ips1), "n_benign_ips": len(bips1)},
        "session2": {"n": int(len(s2)), "benign": int(s2["is_normal"].sum()),
                     "t0": int(s2["Stime"].min()), "t1": int(s2["Stime"].max()),
                     "hours": float((s2["Stime"].max() - s2["Stime"].min()) / 3600),
                     "n_ips": len(ips2), "n_benign_ips": len(bips2)},
        "ips_only_in_s2": sorted(ips2 - ips1),
        "benign_ips_only_in_s2": sorted(bips2 - bips1),
        "benign_src_ips_s1": ben1["srcip"].value_counts().head(40).to_dict(),
        "attack_src_ips": df.loc[~df["is_normal"], "srcip"].value_counts().head(20).to_dict(),
        "attack_frac_by_hour_s2": (s2.assign(h=(s2["Stime"] - s2["Stime"].min()) // 3600)
                                   .groupby("h")["is_normal"].apply(lambda x: float(1 - x.mean()))
                                   .to_dict()),
        "proto_counts": df["proto"].value_counts().head(15).to_dict(),
    }
    return prof


if __name__ == "__main__":
    df = build()
    prof = profile(df)
    path = os.path.join(DATA_DIR, "unsw_full_profile.json")
    with open(path, "w") as f:
        json.dump(prof, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in prof.items()
                      if k not in ("attack_frac_by_hour_s2",)}, indent=1, default=str)[:6000])
