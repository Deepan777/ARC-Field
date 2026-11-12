"""CIC-IDS2017 (MachineLearningCVE release) corpus and cross-day protocol.

This release carries 77 flow-statistic columns, the destination port and the label,
but no endpoint addresses and no timestamps. ARC-Field is therefore
instantiated with service-level context (destination port as the endpoint
key) and the within-day row index as the clock.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from paths import CIC_RAW_DIR, DATA_DIR
import context as ctx

FILES = [
    ("Monday-WorkingHours.pcap_ISCX.csv", 0),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", 1),
    ("Wednesday-workingHours.pcap_ISCX.csv", 2),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", 3),
    ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", 3),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", 4),
    ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 4),
    ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 4),
]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

GROUPS = {
    "forward": ["Total Fwd Packets", "Total Length of Fwd Packets", "Fwd Packet Length Max",
                "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
                "Fwd Header Length", "Fwd Packets/s"],
    "backward": ["Total Backward Packets", "Total Length of Bwd Packets", "Bwd Packet Length Max",
                 "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
                 "Bwd Header Length", "Bwd Packets/s"],
    "timing": ["Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
               "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Mean",
               "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"],
    "aggregate": ["Flow Bytes/s", "Flow Packets/s", "Min Packet Length", "Max Packet Length",
                  "Packet Length Mean", "Packet Length Std", "Fwd PSH Flags"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
HEAVY = set(FEATURES) - {"Fwd PSH Flags"}


def groups_for(features):
    return [[features.index(f) for f in g] for g in GROUPS.values()]


def load_corpus(verbose: bool = True) -> Dict:
    cache = os.path.join(DATA_DIR, "cic_arcfield_corpus.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        out = {k: d[k] for k in d.files}
        if verbose:
            print(f"[DATA] CIC corpus from cache: {len(out['label']):,} flows")
        return out
    Xs, lab, day, port, pos = [], [], [], [], []
    for fname, di in FILES:
        df = pd.read_csv(os.path.join(CIC_RAW_DIR, fname), encoding="utf-8",
                         encoding_errors="replace", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        X = np.zeros((len(df), len(FEATURES)), dtype=np.float32)
        for j, f in enumerate(FEATURES):
            X[:, j] = pd.to_numeric(df[f], errors="coerce").fillna(0.0).to_numpy(np.float64).clip(-1e12, 1e12)
        Xs.append(np.nan_to_num(X, nan=0.0, posinf=1e12, neginf=-1e12))
        L = df["Label"].astype(str).str.strip().to_numpy()
        L = np.array(["Normal" if l.upper() == "BENIGN" else " ".join(l.replace("�", "-").split())
                      for l in L], dtype=object)
        lab.append(L)
        day.append(np.full(len(df), di, dtype=np.int16))
        port.append(pd.to_numeric(df["Destination Port"], errors="coerce").fillna(-1).to_numpy(np.int64))
        if verbose:
            print(f"  {fname}: {len(df):,}")
    X = np.concatenate(Xs)
    label = np.concatenate(lab).astype(str)
    day = np.concatenate(day)
    port = np.concatenate(port)
    # within-day row index as the clock; days are separated by a large offset
    pos = np.zeros(len(day), dtype=np.float64)
    for d in np.unique(day):
        m = np.where(day == d)[0]
        pos[m] = np.arange(len(m)) + d * 1e8
    fi = {f: i for i, f in enumerate(FEATURES)}
    C, names = ctx.cic_context(port, pos,
                               X[:, fi["Total Length of Fwd Packets"]].astype(np.float64),
                               X[:, fi["Total Backward Packets"]].astype(np.float64),
                               X[:, fi["Flow Duration"]].astype(np.float64))
    out = {"X": X, "C": C, "ctx_names": np.array(names), "label": label, "day": day,
           "dport": port, "pos": pos, "is_normal": label == "Normal"}
    np.savez_compressed(cache, **out)
    if verbose:
        print(f"[DATA] CIC corpus built: {len(X):,} flows, context dim {C.shape[1]}")
    return out


def cross_day(corpus: Dict, seed: int, n_train: int = 64000, n_val: int = 5000,
              n_cal: int = 5000, per_day_test: int = 80000):
    """Fit on Monday (attack-free); stream each of Tuesday-Friday separately,
    each in capture-row order and systematically thinned to about
    `per_day_test` flows. Returns (train, val, cal, {day: test_idx})."""
    rng = np.random.default_rng(seed)
    mon = np.where((corpus["day"] == 0) & corpus["is_normal"])[0]
    cal = mon[-n_cal:]
    pool = mon[:-n_cal].copy()
    rng.shuffle(pool)
    train, val = np.sort(pool[:n_train]), np.sort(pool[n_train:n_train + n_val])
    tests = {}
    for d in (1, 2, 3, 4):
        idx = np.where(corpus["day"] == d)[0]
        stride = max(1, len(idx) // per_day_test)
        tests[d] = idx[(seed % stride)::stride]
    return train, val, cal, tests


if __name__ == "__main__":
    c = load_corpus()
    print(c["C"].shape, list(c["ctx_names"]))
    for d in range(5):
        m = c["day"] == d
        print(DAY_NAMES[d], m.sum(), 1 - c["is_normal"][m].mean())
