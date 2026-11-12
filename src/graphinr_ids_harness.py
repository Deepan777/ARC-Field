"""GraphINR-IDS harness: its transductive spectral endpoint coordinates and
absolute-time normalisation (used for the GraphINR-IDS baseline and as
comparators in the ablation and churn studies), its min-max preprocessing,
published configuration, training loop and residual extraction, and the
settings of its published UNSW-NB15 cross-session protocol (read by
analysis/tables_extra.py; the loader is not called). The model itself is in
third_party/graphinr_ids/. Reused from P. Deepanramkumar and A. Helen Sharmila, GraphINR-IDS, J. King Saud
Univ. - Comput. Inf. Sci. 38(8):916, 2026, doi:10.1007/s44443-026-01260-x."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

UNSW_FEATURES = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "Sload",
    "Dload", "Spkts", "Dpkts", "swin", "dwin", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm",
]


def load_unsw_corpus(rows_per_file: int = 150000, verbose: bool = True) -> Dict:
    """Load the UNSW-NB15 corpus with REAL capture timestamps (`Stime`).

    Matches the original loader's row budget (150k rows per raw file) so the
    corpus is the same one the published results were computed on, except that
    `Stime` is now carried through instead of being replaced by a synthetic
    index ramp.
    """
    cache = os.path.join(CACHE_DIR, f"unsw_corpus_{rows_per_file}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"[DATA] UNSW corpus from cache: {d['features'].shape[0]:,} flows")
        return {k: d[k] for k in d.files}

    feats, labels, srcs, dsts, stimes = [], [], [], [], []
    for i in range(1, 5):
        path = os.path.join(UNSW_DIR, f"UNSW-NB15_{i}.csv")
        df = pd.read_csv(path, header=None, names=UNSW_RAW_COLS,
                         encoding="latin-1", low_memory=False, nrows=rows_per_file)
        if str(df.iloc[0]["srcip"]).strip() == "srcip":
            df = df.iloc[1:].reset_index(drop=True)
        # The first cell of file 1 carries a UTF-8 BOM.
        df["srcip"] = df["srcip"].astype(str).str.strip().str.lstrip("﻿")
        df["dstip"] = df["dstip"].astype(str).str.strip()

        m = np.zeros((len(df), len(UNSW_FEATURES)), dtype=np.float32)
        for j, c in enumerate(UNSW_FEATURES):
            m[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        feats.append(np.nan_to_num(m, nan=0.0, posinf=1e10, neginf=-1e10))
        labels.append(np.array([_canon_unsw_label(v) for v in df["attack_cat"].to_numpy()], dtype=object))
        srcs.append(df["srcip"].to_numpy().astype(object))
        dsts.append(df["dstip"].to_numpy().astype(object))
        stimes.append(pd.to_numeric(df["Stime"], errors="coerce").fillna(0).to_numpy(dtype=np.float64))
        if verbose:
            print(f"  loaded UNSW-NB15_{i}.csv: {len(df):,} rows")
        del df

    out = {
        "features": np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "src_ips": np.concatenate(srcs, axis=0),
        "dst_ips": np.concatenate(dsts, axis=0),
        "stime": np.concatenate(stimes, axis=0),
    }
    out["is_normal"] = (out["labels"] == "Normal")
    np.savez_compressed(cache, **out)
    if verbose:
        print(f"[DATA] UNSW corpus built: {out['features'].shape[0]:,} flows, "
              f"{int(out['is_normal'].sum()):,} benign")
    return out


@dataclass
class Split:
    """A concrete train / validation / calibration / test partition."""
    name: str
    train: np.ndarray            # benign only
    val: np.ndarray              # benign only, fits residual statistics
    cal: np.ndarray              # benign only, fixes the threshold
    test_normal: np.ndarray
    test_attack: np.ndarray
    protocol: str = "random"
    meta: Dict = field(default_factory=dict)

    @property
    def test(self) -> np.ndarray:
        return np.concatenate([self.test_normal, self.test_attack])


def make_chronological_split(corpus: Dict, seed: int = 0,
                             n_test_normal: int = 20000,
                             n_test_attack: int = 40000,
                             n_train: int = 64000, n_val: int = 5000,
                             n_cal: int = 5000) -> Split:
    """Cross-session split for UNSW-NB15.

    The corpus was captured in two campaigns separated by a 630-hour gap
    (22 Jan 2015 and 17-18 Feb 2015). The detector is fitted entirely on benign
    traffic from the first campaign and tested on the second, so every test
    timestamp post-dates the training window by roughly 26 days. This is the
    deployment-realistic protocol requested in review and is also the setting in
    which temporal extrapolation of the time encoding is exercised.
    """
    rng = np.random.default_rng(seed)
    t = corpus["stime"]
    is_norm = corpus["is_normal"]

    order = np.sort(np.unique(t))
    gaps = np.diff(order)
    cut = float(order[int(np.argmax(gaps))])          # largest inter-session gap

    early = np.where((t <= cut) & is_norm)[0]
    early = early[np.argsort(t[early], kind="stable")]
    # The three fitting partitions stay strictly ordered in time and are capped
    # at the Protocol I sizes so that the two protocols differ only in how the
    # partition is drawn, not in how much data the detector is given.
    n = len(early)
    n_tr = min(n_train, int(n * 0.75))
    n_va = min(n_val, int(n * 0.12))
    n_ca = min(n_cal, n - n_tr - n_va)
    train = early[:n_tr]
    val = early[n_tr:n_tr + n_va]
    cal = early[n_tr + n_va:n_tr + n_va + n_ca]

    late_n = np.where((t > cut) & is_norm)[0]
    late_a = np.where((t > cut) & (~is_norm))[0]
    rng.shuffle(late_n)
    rng.shuffle(late_a)

    return Split(
        name="unsw_chronological", train=train, val=val, cal=cal,
        test_normal=late_n[:n_test_normal], test_attack=late_a[:n_test_attack],
        protocol="chronological (session 1 -> session 2, +26 days)",
        meta={"session_cut_epoch": cut, "seed": seed},
    )


class RevGraph:
    """Communication graph of the fitting flows and its spectral endpoint
    coordinates (smallest non-trivial eigenvectors of the normalised
    Laplacian, sign-fixed). Node index 0 is reserved for endpoints absent from
    the fitting graph, whose coordinate is the zero vector (transductive
    fallback)."""

    OOV = 0

    def __init__(self, spectral_k: int = 32, max_nodes: int = 10000):
        self.k = spectral_k
        self.max_nodes = max_nodes
        self.ip_to_idx: Dict[str, int] = {}
        self.eigenvectors: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.num_nodes = 0
        self.edge_index: Optional[np.ndarray] = None

    def fit(self, src: np.ndarray, dst: np.ndarray, verbose: bool = True):
        ips = np.unique(np.concatenate([src, dst]))
        if len(ips) > self.max_nodes - 1:
            cnt: Dict[str, int] = {}
            for ip in np.concatenate([src, dst]):
                cnt[ip] = cnt.get(ip, 0) + 1
            ips = np.array(sorted(cnt, key=cnt.get, reverse=True)[: self.max_nodes - 1])
        # slot 0 reserved for out-of-vocabulary endpoints
        self.ip_to_idx = {ip: i + 1 for i, ip in enumerate(ips)}
        self.num_nodes = len(ips) + 1

        counts: Dict[Tuple[int, int], int] = {}
        for a, b in zip(src, dst):
            i, j = self.ip_to_idx.get(a), self.ip_to_idx.get(b)
            if i is not None and j is not None:
                counts[(i, j)] = counts.get((i, j), 0) + 1

        rows, cols, w = [], [], []
        for (i, j), c in counts.items():
            rows += [i, j]
            cols += [j, i]
            w += [float(c), float(c)]
        A = csr_matrix((w, (rows, cols)), shape=(self.num_nodes, self.num_nodes))
        self.edge_index = np.array([[i for i, _ in counts], [j for _, j in counts]],
                                   dtype=np.int64)
        self._spectral(A)
        # OOV node carries the zero coordinate.
        self.eigenvectors[self.OOV, :] = 0.0
        if verbose:
            print(f"[GRAPH] nodes={self.num_nodes - 1} (+1 OOV), "
                  f"edges={len(counts)}, K={self.eigenvectors.shape[1]}")
        return self

    def _spectral(self, A: csr_matrix):
        N = self.num_nodes
        Ad = A.toarray().astype(np.float64)
        Ad = (Ad + Ad.T) / 2.0
        deg = Ad.sum(axis=1)
        deg[deg == 0] = 1.0
        dis = 1.0 / np.sqrt(deg)
        L = np.eye(N) - (dis[:, None] * Ad * dis[None, :])
        L = (L + L.T) / 2.0

        k = min(self.k, max(1, N - 2))
        if N < 512 or N <= k + 2:
            # Dense symmetric eigensolver: exact, and faster than Lanczos at
            # this size. The measured UNSW-NB15 graph has ~50 nodes.
            vals, vecs = np.linalg.eigh(L)
            o = np.argsort(vals)
            vals, vecs = vals[o], vecs[:, o]
        else:
            try:
                vals, vecs = eigsh(csr_matrix(L), k=k + 1, which="SM", tol=1e-6)
                o = np.argsort(vals)
                vals, vecs = vals[o], vecs[:, o]
            except Exception:
                vals, vecs = np.linalg.eigh(L)
                o = np.argsort(vals)
                vals, vecs = vals[o], vecs[:, o]
        upper = min(k + 1, vecs.shape[1])
        vals, vecs = vals[1:upper], vecs[:, 1:upper]
        if vecs.shape[1] == 0:
            vecs = np.zeros((N, 1))
            vals = np.zeros(1)

        if vecs.shape[1] < self.k:
            pad = self.k - vecs.shape[1]
            vecs = np.pad(vecs, ((0, 0), (0, pad)))
            vals = np.pad(vals, (0, pad))
        # Sign disambiguation: largest-magnitude entry made positive.
        for c in range(vecs.shape[1]):
            if vecs[np.argmax(np.abs(vecs[:, c])), c] < 0:
                vecs[:, c] *= -1
        self.eigenvectors = vecs.astype(np.float32)
        self.eigenvalues = vals.astype(np.float32)

    def index(self, ips: np.ndarray) -> np.ndarray:
        return np.array([self.ip_to_idx.get(ip, self.OOV) for ip in ips], dtype=np.int64)

    def oov_mask(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        return np.array([(a not in self.ip_to_idx) or (b not in self.ip_to_idx)
                         for a, b in zip(src, dst)], dtype=bool)


@dataclass
class TimeNormaliser:
    """t_norm = (t - t_min_train) / (t_max_train - t_min_train).

    Training maps to [0, 1] by construction. Test timestamps that post-date the
    training window map to values > 1; the encoder is periodic so it stays
    bounded, and `clip` is available to force the training range when the
    extrapolation behaviour is being isolated.
    """
    t0: float = 0.0
    span: float = 1.0

    @classmethod
    def fit(cls, t_train: np.ndarray) -> "TimeNormaliser":
        t0 = float(np.min(t_train))
        span = float(np.max(t_train) - t0)
        return cls(t0=t0, span=span if span > 0 else 1.0)

    def __call__(self, t: np.ndarray, clip: bool = False) -> np.ndarray:
        z = (np.asarray(t, dtype=np.float64) - self.t0) / self.span
        if clip:
            z = np.clip(z, 0.0, 1.0)
        return z.astype(np.float32)


class MinMax:
    def __init__(self):
        self.lo = None
        self.hi = None
        self.rng = None

    def fit(self, X: np.ndarray) -> "MinMax":
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        r = self.hi - self.lo
        r[r == 0] = 1.0
        self.rng = r
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.lo) / self.rng, 0.0, 1.0).astype(np.float32)


class Flows(torch.utils.data.Dataset):
    def __init__(self, X, s, d, t, y):
        self.X = torch.from_numpy(X).float()
        self.s = torch.from_numpy(s).long()
        self.d = torch.from_numpy(d).long()
        self.t = torch.from_numpy(t).float()
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return {"features": self.X[i], "src_idx": self.s[i], "dst_idx": self.d[i],
                "timestamp": self.t[i], "is_normal": self.y[i]}


def base_config(n_features: int = 36, gcm: bool = True, hfd: bool = True,
                epochs: int = 50, seed: int = 42) -> dict:
    """Published configuration, with the module switches exposed."""
    return {
        "paths": {"output_dir": "logs",
                  "checkpoint_dir": "logs",
                  "log_dir": "logs",
                  "data_root": "data"},
        "graph": {"spectral_k": 32, "max_nodes": 10000, "edge_encoding_dim": 96},
        "temporal": {"num_frequencies": 16, "freq_min": 1.0, "freq_max": 10000.0,
                     "encoding_dim": 32},
        "siren": {"input_dim": 96 + 32 + n_features, "hidden_dim": 256,
                  "hidden_layers": 6, "output_dim": n_features, "omega_0": 30.0},
        "gcm": {"enabled": gcm, "gnn_layers": 2, "gnn_hidden_dim": 128,
                "modulation_type": "scale_shift"},
        "hfd": {"enabled": hfd, "low_omega": 10.0, "high_omega": 60.0,
                "attention_dim": 64},
        "training": {"optimizer": "adamw", "lr": 5.0e-5, "beta1": 0.9,
                     "beta2": 0.999, "weight_decay": 1.0e-4,
                     "scheduler_T0": epochs, "scheduler_Tmult": 2,
                     "batch_size": 1024, "epochs": epochs, "loss": "huber",
                     "huber_delta": 1.0, "lipschitz_reg": 1.0e-4,
                     "compactness_weight": 0.1,
                     "gradient_checkpointing": False, "checkpoint_layers": 2},
        "amp": {"enabled": False, "fp32_loss": True},
        "anomaly": {"temporal_windows": [1, 10, 60], "aggregation": "max_pool"},
        "dataloader": {"num_workers": 0, "pin_memory": False,
                       "prefetch_factor": 4, "persistent_workers": False},
        "hardware": {"device": "cpu", "compile_model": False},
        "meta_learning": {"enabled": False},
        "seed": seed,
    }


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict, graph: RevGraph, device: torch.device):
    from graphinr_ids.models.graphinr_ids import GraphINRIDS
    m = GraphINRIDS(cfg, num_nodes=graph.num_nodes).to(device)
    m.set_graph_data(eigenvectors=graph.eigenvectors, edge_index=graph.edge_index)
    return m


def train_model(model, loader, cfg, device, log_every: int = 0) -> List[float]:
    from graphinr_ids.training.losses import GraphINRLoss
    crit = GraphINRLoss(huber_delta=cfg["training"]["huber_delta"],
                        lipschitz_weight=cfg["training"]["lipschitz_reg"],
                        compactness_weight=cfg["training"]["compactness_weight"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
                            weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg["training"]["scheduler_T0"], T_mult=cfg["training"]["scheduler_Tmult"])
    history = []
    for ep in range(cfg["training"]["epochs"]):
        model.train()
        tot, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            losses = crit(out, model)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(losses["reconstruction"].detach()) * batch["features"].size(0)
            n += batch["features"].size(0)
        sched.step()
        history.append(tot / max(1, n))
        if log_every and (ep + 1) % log_every == 0:
            print(f"    epoch {ep+1}/{cfg['training']['epochs']}  recon={history[-1]:.6f}")
    return history


@torch.no_grad()
def per_feature_sq_error(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        o = model(b)
        out.append(((o["prediction"] - o["target"]) ** 2).cpu().numpy())
    return np.concatenate(out, axis=0)
