"""The original GraphINR-IDS, run unchanged on the new protocols.

Model, graph construction (spectral coordinates with an explicit zero vector
for out-of-vocabulary endpoints), absolute-time normalisation, min-max feature
scaling and training loop are all imported from the vendored GraphINR-IDS
code; only the data partitions and the device differ. The published
configuration is used (graph-conditioned modulation and hierarchical
frequency decomposition enabled, 50 epochs, lr 5e-5).

Returns per-feature squared residual matrices so that every calibration
policy can be applied to GraphINR-IDS exactly as to ARC-Field.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch

REUSED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party"))
if REUSED not in sys.path:
    sys.path.insert(0, REUSED)
import graphinr_ids_harness as rc  # noqa: E402


def run(corpus: Dict, P, feat_idx, seed: int, epochs: int = 50,
        device: str = "cuda", test_ip_override: Optional[Dict[str, np.ndarray]] = None,
        extra_test_sets: Optional[Dict[str, Dict[str, np.ndarray]]] = None) -> Dict:
    """Fit on P.train and return residual matrices for val / cal / test.

    `test_ip_override` = {"src": array of IP strings, "dst": ...} replaces the
    test endpoints (endpoint-renumbering experiments).
    `extra_test_sets` = {name: {"idx": rows, "src": ips, "dst": ips}} scores
    additional variants of the test stream with the same fitted model.
    """
    rc.set_seed(seed)
    t0 = time.time()
    ips = corpus["ips"]
    src_ip = ips[corpus["src"]]
    dst_ip = ips[corpus["dst"]]
    X_raw = corpus["X"][:, feat_idx]
    scaler = rc.MinMax().fit(X_raw[P.train])
    graph = rc.RevGraph(spectral_k=32).fit(src_ip[P.train], dst_ip[P.train], verbose=False)
    tn = rc.TimeNormaliser.fit(corpus["stime"][P.train].astype(np.float64))

    cfg = rc.base_config(n_features=len(feat_idx), gcm=True, hfd=True, epochs=epochs, seed=seed)
    dev = torch.device(device)
    model = rc.build_model(cfg, graph, dev)

    def loader(idx, s_ips=None, d_ips=None, shuffle=False):
        s_ips = src_ip[idx] if s_ips is None else s_ips
        d_ips = dst_ip[idx] if d_ips is None else d_ips
        ds = rc.Flows(scaler(X_raw[idx]), graph.index(s_ips), graph.index(d_ips),
                      tn(corpus["stime"][idx].astype(np.float64)),
                      corpus["is_normal"][idx])
        return torch.utils.data.DataLoader(ds, batch_size=1024 if shuffle else 8192,
                                           shuffle=shuffle, drop_last=shuffle, num_workers=0)

    hist = rc.train_model(model, loader(P.train, shuffle=True), cfg, dev)
    out = {"val": rc.per_feature_sq_error(model, loader(P.val), dev),
           "cal": rc.per_feature_sq_error(model, loader(P.cal), dev)}
    if test_ip_override is not None:
        out["test"] = rc.per_feature_sq_error(
            model, loader(P.test, test_ip_override["src"], test_ip_override["dst"]), dev)
    else:
        out["test"] = rc.per_feature_sq_error(model, loader(P.test), dev)
    if extra_test_sets:
        for name, spec in extra_test_sets.items():
            out[name] = rc.per_feature_sq_error(
                model, loader(spec["idx"], spec.get("src"), spec.get("dst")), dev)
    out["n_params"] = int(sum(p.numel() for p in model.parameters()))
    out["train_seconds"] = time.time() - t0
    out["loss_history"] = hist
    out["oov_test_flows"] = int(graph.oov_mask(src_ip[P.test], dst_ip[P.test]).sum())
    out["graph_nodes"] = graph.num_nodes - 1
    out["_model"] = model
    out["_graph"] = graph
    return out


def run_parts(X_raw: np.ndarray, src_ip: np.ndarray, dst_ip: np.ndarray, stime: np.ndarray,
              is_normal: np.ndarray, parts: Dict[str, np.ndarray], seed: int, epochs: int = 50,
              device: str = "cuda") -> Dict:
    """`run` for corpora without the UNSW-NB15 layout (Protocols IV-VI).

    Same steps as `run`: min-max scaling, spectral graph and time normaliser
    fitted on parts["train"], published configuration, per-feature squared
    residuals for every other part. Endpoint keys and times are passed
    explicitly; a corpus without addresses passes one constant endpoint and
    a corpus without timestamps a constant time, which makes the
    corresponding coordinate inert (as in GraphINR-IDS's own CIC-IDS2017 run).
    """
    rc.set_seed(seed)
    t0 = time.time()
    tr = parts["train"]
    scaler = rc.MinMax().fit(X_raw[tr])
    graph = rc.RevGraph(spectral_k=32).fit(src_ip[tr], dst_ip[tr], verbose=False)
    tn = rc.TimeNormaliser.fit(stime[tr].astype(np.float64))
    cfg = rc.base_config(n_features=X_raw.shape[1], gcm=True, hfd=True, epochs=epochs, seed=seed)
    dev = torch.device(device)
    model = rc.build_model(cfg, graph, dev)

    def loader(idx, shuffle=False):
        ds = rc.Flows(scaler(X_raw[idx]), graph.index(src_ip[idx]), graph.index(dst_ip[idx]),
                      tn(stime[idx].astype(np.float64)), is_normal[idx])
        return torch.utils.data.DataLoader(ds, batch_size=1024 if shuffle else 8192,
                                           shuffle=shuffle, drop_last=shuffle, num_workers=0)

    hist = rc.train_model(model, loader(tr, shuffle=True), cfg, dev)
    out = {k: rc.per_feature_sq_error(model, loader(ix), dev) for k, ix in parts.items() if k != "train"}
    test_idx = np.concatenate([ix for k, ix in parts.items() if k not in ("train", "val", "cal")])
    out["n_params"] = int(sum(p.numel() for p in model.parameters()))
    out["train_seconds"] = time.time() - t0
    out["loss_history"] = hist
    out["oov_test_flows"] = int(graph.oov_mask(src_ip[test_idx], dst_ip[test_idx]).sum())
    out["graph_nodes"] = graph.num_nodes - 1
    return out
