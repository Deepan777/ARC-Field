"""Evaluation metrics. Threshold-dependent metrics use the decisions produced
by the calibration policy; threshold-free metrics use the (monotone) score."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def decision_metrics(alarm: np.ndarray, y: np.ndarray) -> Dict:
    alarm = alarm.astype(bool)
    y = y.astype(bool)
    tp = int((alarm & y).sum())
    fp = int((alarm & ~y).sum())
    tn = int((~alarm & ~y).sum())
    fn = int((~alarm & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0
    bacc = 0.5 * (rec + (1 - fpr))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "fpr": fpr, "mcc": float(mcc), "balanced_acc": bacc}


def ranking_metrics(score: np.ndarray, y: np.ndarray) -> Dict:
    y = y.astype(int)
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan"), "tpr_at_1fpr": float("nan")}
    s = np.nan_to_num(score.astype(np.float64), nan=0.0, posinf=1e12, neginf=-1e12)
    out = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
    # detection rate at 1 % FPR (oracle threshold on the benign test flows);
    # reported as a threshold-free operating-point measure
    thr = np.quantile(s[y == 0], 0.99)
    out["tpr_at_1fpr"] = float((s[y == 1] > thr).mean())
    return out


def all_metrics(score: np.ndarray, alarm: np.ndarray, y: np.ndarray, alpha: float) -> Dict:
    m = decision_metrics(alarm, y)
    m.update(ranking_metrics(score, y))
    m["fpr_calibration_error"] = abs(m["fpr"] - alpha)
    return m


def per_family(alarm: np.ndarray, labels: np.ndarray) -> Dict:
    out = {}
    for fam in sorted(set(labels)):
        if fam == "Normal":
            continue
        msk = labels == fam
        out[fam] = {"n": int(msk.sum()), "tpr": float(alarm[msk].mean())}
    return out


def time_resolved(alarm: np.ndarray, y: np.ndarray, t: np.ndarray, bin_s: float = 1800.0) -> Dict:
    """Realised FPR and recall per time bin (default 30 min)."""
    b = ((t - t.min()) // bin_s).astype(int)
    rows = []
    for k in np.unique(b):
        m = b == k
        ben = m & ~y.astype(bool)
        atk = m & y.astype(bool)
        rows.append({"bin": int(k), "n": int(m.sum()), "n_benign": int(ben.sum()),
                     "n_attack": int(atk.sum()),
                     "fpr": float(alarm[ben].mean()) if ben.any() else None,
                     "recall": float(alarm[atk].mean()) if atk.any() else None})
    fprs = np.array([r["fpr"] for r in rows if r["fpr"] is not None and r["n_benign"] >= 200])
    return {"bins": rows, "bin_seconds": bin_s,
            "fpr_bin_max": float(fprs.max()) if len(fprs) else None,
            "fpr_bin_mean_abs_dev": float(np.abs(fprs - np.mean(fprs)).mean()) if len(fprs) else None}


def stratified_bootstrap(score, alarm, y, n_boot=1000, seed=0):
    """95 % CIs for F1, FPR and AUROC over flows (within one fitted model)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = {"f1": [], "fpr": [], "auroc": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        dm = decision_metrics(alarm[idx], y[idx])
        vals["f1"].append(dm["f1"])
        vals["fpr"].append(dm["fpr"])
        vals["auroc"].append(roc_auc_score(y[idx], score[idx]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}
