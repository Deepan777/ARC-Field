"""Robust, drift-tolerant input scaling.

GraphINR-IDS used min-max scaling fitted on the training window and clipped to
[0, 1], so any post-deployment value beyond the fitted range saturates. Here
heavy-tailed features are log-compressed and every feature is centred on its
median and scaled by its inter-quartile range, then soft-bounded. Values beyond
the fitted range remain ordered instead of collapsing onto a boundary.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class RobustScaler:
    def __init__(self, log_mask: Optional[np.ndarray] = None, bound: float = 6.0):
        self.log_mask = log_mask
        self.bound = bound
        self.med = None
        self.scale = None

    def _pre(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.log_mask is not None and self.log_mask.any():
            X = X.copy()
            L = X[:, self.log_mask]
            X[:, self.log_mask] = np.sign(L) * np.log1p(np.abs(L))
        return X

    def fit(self, X: np.ndarray) -> "RobustScaler":
        Z = self._pre(X)
        self.med = np.median(Z, axis=0)
        q75, q25 = np.percentile(Z, [75, 25], axis=0)
        iqr = q75 - q25
        # Features that are constant over the inter-quartile range (counters
        # that are almost always zero) fall back to the standard deviation,
        # then to 1.
        sd = Z.std(axis=0)
        s = np.where(iqr > 1e-9, iqr / 1.349, np.where(sd > 1e-9, sd, 1.0))
        self.scale = s
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (self._pre(X) - self.med) / self.scale
        # smooth bound: identity near 0, saturates at +-bound
        return (self.bound * np.tanh(Z / self.bound)).astype(np.float32)


def log_mask_for(names: List[str], heavy: set) -> np.ndarray:
    return np.array([n in heavy for n in names], dtype=bool)
