"""Label-free calibration of benign-only detectors on a non-stationary stream.

Two kinds of detector output are supported:

* residual matrices R (n, d) of per-feature deviations (ARC-Field or
  any reconstruction detector), which are *restandardised* per feature and
  aggregated by a top-kappa mean;
* scalar scores (every other baseline), for which only the threshold adapts.

Policies
--------
static        statistics from the benign validation split, threshold from the
              benign calibration split (the GraphINR-IDS operating mode).
burnin_thr    threshold re-estimated as the (1-alpha) quantile of the first B
              stream flows (label-free version of the GraphINR-IDS
              "threshold-only" refresh).
burnin_full   per-feature mean/SD *and* threshold re-estimated on the first B
              stream flows (label-free version of the GraphINR-IDS remedy).
burnin_oracle as burnin_full but on the first B *benign* stream flows, i.e.
              using labels; reported only as an upper reference. GraphINR-IDS
              drew its burn-in this way.
ema           naive streaming: exponentially weighted mean/SD per feature and
              Robbins-Monro threshold, updated with every flow (no gate).
arc           proposed: robust warm start (median/MAD, location-scale tail
              transfer), then gated, quota-limited Robbins-Monro updates of the
              per-feature median/MAD trackers and of the threshold.

All policies exclude the B burn-in flows from evaluation so that no flow is
judged against statistics derived from itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

EPS = 1e-6


def _log_u(R: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(R, 0.0) + EPS)


def topk_mean(V: np.ndarray, kappa: int) -> np.ndarray:
    k = min(kappa, V.shape[1])
    return np.partition(V, V.shape[1] - k, axis=1)[:, -k:].mean(axis=1)


@dataclass
class CalConfig:
    """Defaults = configuration selected on development Protocol D
    (logs/dev/calibrator_selection.json) and frozen before any test run.
    Ties on Protocol D were broken in favour of the variant with the stronger
    contamination guarantee (trimmed warm start)."""
    alpha: float = 0.05
    kappa: int = 6
    burnin: int = 5000
    batch: int = 256            # micro-batch for streaming updates
    eta_loc: float = 0.0        # per-flow step, in units of the current scale
    eta_scale: float = 0.0      # per-flow step on log-scale
    eta_tau: float = 0.0        # per-flow step on the transformed score
                                # (Protocol D selected 0: batch recalibration;
                                # streaming tracking is evaluated as an ablation)
    gate: float = 0.25          # admission band above the threshold (asinh units)
    quota: float = 0.2          # max share of one host among recent admissions
    quota_horizon: int = 5000
    ema_decay: float = 0.002    # naive streaming baseline: EWMA weight
    ema_tau_step: float = 0.002 # naive streaming baseline: threshold step
    robust: bool = True         # median/MAD (True) or mean/SD (False)
    use_gate: bool = True
    use_quota: bool = True
    warm_start: bool = True
    restandardise: bool = True
    scoring: str = "log_robust"   # scorer for non-ARC policies
    warm: str = "trimmed"         # warm-start threshold: "trimmed" (contamination-
                                  # trimmed burn-in quantile), "tail" (location-
                                  # scale tail transfer) or "quantile" (plain)
    trim_c: float = 1.5           # trimming point, in multiples of the clean tail distance
    recal_every: int = 0          # periodic robust recalibration period (flows);
                                  # 0 = one-off burn-in recalibration
    stat_gate: str = "none"       # "none": median/MAD trackers see every
                                  # quota-admitted flow; "alarm": only
                                  # non-alarmed flows (unstable, see dev log)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _loc_scale(L: np.ndarray, robust: bool):
    if robust:
        med = np.median(L, axis=0)
        mad = np.median(np.abs(L - med), axis=0) * 1.4826
        return med, np.maximum(mad, 1e-3)
    mu = L.mean(axis=0)
    return mu, np.maximum(L.std(axis=0), 1e-3)


class ScoreTransform:
    """l(A) = asinh((A - a0) / s0), fitted on benign calibration scores.
    Monotone, so every quantile statement about l is one about A."""

    def __init__(self, a0: float, s0: float):
        self.a0, self.s0 = a0, max(s0, 1e-9)

    @classmethod
    def fit(cls, A: np.ndarray):
        med = float(np.median(A))
        mad = float(np.median(np.abs(A - med)) * 1.4826)
        return cls(med, mad if mad > 0 else float(np.std(A)) + 1e-9)

    def __call__(self, A):
        return np.arcsinh((np.asarray(A, dtype=np.float64) - self.a0) / self.s0)


def _quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q, method="higher"))


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_policy(policy: str, cfg: CalConfig, *, R_val: Optional[np.ndarray] = None,
               R_cal: Optional[np.ndarray] = None, R_test: Optional[np.ndarray] = None,
               s_val: Optional[np.ndarray] = None, s_cal: Optional[np.ndarray] = None,
               s_test: Optional[np.ndarray] = None, host: Optional[np.ndarray] = None,
               benign_mask: Optional[np.ndarray] = None,
               inject: Optional[Dict] = None) -> Dict:
    """Apply a calibration policy to a time-ordered test stream.

    Provide either residual matrices (R_*) or scalar scores (s_*).
    `host` gives the source endpoint of every test flow (for the quota).
    `benign_mask` is used only by `burnin_oracle`.
    `inject` optionally carries poisoning flows: {"R" or "s": array,
    "host": array, "pos": insertion positions in the test stream}. Injected
    flows pass through the calibrator but are never evaluated.

    Returns scores, binary decisions and the threshold trajectory for the
    evaluated (post-burn-in, non-injected) flows.
    """
    matrix = R_test is not None
    B = cfg.burnin

    # -- assemble stream (optionally with injected flows) ------------------
    X = R_test if matrix else s_test
    n_real = len(X)
    is_real = np.ones(n_real, dtype=bool)
    hosts = host if host is not None else np.zeros(n_real, dtype=np.int64)
    if inject is not None and len(inject["pos"]):
        Xi = inject["R"] if matrix else inject["s"]
        order_key = np.concatenate([np.arange(n_real).astype(np.float64),
                                    inject["pos"].astype(np.float64) - 0.5])
        o = np.argsort(order_key, kind="stable")
        X = np.concatenate([X, Xi])[o]
        hosts = np.concatenate([hosts, inject["host"]])[o]
        is_real = np.concatenate([is_real, np.zeros(len(Xi), dtype=bool)])[o]
        real_index = np.concatenate([np.arange(n_real), -np.ones(len(Xi), dtype=np.int64)])[o]
    else:
        real_index = np.arange(n_real)
    n = len(X)

    # -- reference (fit-period) statistics -----------------------------------
    if matrix:
        # Scoring family: "log_robust" (log residuals, median/MAD) or
        # "raw_meansd" (raw residuals, mean/SD; the GraphINR-IDS scorer).
        use_log = cfg.scoring == "log_robust" or policy == "arc"
        robust_ref = cfg.robust if policy == "arc" else cfg.scoring == "log_robust"
        tf = _log_u if use_log else (lambda Z: Z)
        loc, scl = _loc_scale(tf(R_val), robust_ref)

        def scores(Xb, loc_, scl_):
            return topk_mean((tf(Xb) - loc_) / scl_, cfg.kappa)
        A_cal = scores(R_cal, loc, scl)
    else:
        loc = scl = None

        def scores(Xb, loc_, scl_):
            return Xb
        A_cal = s_cal
    T = ScoreTransform.fit(A_cal)
    l_cal = T(A_cal)
    tau_l = _quantile(l_cal, 1 - cfg.alpha)

    # tail-transfer constant: (1-alpha) quantile in median/MAD units of l_cal
    med_c = float(np.median(l_cal))
    mad_c = float(np.median(np.abs(l_cal - med_c)) * 1.4826) or 1e-6
    k_tail = (tau_l - med_c) / mad_c
    # gated target correction so that the gated fixed point equals the
    # (1-alpha) quantile on clean calibration data
    beta_g = float(np.mean(l_cal > tau_l + cfg.gate)) if cfg.use_gate else 0.0
    beta_trim = float(np.mean(l_cal > med_c + cfg.trim_c * k_tail * mad_c))
    alpha_t = (cfg.alpha - beta_g) / max(1 - beta_g, 1e-6) if cfg.use_gate else cfg.alpha

    burn = X[:B]
    burn_real = is_real[:B]

    # -- policies ---------------------------------------------------------
    traj_tau = np.full(n, np.nan)
    if policy == "static":
        A = scores(X, loc, scl)
        l = T(A)
        dec = l > tau_l
        traj_tau[:] = tau_l
    elif policy in ("burnin_thr", "burnin_full", "burnin_oracle"):
        if policy == "burnin_oracle":
            bm = benign_mask[real_index[:B].clip(min=0)] & burn_real
            ref = burn[bm]
        else:
            ref = burn
        loc2, scl2 = loc, scl
        if matrix and policy != "burnin_thr":
            loc2, scl2 = _loc_scale(tf(ref), robust=robust_ref)
        A = scores(X, loc2, scl2)
        l = T(A)
        tau2 = _quantile(T(scores(ref, loc2, scl2)), 1 - cfg.alpha)
        dec = l > tau2
        traj_tau[:] = tau2
    elif policy in ("ema", "arc"):
        l, dec, traj_tau = _stream(policy, cfg, X, hosts, scores, T, loc, scl,
                                   tau_l, k_tail, alpha_t, matrix,
                                   tf if matrix else None, beta_trim)
    else:
        raise ValueError(policy)

    keep = is_real.copy()
    keep[:B] = False
    idx_real = real_index[keep]
    order = np.argsort(idx_real)
    return {
        "score": np.asarray(l)[keep][order],
        "alarm": np.asarray(dec)[keep][order].astype(bool),
        "tau": traj_tau[keep][order],
        "eval_index": idx_real[order],       # positions in the original test stream
        "alpha_target_gated": alpha_t,
        "k_tail": k_tail,
    }


def _stream(policy, cfg, X, hosts, scores, T, loc, scl, tau_l, k_tail, alpha_t, matrix, tf,
            beta_trim=0.0):
    n = len(X)
    B = cfg.burnin
    l_out = np.empty(n)
    dec_out = np.zeros(n, dtype=bool)
    traj = np.empty(n)

    if policy == "ema":
        # naive: moments from the reference split, then EWMA on every flow
        mu = None if not matrix else loc.copy()
        var = None if not matrix else scl.copy() ** 2
        tl = tau_l
        start = 0
        lam = cfg.ema_decay
        for s in range(start, n, cfg.batch):
            Xb = X[s:s + cfg.batch]
            A = scores(Xb, mu, np.sqrt(var)) if matrix else Xb
            Lb = tf(Xb) if matrix else None
            lb = T(A)
            l_out[s:s + len(Xb)] = lb
            dec_out[s:s + len(Xb)] = lb > tl
            traj[s:s + len(Xb)] = tl
            if matrix:
                w = 1 - (1 - lam) ** len(Xb)
                mb = Lb.mean(0)
                vb = ((Lb - mu) ** 2).mean(0)
                mu = (1 - w) * mu + w * mb
                var = (1 - w) * var + w * vb
            tl += cfg.ema_tau_step * np.sum((lb > tl).astype(float) - cfg.alpha)
        return l_out, dec_out, traj

    # ---- ARC: robust (re)calibration from a window of recent flows -------
    ref_loc, ref_scl = (None if loc is None else loc.copy()), (None if scl is None else scl.copy())

    def robust_fit(Xw, hw):
        """Host-capped, median-anchored statistics and trimmed-quantile
        threshold from a window of recent flows (no labels)."""
        if cfg.use_quota:
            cap_b = max(1, int(cfg.quota * len(Xw)))
            rank = np.zeros(len(Xw), dtype=np.int64)
            seen: Dict[int, int] = {}
            for j, h in enumerate(hw):
                seen[int(h)] = seen.get(int(h), 0) + 1
                rank[j] = seen[int(h)]
            Xw = Xw[rank <= cap_b]
        loc_, scl_ = ref_loc, ref_scl
        if matrix and cfg.restandardise and cfg.warm_start:
            loc_, scl_ = _loc_scale(tf(Xw), cfg.robust)
        lw = T(scores(Xw, loc_, scl_))
        if not cfg.warm_start:
            return loc_, scl_, tau_l
        med_b = float(np.median(lw))
        mad_b = float(np.median(np.abs(lw - med_b)) * 1.4826) or 1e-6
        if cfg.warm == "tail":
            return loc_, scl_, med_b + k_tail * mad_b
        if cfg.warm == "quantile":
            return loc_, scl_, _quantile(lw, 1 - cfg.alpha)
        # contamination-trimmed quantile: discard window scores beyond trim_c
        # clean tail distances above the median, then take the quantile of the
        # rest at the level that gives benign FPR alpha under the clean law
        # (kept benign law is F(s)/F(g) with F(g) = 1 - beta_trim)
        g = med_b + cfg.trim_c * k_tail * mad_b
        kept = lw[lw <= g]
        q = (1 - cfg.alpha) / max(1 - beta_trim, 1e-6)
        if len(kept) <= 50:
            return loc_, scl_, med_b + k_tail * mad_b
        return loc_, scl_, _quantile(kept, min(max(q, 0.5), 0.9999))

    loc, scl, tl = robust_fit(X[:B], hosts[:B])
    lb_all = T(scores(X[:B], loc, scl))
    l_out[:B] = lb_all
    dec_out[:B] = lb_all > tl
    traj[:B] = tl
    last_recal = B

    recent = np.full(cfg.quota_horizon, -1, dtype=np.int64)
    rp = 0
    counts: Dict[int, int] = {}
    cap = max(1, int(cfg.quota * cfg.quota_horizon))
    loc = None if loc is None else loc.copy()
    scl = None if scl is None else scl.copy()

    for s in range(B, n, cfg.batch):
        if cfg.recal_every and s - last_recal >= cfg.recal_every:
            loc, scl, tl = robust_fit(X[s - B:s], hosts[s - B:s])
            last_recal = s
        Xb = X[s:s + cfg.batch]
        hb = hosts[s:s + cfg.batch]
        A = scores(Xb, loc, scl)
        lb = T(A)
        l_out[s:s + len(Xb)] = lb
        dec_out[s:s + len(Xb)] = lb > tl
        traj[s:s + len(Xb)] = tl

        # admission: gate on the score, then per-host quota
        gate_ok = lb <= tl + cfg.gate if cfg.use_gate else np.ones(len(lb), dtype=bool)
        ok = np.ones(len(lb), dtype=bool)
        if cfg.use_quota:
            for j in range(len(lb)):
                h = int(hb[j])
                if counts.get(h, 0) >= cap:
                    ok[j] = False
                    continue
                old = recent[rp]
                if old >= 0:
                    counts[old] -= 1
                recent[rp] = h
                rp = (rp + 1) % cfg.quota_horizon
                counts[h] = counts.get(h, 0) + 1
        adm = gate_ok & ok
        # quota-admitted flows (before the score gate) feed the robust
        # per-feature trackers; the score gate applies to the threshold only
        q_ok = ok if cfg.use_quota else np.ones(len(lb), dtype=bool)
        if not adm.any():
            continue
        # threshold: Robbins-Monro on the transformed score (gated target)
        la = lb[adm]
        tl_old = tl
        tl += cfg.eta_tau * np.sum((la > tl).astype(float) - alpha_t)
        # per-feature robust trackers
        if matrix and cfg.restandardise and (cfg.eta_loc > 0 or cfg.eta_scale > 0):
            st = q_ok & (lb <= tl_old) if cfg.stat_gate == "alarm" else q_ok
            if st.any():
                L = tf(Xb[st])
                if cfg.robust:
                    loc += cfg.eta_loc * scl * np.sum(np.sign(L - loc), axis=0)
                    dev = np.abs(L - loc)
                    scl *= np.exp(cfg.eta_scale * np.sum((dev > scl / 1.4826 * 1.0).astype(float) - 0.5, axis=0))
                else:
                    w = 1 - (1 - cfg.eta_loc) ** len(L)
                    loc = (1 - w) * loc + w * L.mean(0)
                    scl = np.sqrt((1 - w) * scl ** 2 + w * ((L - loc) ** 2).mean(0))
                scl = np.maximum(scl, 1e-3)
    return l_out, dec_out, traj
