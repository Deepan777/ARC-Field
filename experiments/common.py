"""Shared plumbing for the experiment scripts."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
REUSED = os.path.abspath(os.path.join(HERE, "..", "third_party"))
for p in (SRC, REUSED):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from paths import RESULTS_RAW, PROJECT  # noqa: E402
import arcfield as af  # noqa: E402
import calibration as cal  # noqa: E402
import metrics as mt  # noqa: E402
from scaling import RobustScaler, log_mask_for  # noqa: E402

# Frozen ARC-Field hyper-parameters (selected on Protocol D; see
# logs/dev/model_selection.json and logs/dev/calibrator_selection.json).
ARC_HP = {"hidden": 256, "layers": 4, "omega_0": 30.0, "epochs": 20, "lr": 2e-4,
          "batch": 1024, "beta": 0.5, "extra_p": 0.15, "hetero": False}

REUSED_BASELINES = ["Isolation Forest", "Local Outlier Factor", "One-Class SVM",
                    "Dense Autoencoder", "LSTM Autoencoder", "Deep SVDD", "DAGMM",
                    "MemAE", "USAD", "Transformer Autoencoder", "Diffusion (DDPM) detector"]
NEW_BASELINES = ["KitNET", "MCM", "DIF", "Anomal-E"]
# GraphINR-IDS (the authors' published predecessor), run unchanged through
# src/graphinr_baseline.py with its own min-max scaling and mean/SD scorer
GRAPHINR_BASELINE = ["GraphINR-IDS"]
CTX_VARIANTS = ["Isolation Forest", "Dense Autoencoder", "Deep SVDD", "KitNET", "MCM"]

MATRIX_POLICIES = ["static", "burnin_thr", "burnin_full", "burnin_oracle", "ema", "arc"]
SCALAR_POLICIES = ["static", "burnin_thr", "ema", "arc"]


def to_unit(Z: np.ndarray) -> np.ndarray:
    """Map robust-scaled values (bounded in [-6, 6]) monotonically to [0, 1]
    so that every baseline, including sigmoid-output ones, receives the same
    information as ARC-Field."""
    return ((Z / 6.0 + 1.0) / 2.0).astype(np.float32)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:     # atomic: never a partial file
        json.dump(conv(obj), f, indent=1)
    os.replace(tmp, path)


def evaluate_policies(kind: str, base: Dict, policies: List[str], ctx: Dict,
                      cfg_kw: Dict = None) -> Dict:
    """Apply calibration policies to one detector's base outputs.

    kind: "matrix" (base has R_val/R_cal/R_test) or "scalar" (s_val/...).
    ctx: y (bool attack), labels, host, t (emission time) for the test stream.
    """
    cfg_kw = cfg_kw or {}
    out = {}
    for pol in policies:
        kw = dict(cfg_kw)
        if kind == "matrix" and pol in ("burnin_full", "burnin_oracle"):
            kw["scoring"] = "raw_meansd"      # the GraphINR-IDS remedy, as published
        cfg = cal.CalConfig(**kw)
        t0 = time.time()
        if kind == "matrix":
            o = cal.run_policy(pol, cfg, R_val=base["val"], R_cal=base["cal"], R_test=base["test"],
                               host=ctx["host"], benign_mask=~ctx["y"])
        else:
            o = cal.run_policy(pol, cfg, s_val=base["val"], s_cal=base["cal"], s_test=base["test"],
                               host=ctx["host"], benign_mask=~ctx["y"])
        ei = o["eval_index"]
        y = ctx["y"][ei]
        m = mt.all_metrics(o["score"], o["alarm"], y, cfg.alpha)
        m["per_family"] = mt.per_family(o["alarm"], ctx["labels"][ei])
        if ctx.get("time_resolved"):
            m["time_resolved"] = mt.time_resolved(o["alarm"], y, ctx["t"][ei])
        m["calib_seconds"] = time.time() - t0
        out[pol] = m
    return out


def evaluate_days(kind: str, base_val, base_cal, tests: Dict, ctx: Dict, policies: List[str],
                  cfg_kw: Dict = None) -> Dict:
    """Per-day evaluation for day-streamed protocols (Protocol V and VI): each
    test day is recalibrated on its own first window; decisions are pooled
    over days and ranking metrics averaged over days (independently
    recalibrated days are not on a common score scale). Same computation as
    experiments/run_nf.py."""
    cfg_kw = cfg_kw or {}
    out = {}
    for pol in policies:
        kw = dict(cfg_kw)
        if kind == "matrix" and pol in ("burnin_full", "burnin_oracle"):
            kw["scoring"] = "raw_meansd"
        cfg = cal.CalConfig(**kw)
        A, Y, L, per_day = [], [], [], {}
        for d, T in tests.items():
            c = ctx[d]
            if kind == "matrix":
                o = cal.run_policy(pol, cfg, R_val=base_val, R_cal=base_cal, R_test=T, host=c["host"],
                                   benign_mask=~c["y"])
            else:
                o = cal.run_policy(pol, cfg, s_val=base_val, s_cal=base_cal, s_test=T, host=c["host"],
                                   benign_mask=~c["y"])
            ei = o["eval_index"]
            y, lab = c["y"][ei], c["labels"][ei]
            m = mt.all_metrics(o["score"], o["alarm"], y, cfg.alpha)
            m["per_family"] = mt.per_family(o["alarm"], lab)
            m["time_resolved"] = mt.time_resolved(o["alarm"], y, c["t"][ei])
            per_day[f"day{d}"] = m
            A.append(o["alarm"]); Y.append(y); L.append(lab)
        pooled = mt.decision_metrics(np.concatenate(A), np.concatenate(Y))
        for k in ("auroc", "auprc", "tpr_at_1fpr"):      # days without attacks give NaN
            pooled[k] = float(np.nanmean([v[k] for v in per_day.values()]))
        bm = [v["time_resolved"]["fpr_bin_max"] for v in per_day.values()]
        bm = [b for b in bm if b is not None]
        pooled["fpr_bin_max"] = float(max(bm)) if bm else float("nan")
        pooled["fpr_day_max"] = float(max(v["fpr"] for v in per_day.values()))
        pooled["per_family"] = mt.per_family(np.concatenate(A), np.concatenate(L))
        out[pol] = {"pooled": pooled, "per_day": per_day}
    return out


def fit_arcfield(M_tr, C_tr, groups, seed, device="cuda", **over):
    hp = dict(ARC_HP)
    hp.update(over)
    model = af.ARCField(M_tr.shape[1], 0 if C_tr is None else C_tr.shape[1],
                        hidden=hp["hidden"], layers=hp["layers"], omega_0=hp["omega_0"],
                        activation=hp.get("activation", "sine"), hetero=hp.get("hetero", True))
    info = af.train_field(model, M_tr, C_tr if C_tr is not None else np.zeros((len(M_tr), 0), np.float32),
                          groups, epochs=hp["epochs"], lr=hp["lr"], batch=hp["batch"], seed=seed,
                          device=device, masked=hp.get("masked", True), beta=hp["beta"],
                          extra_p=hp["extra_p"])
    return model, info


_FORCE_CPU = {"on": False}


def device():
    if _FORCE_CPU["on"] or os.environ.get("ARC_FORCE_CPU") == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def is_cuda_error(e: BaseException) -> bool:
    msg = repr(e)
    return ("out of memory" in msg or "CUDA" in msg or "cuda" in msg
            or type(e).__name__ in ("OutOfMemoryError", "AcceleratorError"))


def resilient(fn, *args, retries: int = 3, wait_s: int = 90, **kw):
    """Run fn; on a CUDA failure free memory, wait and retry, and finally fall
    back to CPU. The laptop GPU is shared with the desktop (WDDM), so free
    VRAM fluctuates; results do not depend on the device."""
    import time as _t
    if os.environ.get("ARC_CHILD") == "1":
        # isolated child: a CUDA error poisons this process's context, so fail
        # fast and let the dispatcher retry in a fresh process
        return fn(*args, **kw)
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            if not is_cuda_error(e):
                raise
            print(f"    [cuda] {type(e).__name__}; attempt {attempt + 1}/{retries + 1}", flush=True)
            free_gpu()
            if attempt == retries - 1:
                _FORCE_CPU["on"] = True
                print("    [cuda] falling back to CPU for this run", flush=True)
            _t.sleep(wait_s)
        finally:
            free_gpu()
    raise RuntimeError("run failed after retries")


def dispatch(script: str, base_args, jobs, log_prefix: str = "", retries: int = 3, wait_s: int = 60):
    """Run each job (list of extra CLI args) in its own subprocess of `script`
    with --child. A CUDA failure corrupts the CUDA context of its process, so
    isolation is the only reliable way to retry: failed jobs are retried in a
    fresh process, and the last attempt runs on CPU (ARC_FORCE_CPU=1)."""
    import subprocess
    import time as _t
    failed = []
    for job in jobs:
        ok = False
        for attempt in range(retries + 1):
            env = dict(os.environ)
            env["ARC_CHILD"] = "1"
            if attempt == retries:
                env["ARC_FORCE_CPU"] = "1"
            cmd = [sys.executable, script] + list(base_args) + list(job) + ["--child"]
            print(f"{log_prefix}job {' '.join(job)} attempt {attempt + 1}"
                  + (" (CPU)" if attempt == retries else ""), flush=True)
            r = subprocess.run(cmd, env=env)
            if r.returncode == 0:
                ok = True
                break
            print(f"{log_prefix}  exit {r.returncode}; retrying in {wait_s}s", flush=True)
            _t.sleep(wait_s)
        if not ok:
            failed.append(job)
    if failed:
        print(f"{log_prefix}FAILED jobs: {failed}", flush=True)
    return failed
