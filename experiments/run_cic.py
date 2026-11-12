"""Protocol IV: CIC-IDS2017, fit on Monday, stream Tuesday-Friday.

Each test day is an independent stream with its own burn-in (an operator
recalibrating at the start of a working day). Metrics are reported per day
and pooled over the four days.
Output: results/raw/cic_crossday/<method>_s<seed>.json
"""
import argparse
import os
import time

import numpy as np
import torch

from common import (resilient, _FORCE_CPU, NEW_BASELINES, REUSED_BASELINES, RESULTS_RAW, af, cal, device, fit_arcfield,
                    mt, save_json, to_unit, RobustScaler, log_mask_for)
import data_cic as dc
import baselines as B0
import baselines_new as B1

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--methods", nargs="*", default=None)
ap.add_argument("--child", action="store_true")
args = ap.parse_args()
if not args.child:
    import sys as _sys
    from common import dispatch
    dc_ = __import__("data_cic")
    dc_.load_corpus(verbose=True)          # build the cache once, in the parent
    _ms = args.methods or (["ARC-Field", "ARC-Field (no context)"] + REUSED_BASELINES
                           + [b for b in NEW_BASELINES if b != "Anomal-E"])
    _od = os.path.join(RESULTS_RAW, "cic_crossday")
    jobs = [["--seeds", str(sd), "--methods", m] for sd in args.seeds for m in _ms
            if not os.path.exists(os.path.join(_od, f"{m.replace(' ', '_').replace('(', '').replace(')', '')}_s{sd}.json"))]
    _sys.exit(1 if dispatch(os.path.abspath(__file__), [], jobs, log_prefix="[dispatch] ") else 0)
corpus = dc.load_corpus()
feats = dc.FEATURES
groups = dc.groups_for(feats)
kappa = max(1, round(len(feats) / 4))
outdir = os.path.join(RESULTS_RAW, "cic_crossday")
os.makedirs(outdir, exist_ok=True)
# Anomal-E needs endpoint addresses, which this release does not contain
methods = args.methods or (["ARC-Field", "ARC-Field (no context)"] + REUSED_BASELINES
                           + [b for b in NEW_BASELINES if b != "Anomal-E"])
torch.set_num_threads(8)
POLICIES_M = ["static", "burnin_thr", "burnin_full", "arc"]
POLICIES_S = ["static", "burnin_thr", "arc"]


def evaluate(kind, base_val, base_cal, tests, labels_by_day, y_by_day, pols, cfg_kw):
    out = {}
    for pol in pols:
        kw = dict(cfg_kw)
        if kind == "matrix" and pol == "burnin_full":
            kw["scoring"] = "raw_meansd"
        cfg = cal.CalConfig(**kw)
        S, A, Y, L, per_day = [], [], [], [], {}
        for d, T in tests.items():
            # no endpoint identity in this release: the per-host cap is disabled
            host = np.zeros(len(T), dtype=np.int64)
            if kind == "matrix":
                o = cal.run_policy(pol, cfg, R_val=base_val, R_cal=base_cal, R_test=T, host=host)
            else:
                o = cal.run_policy(pol, cfg, s_val=base_val, s_cal=base_cal, s_test=T, host=host)
            y = y_by_day[d][o["eval_index"]]
            lab = labels_by_day[d][o["eval_index"]]
            m = mt.all_metrics(o["score"], o["alarm"], y, cfg.alpha)
            m["per_family"] = mt.per_family(o["alarm"], lab)
            per_day[dc.DAY_NAMES[d]] = m
            S.append(o["score"]); A.append(o["alarm"]); Y.append(y); L.append(lab)
        # pooled: decisions concatenated; ranking metrics on per-day scores are
        # not comparable across independently recalibrated days, so the pooled
        # AUROC/AUPRC are averages of the per-day values
        pooled = mt.decision_metrics(np.concatenate(A), np.concatenate(Y))
        pooled["auroc"] = float(np.mean([v["auroc"] for v in per_day.values()]))
        pooled["auprc"] = float(np.mean([v["auprc"] for v in per_day.values()]))
        pooled["per_family"] = mt.per_family(np.concatenate(A), np.concatenate(L))
        out[pol] = {"pooled": pooled, "per_day": per_day}
    return out


for seed in args.seeds:
    tr, va, ca, tests = dc.cross_day(corpus, seed)
    Xs = RobustScaler(log_mask_for(feats, dc.HEAVY)).fit(corpus["X"][tr])
    Cs = RobustScaler(None).fit(corpus["C"][tr])
    M = {"train": Xs(corpus["X"][tr]), "val": Xs(corpus["X"][va]), "cal": Xs(corpus["X"][ca])}
    C = {"train": Cs(corpus["C"][tr]), "val": Cs(corpus["C"][va]), "cal": Cs(corpus["C"][ca])}
    Mt = {d: Xs(corpus["X"][ix]) for d, ix in tests.items()}
    Ct = {d: Cs(corpus["C"][ix]) for d, ix in tests.items()}
    y_by_day = {d: ~corpus["is_normal"][ix] for d, ix in tests.items()}
    lab_by_day = {d: corpus["label"][ix] for d, ix in tests.items()}
    print(f"\n=== CIC seed {seed}: " + ", ".join(f"{dc.DAY_NAMES[d]} {len(ix):,} ({y_by_day[d].mean():.3f})"
                                                 for d, ix in tests.items()), flush=True)
    for meth in methods:
        fn = os.path.join(outdir, f"{meth.replace(' ', '_').replace('(', '').replace(')', '')}_s{seed}.json")
        if os.path.exists(fn):
            continue
        def one(meth=meth):
            t0 = time.time()
            res = {"method": meth, "seed": seed, "n_train": len(tr),
                   "test_sizes": {dc.DAY_NAMES[d]: len(ix) for d, ix in tests.items()}}
            if meth.startswith("ARC-Field"):
                use_c = "no context" not in meth
                Z = lambda A: A if use_c else np.zeros((len(A), 0), np.float32)
                model, info = fit_arcfield(M["train"], Z(C["train"]), groups, seed, device())
                Rv = af.residuals(model, M["val"], Z(C["val"]), groups, device())
                Rc = af.residuals(model, M["cal"], Z(C["cal"]), groups, device())
                Rt = {d: af.residuals(model, Mt[d], Z(Ct[d]), groups, device()) for d in tests}
                res.update(n_params=af.count_params(model), train_seconds=info["train_seconds"])
                res["policies"] = evaluate("matrix", Rv, Rc, Rt, lab_by_day, y_by_day, POLICIES_M, {"kappa": kappa, "use_quota": False})
            elif meth == "GraphINR-IDS":
                # Published GraphINR-IDS (min-max scaling, its own mean/SD scorer).
                # This release has no addresses or timestamps: one constant
                # endpoint and a constant time, so both coordinates are inert.
                import graphinr_baseline as gb
                n = len(corpus["is_normal"])
                parts = {"train": tr, "val": va, "cal": ca, **{f"d{d}": ix for d, ix in tests.items()}}
                g = gb.run_parts(corpus["X"], np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n),
                                 corpus["is_normal"], parts, seed, device=device())
                Rt = {d: g[f"d{d}"] for d in tests}
                res.update(n_params=g["n_params"], train_seconds=g["train_seconds"],
                           graph_nodes=g["graph_nodes"], oov_test_flows=g["oov_test_flows"])
                res["policies"] = evaluate("matrix", g["val"], g["cal"], Rt, lab_by_day, y_by_day, POLICIES_M,
                                           {"kappa": kappa, "use_quota": False, "scoring": "raw_meansd"})
            else:
                U = {k: to_unit(v) for k, v in M.items()}
                Ut = {d: to_unit(v) for d, v in Mt.items()}
                factory = B0.BASELINES.get(meth) or B1.NEW_BASELINES[meth]
                t1 = time.time()
                sf = factory(U["train"], seed)
                res["train_seconds"] = time.time() - t1
                sv, sc = np.asarray(sf(U["val"]), float), np.asarray(sf(U["cal"]), float)
                st = {d: np.asarray(sf(Ut[d]), float) for d in tests}
                res["policies"] = evaluate("scalar", sv, sc, st, lab_by_day, y_by_day, POLICIES_S, {"use_quota": False})

            res["wall_seconds"] = time.time() - t0
            return res

        res = resilient(one)
        _FORCE_CPU["on"] = False
        save_json(res, fn)
        a = res["policies"]["arc"]["pooled"]
        s = res["policies"]["static"]["pooled"]
        print(f"  {meth:28s} static F1={s['f1']:.4f} FPR={s['fpr']:.4f} AUROC={s['auroc']:.4f} | "
              f"arc F1={a['f1']:.4f} FPR={a['fpr']:.4f} ({res['wall_seconds']:.0f}s)", flush=True)
