"""Endpoint churn on Protocol II: identity-free role coordinates versus
transductive spectral coordinates.

A fraction rho of the endpoints active in the test campaign is given fresh
identifiers, in two modes:
  renumber  every test-campaign record of a selected endpoint carries the new
            identifier (re-addressing of the whole deployment segment)
  lease     the identifier switches at a uniformly random instant of the test
            campaign (DHCP lease change): records before the switch keep the
            old identifier, records after it the new one, so the new
            identifier starts with an empty history
Two models are compared, identical except for the endpoint coordinate:
ARC-Field (causal role coordinates, recomputed from the modified stream) and
the same masked field with Laplacian-eigenvector (spectral) endpoint
coordinates of the fitting graph, which assign an unseen identifier the zero
vector (a transductive coordinate, as in spectral positional encodings).
Both are fitted once per seed on unmodified campaign-1 data.
Output: results/raw/churn/churn_results.json
"""
import argparse
import json
import os

import numpy as np

from common import (resilient, RESULTS_RAW, af, device, evaluate_policies, fit_arcfield, save_json,
                    RobustScaler, log_mask_for)
import context as cx
import data_unsw as du
import graphinr_ids_harness as rc

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()
corpus = du.load_corpus()
feats = du.FEATURE_SETS["A"]
fi = [du.ALL36.index(f) for f in feats]
groups = du.groups_for(feats)
RHOS = [0.0, 0.1, 0.25, 0.5, 1.0]
cut = du.session_cut(corpus)
s2 = np.where(corpus["stime"] > cut)[0]          # emission-ordered campaign-2 rows
proto = corpus["proto"][s2]
col = {n: i for i, n in enumerate(du.ALL36)}


def context_for(src, dst):
    C, _ = cx.unsw_context(src=src, dst=dst, dport=corpus["dport"][s2], t=corpus["ltime"][s2].astype(float),
                           sbytes=corpus["X"][s2, col["sbytes"]].astype(float),
                           dbytes=corpus["X"][s2, col["dbytes"]].astype(float),
                           dur=corpus["X"][s2, col["dur"]].astype(float), proto=proto)
    return C


def churned(rho, mode, rng):
    src, dst = corpus["src"][s2].copy(), corpus["dst"][s2].copy()
    ents = np.unique(np.concatenate([src, dst]))
    chosen = rng.choice(ents, int(round(rho * len(ents))), replace=False) if rho > 0 else np.array([], int)
    base = int(max(corpus["src"].max(), corpus["dst"].max())) + 1
    t = corpus["ltime"][s2]
    for k, e in enumerate(chosen):
        new = base + k
        if mode == "renumber":
            src[src == e] = new
            dst[dst == e] = new
        else:
            ts = rng.uniform(t.min(), t.max())
            src[(src == e) & (t >= ts)] = new
            dst[(dst == e) & (t >= ts)] = new
    return src, dst, len(chosen)


# sanity: recomputing on campaign 2 alone reproduces the cached coordinates
C0 = context_for(corpus["src"][s2], corpus["dst"][s2])
max_dev = float(np.abs(C0 - corpus["C"][s2]).max())
print("context recomputation max deviation vs cache:", max_dev, flush=True)

OUT = os.path.join(RESULTS_RAW, "churn", "churn_results.json")
out = {"rhos": RHOS, "context_recompute_max_abs_dev": max_dev, "results": [], "done_seeds": []}
if os.path.exists(OUT):
    out = json.load(open(OUT))
pos_in_s2 = {int(r): i for i, r in enumerate(s2)}
for seed in args.seeds:
    if seed in out.get("done_seeds", []):
        continue
    P = du.chronological(corpus, seed=seed)
    tpos = np.array([pos_in_s2[int(r)] for r in P.test])
    Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(corpus["X"][P.train][:, fi])
    Cs = RobustScaler(None).fit(corpus["C"][P.train])
    M = {k: Xs(corpus["X"][getattr(P, k)][:, fi]) for k in ("train", "val", "cal", "test")}
    C = {k: Cs(corpus["C"][getattr(P, k)]) for k in ("train", "val", "cal")}
    model, _ = resilient(fit_arcfield, M["train"], C["train"], groups, seed, device())
    Rv = af.residuals(model, M["val"], C["val"], groups, device())
    Rc = af.residuals(model, M["cal"], C["cal"], groups, device())
    y = ~corpus["is_normal"][P.test]
    ctx = {"y": y, "labels": corpus["label"][P.test], "t": corpus["ltime"][P.test].astype(float)}
    rng = np.random.default_rng(seed)
    variants, extra = [], {}
    for mode in ("renumber", "lease"):
        for rho in RHOS:
            if rho == 0 and mode == "lease":
                continue
            src, dst, nch = churned(rho, mode, rng)
            Ct = Cs(context_for(src, dst)[tpos])
            Rt = af.residuals(model, M["test"], Ct, groups, device())
            ctx_i = dict(ctx, host=src[tpos])
            res = evaluate_policies("matrix", {"val": Rv, "cal": Rc, "test": Rt},
                                    ["static", "burnin_thr", "arc"], ctx_i, {"kappa": 6})
            out["results"].append({"seed": seed, "method": "ARC-Field", "mode": mode, "rho": rho,
                                   "n_churned_endpoints": nch,
                                   "policies": {p: {k: v for k, v in m.items() if k != "per_family"}
                                                for p, m in res.items()}})
            name = f"{mode}_{rho}"
            variants.append((name, mode, rho, nch, src[tpos]))
            ips = corpus["ips"]
            fresh = lambda codes: np.array([ips[c] if c < len(ips) else f"fresh-{c}" for c in codes], dtype=object)
            extra[name] = {"idx": P.test, "src": fresh(src[tpos]), "dst": fresh(dst[tpos])}
            print(f"s{seed} ARC {mode} rho={rho} F1={res['arc']['f1']:.4f} FPR={res['arc']['fpr']:.4f} "
                  f"AUROC={res['static']['auroc']:.4f}", flush=True)
    # transductive comparator: same field, spectral endpoint coordinates
    ips = corpus["ips"]
    graph = rc.RevGraph(spectral_k=32).fit(ips[corpus["src"][P.train]], ips[corpus["dst"][P.train]], verbose=False)

    def pe(src_ip, dst_ip):
        ps = graph.eigenvectors[graph.index(src_ip)]
        pd_ = graph.eigenvectors[graph.index(dst_ip)]
        return np.concatenate([ps, pd_, ps * pd_], 1).astype(np.float32)
    PEs = {k: pe(ips[corpus["src"][getattr(P, k)]], ips[corpus["dst"][getattr(P, k)]]) for k in ("train", "val", "cal")}
    smodel, _ = resilient(fit_arcfield, M["train"], PEs["train"], groups, seed, device())
    Sv = af.residuals(smodel, M["val"], PEs["val"], groups, device())
    Sc = af.residuals(smodel, M["cal"], PEs["cal"], groups, device())
    for name, mode, rho, nch, hosts in variants:
        spec = extra[name]
        St = af.residuals(smodel, M["test"], pe(spec["src"], spec["dst"]), groups, device())
        res = evaluate_policies("matrix", {"val": Sv, "cal": Sc, "test": St},
                                ["static", "burnin_thr", "arc"], dict(ctx, host=hosts), {"kappa": 6})
        out["results"].append({"seed": seed, "method": "Spectral coordinates", "mode": mode, "rho": rho,
                               "n_churned_endpoints": nch,
                               "policies": {p: {k: v for k, v in m.items() if k != "per_family"}
                                            for p, m in res.items()}})
        print(f"s{seed} spectral {mode} rho={rho} F1={res['arc']['f1']:.4f} FPR={res['arc']['fpr']:.4f} "
              f"static FPR={res['static']['fpr']:.4f} AUROC={res['static']['auroc']:.4f}", flush=True)
    out.setdefault("done_seeds", []).append(seed)
    save_json(out, OUT)
