"""Aggregate the context-baseline study (experiments/run_context_baselines.py).

Outputs results/processed/context_baselines.json with, per protocol:
  summary       mean and SD over seeds of every (method, context) cell
                (static policy: AUROC, AUPRC, detection at 1 % FPR;
                ARC policy: MCC and FPR)
  paired        ARC-Field with role coordinates against ARC-Field with every
                other encoding: mean paired difference, paired t-test with
                Holm correction across encodings, wins over seeds
  conditioning  gain of each encoding over no context, for ARC-Field (the
                encoding conditions the field) and averaged over the appended
                baselines (the encoding is extra input columns)
  pooled        the paired comparison over the protocol-seed blocks of the
                independent protocols (POOL_PROTOCOLS: II and V)
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import PROJECT, RESULTS_PROC  # noqa: E402

BASE = os.path.join(PROJECT, "results", "context_baselines")
PROTOCOLS = ["chrono", "hostdisjoint", "nf", "hikari"]
STATIC = ["auroc", "auprc", "tpr_at_1fpr"]
ENC = ["none", "spectral", "afterimage", "rolx", "roles"]
APPENDED = ["MemAE", "Deep SVDD", "Dense Autoencoder", "Isolation Forest", "KitNET", "MCM"]
# The pooled comparison uses only the independent protocols: Protocol III
# (hostdisjoint) evaluates on attacks of the development protocol D
# (results/processed/dev_overlap.json), so it is reported per protocol only.
POOL_PROTOCOLS = ("chrono", "nf")


def load(prot):
    cells = {}
    for f in sorted(glob.glob(os.path.join(BASE, prot, "*.json"))):
        d = json.load(open(f))
        st = d["policies"]["static"]
        st = st.get("pooled", st)
        ar = d["policies"]["arc"]
        ar = ar.get("pooled", ar)
        row = {m: 100.0 * float(st[m]) for m in STATIC}
        row["arc_mcc"] = 100.0 * float(ar["mcc"])
        row["arc_fpr"] = 100.0 * float(ar["fpr"])
        row["dim"] = float(d.get("context_dim", 0))
        cells.setdefault((d["method"], d["context"]), {})[int(d["seed"])] = row
    return cells


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    out = {}
    for m in STATIC + ["arc_mcc"]:
        x = np.array([a[s][m] for s in seeds])
        y = np.array([b[s][m] for s in seeds])
        d = x - y
        p = float(stats.ttest_rel(x, y).pvalue) if len(d) > 1 and d.std() > 0 else float("nan")
        out[m] = {"diff": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "p": p,
                  "wins": int((d > 0).sum()), "n": len(d)}
    return out


def main():
    res = {}
    pooled_blocks = {e: {m: [] for m in STATIC + ["arc_mcc"]} for e in ENC}
    for prot in PROTOCOLS:
        cells = load(prot)
        if not cells:
            continue
        summ = {}
        for (meth, cx), seeds in cells.items():
            summ.setdefault(meth, {})[cx] = {
                m: {"mean": float(np.mean([v[m] for v in seeds.values()])),
                    "sd": float(np.std([v[m] for v in seeds.values()], ddof=1)) if len(seeds) > 1 else 0.0}
                for m in STATIC + ["arc_mcc", "arc_fpr"]}
            summ[meth][cx]["n_seeds"] = len(seeds)
            summ[meth][cx]["dim"] = float(np.mean([v["dim"] for v in seeds.values()]))
        ref = cells.get(("ARC-Field", "roles"), {})
        pr = {}
        for cx in ENC:
            if cx == "roles" or ("ARC-Field", cx) not in cells or not ref:
                continue
            pr[cx] = paired(ref, cells[("ARC-Field", cx)])
            if prot not in POOL_PROTOCOLS:
                continue
            for m in STATIC + ["arc_mcc"]:
                for s in sorted(set(ref) & set(cells[("ARC-Field", cx)])):
                    pooled_blocks[cx][m].append(ref[s][m] - cells[("ARC-Field", cx)][s][m])
        for m in STATIC + ["arc_mcc"]:
            keys = [cx for cx in pr if np.isfinite(pr[cx][m]["p"])]
            for cx, pa in zip(keys, holm([pr[cx][m]["p"] for cx in keys])):
                pr[cx][m]["p_holm"] = float(pa)
        cond = {}
        for cx in ("afterimage", "rolx", "roles"):
            g = {}
            if ("ARC-Field", cx) in cells and ("ARC-Field", "none") in cells:
                g["ARC-Field"] = {m: summ["ARC-Field"][cx][m]["mean"] - summ["ARC-Field"]["none"][m]["mean"]
                                  for m in STATIC}
            gains = [{m: summ[b][cx][m]["mean"] - summ[b]["none"][m]["mean"] for m in STATIC}
                     for b in APPENDED if b in summ and cx in summ[b] and "none" in summ[b]]
            if gains:
                g["appended_mean"] = {m: float(np.mean([x[m] for x in gains])) for m in STATIC}
                g["appended_n_improved_auprc"] = int(sum(x["auprc"] > 0 for x in gains))
                g["appended_n"] = len(gains)
            cond[cx] = g
        best = {}
        for cx in ENC + ["afterimage-only"]:
            cand = [(summ[m][cx]["auprc"]["mean"], m) for m in summ if cx in summ[m]]
            if cand:
                v, m = max(cand)
                best[cx] = {"method": m, "auprc": v}
        res[prot] = {"summary": summ, "paired": pr, "conditioning": cond, "best_by_context": best}
    pooled = {}
    for cx, ms in pooled_blocks.items():
        if ms["auprc"]:
            pooled[cx] = {}
            for m, d in ms.items():
                d = np.array(d)
                w = stats.wilcoxon(d) if len(d) > 5 and np.any(d != 0) else None
                pooled[cx][m] = {"diff": float(d.mean()), "wins": int((d > 0).sum()), "n": len(d),
                                 "p_wilcoxon": float(w.pvalue) if w is not None else float("nan")}
    res["pooled_roles_vs"] = pooled
    os.makedirs(RESULTS_PROC, exist_ok=True)
    out = os.path.join(RESULTS_PROC, "context_baselines.json")
    json.dump(res, open(out, "w"), indent=1)
    for prot in PROTOCOLS:
        if prot not in res:
            continue
        print(f"\n== {prot}")
        s = res[prot]["summary"].get("ARC-Field", {})
        for cx in ENC:
            if cx in s:
                print(f"  ARC-Field|{cx:11s} AUROC {s[cx]['auroc']['mean']:6.2f}  AUPRC {s[cx]['auprc']['mean']:6.2f}"
                      f"  TPR@1% {s[cx]['tpr_at_1fpr']['mean']:6.2f}  arcMCC {s[cx]['arc_mcc']['mean']:6.2f}"
                      f"  (n={s[cx]['n_seeds']})")
        for cx, g in res[prot]["conditioning"].items():
            if "ARC-Field" in g and "appended_mean" in g:
                print(f"  gain {cx:10s} conditioned AUPRC {g['ARC-Field']['auprc']:+6.2f} | appended mean "
                      f"{g['appended_mean']['auprc']:+6.2f} ({g['appended_n_improved_auprc']}/{g['appended_n']} improve)")
        print("  best by context:", {k: (v['method'], round(v['auprc'], 2)) for k, v in res[prot]["best_by_context"].items()})
    print("\npooled roles vs:", json.dumps({k: {m: (round(v['diff'], 2), f"{v['wins']}/{v['n']}")
                                                for m, v in d.items()} for k, d in pooled.items()}))
    print("written", out)


if __name__ == "__main__":
    main()
