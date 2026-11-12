"""Aggregate raw per-run JSON into processed tables and statistical tests.

Inputs : results/raw/unsw_<protocol>_<fs>/*.json, results/raw/cic_crossday/*.json
Outputs: results/processed/*.csv|json, results/statistics/*.json
Every number in the manuscript tables is produced from these outputs.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import RESULTS_RAW, RESULTS_PROC, RESULTS_STATS  # noqa: E402

METRICS = ["auroc", "auprc", "tpr_at_1fpr", "f1", "precision", "recall", "fpr", "mcc",
           "balanced_acc", "fpr_calibration_error",
           "fpr_bin_max", "fpr_day_max"]          # the last two: pooled Protocol V records only
DISPLAY = {
    "ARC-Field": "ARC-Field (ours)", "Anomal-E": "Anomal-E", "DIF": "Deep IF",
    "Isolation Forest": "Isolation Forest", "Local Outlier Factor": "LOF",
    "One-Class SVM": "OC-SVM", "Dense Autoencoder": "Dense AE", "LSTM Autoencoder": "LSTM-AE",
    "Deep SVDD": "Deep SVDD", "DAGMM": "DAGMM", "MemAE": "MemAE", "USAD": "USAD",
    "Transformer Autoencoder": "Transformer AE", "Diffusion (DDPM) detector": "DDPM",
    "KitNET": "KitNET", "MCM": "MCM", "GraphINR-IDS": "GraphINR-IDS",
}

# Friedman-Nemenyi scopes. Protocols I and III test on the campaign-1 attacks
# that the development protocol D streamed (code/tools/audit/issue1_overlap.py),
# so the primary ranking pools only the independent protocols II and V; the
# ranking over I, II, III and V is kept for the supplement with that caveat.
FR_SCOPES = {"friedman_nemenyi.json": ("unsw_chrono_A", "nf_crossday_A"),
             "friedman_nemenyi_all.json": ("unsw_random_A", "unsw_chrono_A", "unsw_hostdisjoint_A",
                                           "nf_crossday_A")}


def load_dir(d):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        runs[r["method"]].append(r)
    return runs


def summarise(runs, pol_key=lambda r, p: r["policies"].get(p)):
    out = {}
    for meth, rs in runs.items():
        out[meth] = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs]}
        pols = sorted(set().union(*[r["policies"].keys() for r in rs]))     # sorted: reproducible output
        for p in pols:
            vals = defaultdict(list)
            for r in rs:
                m = pol_key(r, p)
                if m is None:
                    continue
                for k in METRICS:
                    if k in m and m[k] is not None:
                        vals[k].append(m[k])
                tr = m.get("time_resolved") if isinstance(m, dict) else None
                if tr:
                    vals["fpr_bin_max"].append(tr["fpr_bin_max"])
                    fb = [b["fpr"] for b in tr["bins"] if b["fpr"] is not None and b["n_benign"] >= 200]
                    vals["fpr_bin_sd"].append(float(np.std(fb)))
            out[meth][p] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                "values": [float(x) for x in v]} for k, v in vals.items()}
        for k in ("train_seconds", "n_params"):
            v = [r[k] for r in rs if k in r]
            if v:
                out[meth][k] = float(np.mean(v))
    return out


def paired_tests(runs, ref="ARC-Field", policy="arc", metrics=("auroc", "auprc", "mcc", "f1", "tpr_at_1fpr", "fpr")):
    """Paired comparisons of ref against every other method over seeds
    (paired t-test and Wilcoxon signed-rank; Cohen's d_z; Holm-adjusted)."""
    res = {}
    if ref not in runs:
        return res
    rmap = {r["seed"]: r for r in runs[ref]}
    for metric in metrics:
        rows = []
        for meth, rs in runs.items():
            if meth == ref:
                continue
            a, b = [], []
            for r in rs:
                if r["seed"] in rmap and policy in r["policies"] and policy in rmap[r["seed"]]["policies"]:
                    a.append(rmap[r["seed"]]["policies"][policy][metric])
                    b.append(r["policies"][policy][metric])
            if len(a) < 2:
                continue
            a, b = np.array(a), np.array(b)
            d = a - b
            t = stats.ttest_rel(a, b)
            try:
                w = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
            except ValueError:
                w = float("nan")
            dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf")
            rows.append({"method": meth, "n": len(d), "mean_diff": float(d.mean()), "t_p": float(t.pvalue),
                         "wilcoxon_p": float(w), "cohen_dz": dz, "wins": int((d > 0).sum())})
        # Holm correction over the family of comparisons for this metric
        ps = np.array([r["t_p"] for r in rows])
        order = np.argsort(ps)
        m = len(ps)
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * ps[i]))
            adj[i] = running
        for r, a_ in zip(rows, adj):
            r["t_p_holm"] = float(a_)
        res[metric] = rows
    return res


def friedman_nemenyi(block_scores):
    """block_scores: {method: [score per block]} with aligned blocks.
    Returns Friedman chi2, p and average ranks plus Nemenyi critical difference
    (alpha = 0.05) following Demsar (2006)."""
    meths = sorted(block_scores)
    M = np.array([block_scores[m] for m in meths]).T          # blocks x methods
    M = M[~np.isnan(M).any(1)]
    n, k = M.shape
    ranks = np.array([stats.rankdata(-row) for row in M])      # 1 = best
    avg = ranks.mean(0)
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    # Nemenyi q_0.05: studentized range quantile / sqrt(2) (Demsar 2006);
    # reproduces the tabulated values (k = 16: 3.426) and extends to any k
    q_alpha = float(stats.studentized_range.ppf(0.95, k, 1e7) / np.sqrt(2))
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"methods": meths, "avg_rank": dict(zip(meths, avg.tolist())), "chi2": float(chi2),
            "p": float(p), "n_blocks": int(n), "k": int(k), "critical_difference": float(cd)}


def main():
    os.makedirs(RESULTS_PROC, exist_ok=True)
    os.makedirs(RESULTS_STATS, exist_ok=True)
    summary, tests, blocks = {}, {}, defaultdict(lambda: defaultdict(dict))
    for d in sorted(glob.glob(os.path.join(RESULTS_RAW, "unsw_*"))):
        tag = os.path.basename(d)
        runs = load_dir(d)
        if not runs:
            continue
        summary[tag] = summarise(runs)
        tests[tag] = {p: paired_tests(runs, policy=p) for p in ("static", "arc")}
        if tag.endswith("_A"):
            for meth, rs in runs.items():
                if meth.endswith("+ctx"):
                    continue
                for r in rs:
                    for metric in ("auroc", "auprc"):
                        blocks[metric][meth][f"{tag}_s{r['seed']}"] = r["policies"]["static"][metric]
                    blocks["mcc_arc"][meth][f"{tag}_s{r['seed']}"] = r["policies"]["arc"]["mcc"]
    cic = os.path.join(RESULTS_RAW, "cic_crossday")
    if os.path.isdir(cic):
        runs = load_dir(cic)
        summary["cic_crossday"] = summarise(runs, pol_key=lambda r, p: r["policies"].get(p, {}).get("pooled"))
        for meth, rs in runs.items():
            if "no context" in meth:
                continue
            for r in rs:
                for metric in ("auroc", "auprc"):
                    blocks[metric][meth][f"cic_s{r['seed']}"] = r["policies"]["static"]["pooled"][metric]
                blocks["mcc_arc"][meth][f"cic_s{r['seed']}"] = r["policies"]["arc"]["pooled"]["mcc"]
    # Protocol V (NF-CSE-CIC-IDS2018-v3): pooled over days, flattened so that
    # the per-protocol summaries and paired tests treat it like UNSW
    for d in sorted(glob.glob(os.path.join(RESULTS_RAW, "nf_crossday_*"))):
        tag = os.path.basename(d)
        raw = load_dir(d)
        if not raw:
            continue
        runs = {m: [dict(r, policies={p: v["pooled"] for p, v in r["policies"].items()}) for r in rs]
                for m, rs in raw.items()}
        summary[tag] = summarise(runs)
        tests[tag] = {p: paired_tests(runs, policy=p) for p in ("static", "arc")}
        if tag.endswith("_A"):
            for meth, rs in runs.items():
                if "no context" in meth:
                    continue
                for r in rs:
                    for metric in ("auroc", "auprc"):
                        blocks[metric][meth][f"{tag}_s{r['seed']}"] = r["policies"]["static"][metric]
                    blocks["mcc_arc"][meth][f"{tag}_s{r['seed']}"] = r["policies"]["arc"]["mcc"]
    # Protocol VI (HIKARI-2021): flow level, no stream order or endpoint
    # context; reported separately and kept out of the Friedman blocks
    for d in sorted(glob.glob(os.path.join(RESULTS_RAW, "hikari_*"))):
        tag = os.path.basename(d)
        runs = load_dir(d)
        if runs:
            summary[tag] = summarise(runs)
            tests[tag] = {p: paired_tests(runs, policy=p) for p in ("static", "arc")}
    fr_all = {}
    for fname, scope in FR_SCOPES.items():
        fr = {}
        for metric, bm in blocks.items():
            sb = {m: {k: v for k, v in kv.items() if k.rsplit("_s", 1)[0] in scope} for m, kv in bm.items()}
            sb = {m: kv for m, kv in sb.items() if kv}
            keys = sorted(set.intersection(*[set(v.keys()) for v in sb.values()]))
            if len(keys) >= 3 and len(sb) >= 3:
                fr[metric] = friedman_nemenyi({m: [sb[m][k] for k in keys] for m in sb})
                fr[metric]["blocks"] = keys
                fr[metric]["protocols"] = list(scope)
        json.dump(fr, open(os.path.join(RESULTS_STATS, fname), "w"), indent=1)
        fr_all[fname] = fr
    fr = fr_all["friedman_nemenyi.json"]
    json.dump(summary, open(os.path.join(RESULTS_PROC, "summary.json"), "w"), indent=1)
    json.dump(tests, open(os.path.join(RESULTS_STATS, "paired_tests.json"), "w"), indent=1)
    # flat CSV for inspection
    with open(os.path.join(RESULTS_PROC, "summary_flat.csv"), "w") as f:
        f.write("protocol,method,policy,metric,mean,sd,n\n")
        for tag, s in summary.items():
            for meth, v in s.items():
                for p, mm in v.items():
                    if not isinstance(mm, dict) or "auroc" not in mm:
                        continue
                    for k, st in mm.items():
                        f.write(f"{tag},{meth},{p},{k},{st['mean']:.6f},{st['sd']:.6f},{len(st['values'])}\n")
    print("aggregated:", list(summary.keys()), "friedman:", {k: (v['p'], v['n_blocks']) for k, v in fr.items()})


if __name__ == "__main__":
    main()
