"""Additional manuscript tables and macros (called from make_tables.py).

Each builder silently skips when its input file does not exist yet, so the
table generator can run at any stage of the pipeline.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import RESULTS_ABL, RESULTS_PROC, RESULTS_RAW  # noqa: E402

CTX = {}
FAM_SHORT = {"Analysis": "Anal.", "Backdoor": "Backd.", "DoS": "DoS", "Exploits": "Expl.",
             "Fuzzers": "Fuzz.", "Generic": "Gen.", "Reconnaissance": "Recon.", "Shellcode": "Shell.",
             "Worms": "Worms", "ALL": "All"}


def set_context(macro, write, name, summary):
    CTX.update(macro=macro, write=write, name=name, S=summary)


def alpha(s):
    return re.sub(r"[^A-Za-z]", "", s)


def numtag(x):
    """Collision-free letter encoding of a number for macro names:
    digits -> a..j, '.' -> p (0.05 -> apaf, 0.2 -> apc, 1000 -> baaa)."""
    return "".join("abcdefghij"[int(c)] if c.isdigit() else ("p" if c == "." else "") for c in str(x))


def disp(m):
    key = m.replace("_", " ").replace("Diffusion DDPM detector", "Diffusion (DDPM) detector")
    return CTX["name"].get(key, key).split("~")[0]


def ms(vals, nd=2, scale=100):
    v = np.array(vals, float) * scale
    if len(v) >= 3:
        return f"{v.mean():.{nd}f}\\,$\\pm$\\,{v.std(ddof=1):.{nd}f}"
    return f"{v.mean():.{nd}f}"


# ---------------------------------------------------------------------------
# Protocols I and III reuse the attacks of the development protocol D
# (results/processed/dev_overlap.json); every caption that reports them says so
NONINDEP_III = (r" Protocol~III is not an independent test: every evaluated attack record was also in the stream "
                r"of the development protocol on which the hyper-parameters were selected.")


def family_matched():
    f = os.path.join(RESULTS_PROC, "family_matched_fpr.json")
    if not os.path.exists(f):
        return
    D = json.load(open(f))
    for fam, sh in D.get("family_share_chrono", {}).items():
        CTX["macro"](f"NShare{alpha(fam)}", f"{100 * sh:.0f}")
    for prot, lab in (("chrono", "II"), ("hostdisjoint", "III")):
        if prot not in D:
            continue
        T = D[prot]["0.01"]
        others = [m for m in T if m != "ARC-Field"]
        ratios = []
        for x in [k for k in T["ARC-Field"] if k not in ("ALL", "Generic", "Fuzzers")]:
            bo = max(np.mean(T[m].get(x, [0])) for m in others)
            ratios.append(np.mean(T["ARC-Field"][x]) / max(bo, 1e-9))
        CTX["macro"](f"NFam{lab}RatioMin", f"{min(ratios):.1f}")
        CTX["macro"](f"NFam{lab}RatioMax", f"{max(ratios):.1f}")
    for prot, lab in (("chrono", "II"), ("hostdisjoint", "III")):
        if prot not in D:
            continue
        T = D[prot]["0.01"]
        fams = [x for x in FAM_SHORT if x in T.get("ARC-Field", {})]
        meths = sorted(T, key=lambda m: -np.mean(T[m]["ALL"]))[:9]
        best = {x: max(np.mean(T[m].get(x, [0])) for m in meths) for x in fams}
        lines = []
        for m in meths:
            cells = []
            for x in fams:
                v = np.mean(T[m].get(x, [np.nan]))
                c = f"{100 * v:.1f}"
                cells.append(r"\textbf{" + c + "}" if abs(v - best[x]) < 1e-12 else c)
            lines.append(disp(m) + " & " + " & ".join(cells) + r" \\")
        tex = (r"\begin{table*}[t]" "\n" r"\centering" "\n"
               r"\caption{Detection rate (\%) per attack family at a matched false-positive rate of 1\,\% on "
               f"Protocol~{lab}. The threshold of each detector is set on its own benign test scores; mean over "
               r"five seeds, nine best detectors by overall rate. Matching the false-positive rate removes the "
               r"confound of different operating points."
               + (NONINDEP_III if lab == "III" else "") + "}" "\n"
               rf"\label{{tab:family_{prot}}}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3.5pt}" "\n"
               r"\begin{tabular}{@{}l" + "c" * len(fams) + "@{}}\n" r"\toprule" "\n"
               "Detector & " + " & ".join(FAM_SHORT[x] for x in fams) + r" \\" "\n" r"\midrule" "\n"
               + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table*}" "\n")
        CTX["write"](f"tab_family_{prot}.tex", tex)
        others = [m for m in T if m != "ARC-Field"]
        for x in fams:
            CTX["macro"](f"NFam{lab}Arc{alpha(x)}", f"{100 * np.mean(T['ARC-Field'][x]):.1f}")
            if "MemAE" in T:
                CTX["macro"](f"NFam{lab}Mem{alpha(x)}", f"{100 * np.mean(T['MemAE'][x]):.1f}")
            bo = max(others, key=lambda m: np.mean(T[m].get(x, [0])))
            CTX["macro"](f"NFam{lab}Best{alpha(x)}", f"{100 * np.mean(T[bo][x]):.1f}")
        T2 = D[prot]["0.001"]
        CTX["macro"](f"NFam{lab}ArcAllStrict", f"{100 * np.mean(T2['ARC-Field']['ALL']):.1f}")
        CTX["macro"](f"NFam{lab}MemAllStrict", f"{100 * np.mean(T2['MemAE']['ALL']):.1f}")


# ---------------------------------------------------------------------------
ABL_NAMES = {"full": "ARC-Field (full)", "no_context": "$-$ role coordinates",
             "src_context": "source roles only", "dst_context": "destination roles only",
             "unmasked": "$-$ masking",
             "heteroscedastic": "+ heteroscedastic head", "relu": "SIREN $\\rightarrow$ ReLU MLP",
             "spectral_coord": "spectral coordinates, raw",
             "abs_time": "+ absolute-time encoding"}


def ablation_model():
    rows = {}
    for prot in ("chrono", "hostdisjoint"):
        for f in glob.glob(os.path.join(RESULTS_ABL, f"model_{prot}", "*.json")):
            r = json.load(open(f))
            rows.setdefault(prot, {}).setdefault(r["variant"], []).append(r)
    if not rows:
        return
    lines = []
    panels = {"chrono": [], "hostdisjoint": []}
    for v in ABL_NAMES:
        for prot in ("chrono", "hostdisjoint"):
            rs = rows.get(prot, {}).get(v, [])
            if not rs:
                panels[prot].append(" & ".join([ABL_NAMES[v]] + ["--"] * 5) + r" \\")
                continue
            panels[prot].append(" & ".join(
                [ABL_NAMES[v], ms([r["policies"]["static"]["auroc"] for r in rs]),
                 ms([r["policies"]["static"]["auprc"] for r in rs]),
                 ms([r["policies"]["static"]["tpr_at_1fpr"] for r in rs]),
                 ms([r["policies"]["arc"]["fpr"] for r in rs]),
                 ms([r["policies"]["arc"]["mcc"] for r in rs])]) + r" \\")
            if v == "full":
                panels[prot].append(r"\addlinespace")
            for key, kg in (("auprc", "Auprc"), ("tpr_at_1fpr", "Tpr"), ("auroc", "Auroc")):
                CTX["macro"](f"NAbl{'Chr' if prot == 'chrono' else 'Host'}{alpha(v)}{kg}",
                             f"{100 * np.mean([r['policies']['static'][key] for r in rs]):.2f}")
            CTX["macro"](f"NAbl{'Chr' if prot == 'chrono' else 'Host'}{alpha(v)}Mcc",
                         f"{100 * np.mean([r['policies']['arc']['mcc'] for r in rs]):.2f}")
            CTX["macro"](f"NAbl{'Chr' if prot == 'chrono' else 'Host'}{alpha(v)}ArFpr",
                         f"{100 * np.mean([r['policies']['arc']['fpr'] for r in rs]):.2f}")
            CTX["macro"](f"NAbl{'Chr' if prot == 'chrono' else 'Host'}{alpha(v)}TprSD",
                         f"{100 * np.std([r['policies']['static']['tpr_at_1fpr'] for r in rs], ddof=1):.2f}")
    for prot, title in (("chrono", "Protocol~II (cross-campaign)"), ("hostdisjoint", "Protocol~III (host-disjoint)")):
        lines.append(r"\multicolumn{6}{@{}l}{\textit{" + title + r"}} \\")
        lines += panels[prot]
        lines.append(r"\midrule")
    lines = lines[:-1]
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Component ablation of ARC-Field (retrained per variant; mean\,$\pm$\,SD over five seeds, \%). "
           r"Ranking metrics use the static scorer; FPR and MCC are those after ARC recalibration. "
           r"``Spectral coordinates, raw'': the Laplacian-eigenvector coordinate of GraphINR-IDS, fed to the field "
           r"unscaled (zero vector for unseen endpoints)." + NONINDEP_III + "}" "\n"
           r"\label{tab:ablation}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{2.5pt}" "\n"
           r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n"
           r"Variant & AUROC & AUPRC & TPR@1\% & FPR & MCC \\" "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    CTX["write"]("tab_ablation.tex", tex)


CALIB_NAMES = {"ARC (full)": "ARC (full)", "- host cap": "$-$ host cap",
               "- trimming (plain quantile)": "$-$ trimming (plain quantile)",
               "tail transfer instead of trimming": "tail extrapolation for trimming",
               "mean/SD instead of median/MAD": "mean/SD for median/MAD",
               "- restandardisation": "$-$ restandardisation",
               "+ periodic recalibration (20k)": "+ periodic (every 20k)",
               "+ continuous RM tracking": "+ Robbins--Monro tracking",
               "+ streaming median/MAD tracking": "+ streaming median/MAD"}


def _calib_tex(A, meths, pname, label, extra=""):
    """ARC-Field with mean +- SD, the other detectors as means (width)."""
    lines = []
    for n in A[meths[0]]:
        cells = [CALIB_NAMES.get(n, n.replace("- ", "$-$ "))]
        for i, m in enumerate(meths):
            rs = A[m].get(n)
            if not rs:
                cells += ["--", "--"]
                continue
            if i == 0:
                cells += [ms([r["fpr"] for r in rs]), ms([r["mcc"] for r in rs])]
            else:
                cells += [f"{100 * np.mean([r['fpr'] for r in rs]):.2f}",
                          f"{100 * np.mean([r['mcc'] for r in rs]):.1f}"]
        lines.append(" & ".join(cells) + r" \\")
    head = " & ".join(r"\multicolumn{2}{c}{" + disp(m) + "}" for m in meths)
    cm = "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(meths)))
    return (r"\begin{table}[t]" "\n" r"\centering" "\n"
            r"\caption{Ablation of the recalibration layer on " + pname + r" (no refitting; \%; mean\,$\pm$\,SD "
            r"over five seeds for ARC-Field, means for the other detectors)." + extra + "}\n"
            r"\label{" + label + "}\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{2.2pt}" "\n"
            r"\begin{tabular}{@{}l" + "cc" * len(meths) + "@{}}\n" r"\toprule" "\n"
            " & " + head + r" \\" "\n" + cm + "\n"
            "Variant & " + " & ".join(["FPR & MCC"] * len(meths)) + r" \\" "\n" r"\midrule" "\n"
            + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")


# ---------------------------------------------------------------------------
def calibration_ablation():
    f = os.path.join(RESULTS_ABL, "calibration_chrono.json")
    if not os.path.exists(f):
        return
    A = json.load(open(f))["ablation"]
    meths = [m for m in ("ARC-Field", "MemAE", "Dense_Autoencoder", "DIF") if m in A]
    names = list(A[meths[0]].keys())
    for n in names:
        for m in meths:
            rs = A[m].get(n)
            if not rs:
                continue
            CTX["macro"](f"NCal{alpha(m)}{alpha(n)}Fpr", f"{100 * np.mean([r['fpr'] for r in rs]):.2f}")
            CTX["macro"](f"NCal{alpha(m)}{alpha(n)}Mcc", f"{100 * np.mean([r['mcc'] for r in rs]):.2f}")
            CTX["macro"](f"NCal{alpha(m)}{alpha(n)}Rec", f"{100 * np.mean([r['recall'] for r in rs]):.2f}")
    tex = _calib_tex(A, meths, "Protocol~II", "tab:calib_ablation",
                     r" Each row removes or replaces one component of Algorithm~\ref{alg:recal}. Rows marked with "
                     r"``+'' add the adaptive mechanisms rejected on the development protocol.")
    CTX["write"]("tab_calib_ablation.tex", tex)
    # sensitivity macros for the text
    Sv = json.load(open(f))["sensitivity"]
    for m in meths:
        for par, vals in Sv.get(m, {}).items():
            for v, rs in vals.items():
                CTX["macro"](f"NSens{alpha(m)}{alpha(par)}{numtag(v)}Fpr",
                             f"{100 * np.mean([r['fpr'] for r in rs]):.2f}")


# ---------------------------------------------------------------------------
def poisoning():
    f = os.path.join(RESULTS_RAW, "poisoning", "poisoning_results.json")
    if not os.path.exists(f):
        return
    P = json.load(open(f))["results"]
    pols = [("burnin_full", "Mean/SD"), ("burnin_thr", "Quantile"), ("arc", "ARC")]
    settings = [("none", 0, "single", "--"),
                ("blind", 1, "single", "1"), ("blind", 10, "single", "10"), ("blind", 50, "single", "50"),
                ("replay", 0.05, "single", "5"), ("replay", 0.2, "single", "20"),
                ("replay", 0.2, "distributed", "20$^{\\ast}$"),
                ("chaff", 0.05, "single", "5"), ("chaff", 0.2, "single", "20")]

    def sel(meth, pol, kind, x, spread):
        out = []
        for r in P:
            if r["method"] != meth or r["policy"] != pol or r["spread"] != spread:
                continue
            if kind == "none" and r["n_injected"] == 0:
                out.append(r)
            elif kind == "blind" and r["kind"] == "blind" and r["n_injected"] == x:
                out.append(r)
            elif kind in ("replay", "chaff") and r["kind"] == kind and abs(r["rate"] - x) < 1e-9:
                out.append(r)
        return out

    lines = []
    for meth in ("ARC-Field", "MemAE", "Dense_Autoencoder"):
        mpols = pols if meth == "ARC-Field" else pols[1:]      # scalar detectors: threshold policies only
        for pol, pname in mpols:
            cells = [disp(meth) if pol == mpols[0][0] else "", pname]
            for kind, x, spread, _ in settings:
                rs = sel(meth, pol, kind, x, spread)
                cells.append(f"{100 * np.mean([r['recall'] for r in rs]):.1f}" if rs else "--")
                if rs:
                    CTX["macro"](f"NPois{alpha(meth)}{alpha(pol)}{alpha(kind)}{numtag(x)}{alpha(spread)}Rec",
                                 f"{100 * np.mean([r['recall'] for r in rs]):.1f}")
                    CTX["macro"](f"NPois{alpha(meth)}{alpha(pol)}{alpha(kind)}{numtag(x)}{alpha(spread)}Fpr",
                                 f"{100 * np.mean([r['fpr'] for r in rs]):.2f}")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Recall (\%) after label-free recalibration when an adversary injects records into the "
           r"5{,}000-record recalibration window (Protocol~II, mean over five seeds). \emph{Blind}: records with an "
           r"extreme residual on the six features that best separate attacks, counted in records; \emph{replay}: "
           r"copies of genuine attack records; \emph{chaff}: log-space interpolations between benign and attack "
           r"records; replay and chaff in \% of the window, from one host or ($^{\ast}$) spread over twenty hosts. "
           r"Mean/SD: per-feature mean/SD restandardisation with a quantile threshold; Quantile: plain quantile "
           r"threshold; ARC: Algorithm~\ref{alg:recal}. Injected records are never evaluated.}" "\n"
           r"\label{tab:poisoning}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
           r"\begin{tabular}{@{}ll" + "c" * len(settings) + "@{}}\n" r"\toprule" "\n"
           r" & & Clean & \multicolumn{3}{c}{Blind} & \multicolumn{3}{c}{Replay} & \multicolumn{2}{c}{Chaff} \\" "\n"
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(l){10-11}" "\n"
           "Detector & Rule & " + " & ".join(s[3] for s in settings) + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    CTX["write"]("tab_poisoning.tex", tex)


# ---------------------------------------------------------------------------
def churn():
    f = os.path.join(RESULTS_RAW, "churn", "churn_results.json")
    if not os.path.exists(f):
        return
    C = json.load(open(f))
    CTX["macro"]("NChurnCtxDev", f"{C['context_recompute_max_abs_dev']:.1e}")
    lines = []
    for meth in ("ARC-Field", "Spectral coordinates"):
        for mode in ("renumber", "lease"):
            label = "ARC-Field (role coordinates)" if meth == "ARC-Field" else "Spectral coordinates, raw"
            cells = [label if mode == "renumber" else "", mode]
            for rho in C["rhos"]:
                rs = [r for r in C["results"] if r["method"] == meth and r["rho"] == rho
                      and (r["mode"] == mode or rho == 0)]
                if not rs:
                    cells.append("--")
                    continue
                fpr = 100 * np.mean([r["policies"]["arc"]["fpr"] for r in rs])
                au = 100 * np.mean([r["policies"]["static"]["auroc"] for r in rs])
                sfpr = 100 * np.mean([r["policies"]["static"]["fpr"] for r in rs])
                rec = 100 * np.mean([r["policies"]["arc"]["recall"] for r in rs])
                cells.append(f"{fpr:.1f} / {au:.1f}")
                tag = f"NChurn{alpha(meth)}{alpha(mode)}{numtag(rho)}"
                CTX["macro"](tag + "Fpr", f"{fpr:.1f}")
                CTX["macro"](tag + "StFpr", f"{sfpr:.1f}")
                CTX["macro"](tag + "Rec", f"{rec:.1f}")
                CTX["macro"](tag + "Auroc", f"{au:.1f}")
            lines.append(" & ".join(cells) + r" \\")
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Endpoint churn on Protocol~II: realised FPR after ARC recalibration / AUROC (\%) when a share "
           r"$\rho$ of the endpoints active in the test campaign receive fresh identifiers (mean over three seeds). "
           r"\emph{Renumber}: all their records; \emph{lease}: from a random instant on. Both models are the same "
           r"masked field; only the endpoint coordinate differs.}" "\n"
           r"\label{tab:churn}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
           r"\begin{tabular}{@{}ll" + "c" * len(C["rhos"]) + "@{}}\n" r"\toprule" "\n"
           r"Detector & Mode & " + " & ".join(f"$\\rho={r}$" for r in C["rhos"]) + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    CTX["write"]("tab_churn.tex", tex)


# ---------------------------------------------------------------------------
def featuresets():
    S = CTX["S"]
    sets = [("A", "Tier A (24)"), ("A_noTTL", "Tier A $-$ TTL (21)"), ("AB", "Tier A+B (31)"), ("ABC", "All (36)")]
    have = [(k, n) for k, n in sets if f"unsw_chrono_{k}" in S]
    if len(have) < 2:
        return
    meths = ["ARC-Field", "MemAE", "Dense Autoencoder", "Deep SVDD", "KitNET", "MCM", "DIF"]
    lines = []
    for m in meths:
        cells = [disp(m)]
        for k, _ in have:
            d = S[f"unsw_chrono_{k}"].get(m, {}).get("static")
            if not d:
                cells.append("--")
                continue
            cells.append(f"{100 * d['auprc']['mean']:.2f}")
            CTX["macro"](f"NFs{alpha(k)}{alpha(m)}Auprc", f"{100 * d['auprc']['mean']:.2f}")
            CTX["macro"](f"NFs{alpha(k)}{alpha(m)}Auroc", f"{100 * d['auroc']['mean']:.2f}")
            if "tpr_at_1fpr" in d:
                CTX["macro"](f"NFs{alpha(k)}{alpha(m)}Tpr", f"{100 * d['tpr_at_1fpr']['mean']:.2f}")
            ar = S[f"unsw_chrono_{k}"].get(m, {}).get("arc")
            if ar:
                CTX["macro"](f"NFs{alpha(k)}{alpha(m)}ArFpr", f"{100 * ar['fpr']['mean']:.2f}")
                CTX["macro"](f"NFs{alpha(k)}{alpha(m)}ArMcc", f"{100 * ar['mcc']['mean']:.2f}")
                CTX["macro"](f"NFs{alpha(k)}{alpha(m)}ArRec", f"{100 * ar['recall']['mean']:.2f}")
        # Protocol V columns (NF-CSE-CIC-IDS2018-v3), when available
        for k, _ in NF_SETS:
            d = S.get(f"nf_crossday_{k}", {}).get(m, {}).get("static")
            cells.append(f"{100 * d['auprc']['mean']:.2f}" if d else "--")
        lines.append(" & ".join(cells) + r" \\")
    nf_have = [(k, n) for k, n in NF_SETS if f"nf_crossday_{k}" in S]
    if len(nf_have) < len(NF_SETS):
        lines = [" & ".join(l.rstrip(r" \\").split(" & ")[:1 + len(have)]) + r" \\" for l in lines]
        nf_have = []
    shorth = {"A": "A", "A_noTTL": "A$-$TTL", "AB": "A+B", "ABC": "All"}
    head1 = (r" & \multicolumn{" + str(len(have)) + r"}{c}{UNSW-NB15 (Prot.~II)}"
             + (r" & \multicolumn{" + str(len(nf_have)) + r"}{c}{NetFlow corpus (Prot.~V)}"
                if nf_have else "") + r" \\")
    cm = (r"\cmidrule(lr){2-" + str(1 + len(have)) + "}"
          + (r"\cmidrule(l){" + str(2 + len(have)) + "-" + str(1 + len(have) + len(nf_have)) + "}" if nf_have else ""))
    head2 = "Detector & " + " & ".join(shorth[k] for k, _ in have + nf_have) + r" \\"
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{AUPRC (\%) by observable feature set. Tier~A of UNSW-NB15: five seeds; every other "
           r"column: three seeds. ``A$-$TTL'' removes the TTL-derived features (\texttt{sttl}, \texttt{dttl}, "
           r"\texttt{ct\_state\_ttl} on UNSW-NB15; \texttt{MIN\_TTL}, \texttt{MAX\_TTL} on the NetFlow corpus); A+B adds the "
           r"TCP-header features; All adds the payload-derived UNSW-NB15 features.}" "\n"
           r"\label{tab:featuresets}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
           r"\begin{tabular}{@{}l" + "c" * (len(have) + len(nf_have)) + "@{}}\n" r"\toprule" "\n"
           + head1 + "\n" + cm + "\n" + head2 + "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    CTX["write"]("tab_featuresets.tex", tex)


NF_SETS = [("A", "Tier A (32)"), ("A_noTTL", "Tier A $-$ TTL (30)"), ("AB", "Tier A+B (41)")]


def nf_extra_macros():
    """Protocol V: feature-set macros, ranking position, calibration counts,
    and the tail-transfer diagnostic of both datasets."""
    S = CTX["S"]
    meths = ["ARC-Field", "MemAE", "Dense Autoencoder", "Deep SVDD", "KitNET", "MCM", "DIF"]
    for k, _ in NF_SETS:
        F = S.get(f"nf_crossday_{k}", {})
        for m in meths + ["Anomal-E", "ARC-Field (no context)"]:
            d = F.get(m)
            if not d:
                continue
            base = f"NNfFs{alpha(k)}{alpha(m)}"
            st, ar, bt = d.get("static", {}), d.get("arc", {}), d.get("burnin_thr", {})
            for src, key, tg in ((st, "auroc", "Auroc"), (st, "auprc", "Auprc"), (st, "tpr_at_1fpr", "Tpr"),
                                 (st, "fpr", "StFpr"), (ar, "fpr", "ArFpr"), (ar, "recall", "ArRec"),
                                 (ar, "mcc", "ArMcc"), (bt, "fpr", "BtFpr"), (bt, "recall", "BtRec"),
                                 (bt, "mcc", "BtMcc")):
                if key in src:
                    CTX["macro"](base + tg, f"{100 * src[key]['mean']:.2f}")
    F = S.get("nf_crossday_A", {})
    base = {m: v for m, v in F.items() if "no context" not in m}
    if base:
        words = ["first", "second", "third", "fourth", "fifth", "sixth"]
        for key, tg in (("auroc", "Auroc"), ("auprc", "Auprc")):
            order = sorted(base, key=lambda m: -base[m]["static"][key]["mean"])
            CTX["macro"](f"NNf{tg}ArcRankWord", words[order.index("ARC-Field")] if order.index("ARC-Field") < 6
                         else str(order.index("ARC-Field") + 1))
            second = order[1] if order[0] == "ARC-Field" else order[0]
            CTX["macro"](f"NNf{tg}Runner", disp(second))
            CTX["macro"](f"NNf{tg}RunnerVal", f"{100 * base[second]['static'][key]['mean']:.2f}")
        CTX["macro"]("NNfNdet", str(len(base)))
        st_f = [100 * v["static"]["fpr"]["mean"] for v in base.values()]
        ar_f = [100 * v["arc"]["fpr"]["mean"] for v in base.values()]
        bt_f = [100 * v["burnin_thr"]["fpr"]["mean"] for v in base.values() if "burnin_thr" in v]
        CTX["macro"]("NNfStFprMin", f"{min(st_f):.1f}")
        CTX["macro"]("NNfStFprMax", f"{max(st_f):.1f}")
        CTX["macro"]("NNfArOverN", str(sum(1 for x in ar_f if x > 5.0)))
        CTX["macro"]("NNfArFprMin", f"{min(ar_f):.1f}")
        CTX["macro"]("NNfArFprMax", f"{max(ar_f):.1f}")
        CTX["macro"]("NNfBtFprMax", f"{max(bt_f):.1f}")
        roles = F.get("ARC-Field"), F.get("ARC-Field (no context)")
        if all(roles):
            CTX["macro"]("NNfRoleGainAuroc", f"{100 * (roles[0]['static']['auroc']['mean'] - roles[1]['static']['auroc']['mean']):.1f}")
            CTX["macro"]("NNfRoleGainAuprc", f"{100 * (roles[0]['static']['auprc']['mean'] - roles[1]['static']['auprc']['mean']):.1f}")
    f = os.path.join(RESULTS_PROC, "tail_transfer.json")
    if os.path.exists(f):
        TT = json.load(open(f))
        # misspecification remark: with no contamination the trimmed rule
        # realises 1 - (1 - alpha)(1 - beta_win) / (1 - beta_cal)
        for ds, tg in (("nf_crossday_A", "Nf"), ("unsw_chrono_A", "Chr")):
            rows = [r for r in TT["rows"].get(ds, []) if r["beta_win_ben"] is not None]
            if rows:
                pred = [1 - 0.95 * (1 - r["beta_win_ben"]) / (1 - r["beta_cal"]) for r in rows]
                obs = [r["arc_c1.5"]["fpr"] for r in rows]
                CTX["macro"](f"NTt{tg}PredFpr", f"{100 * np.mean(pred):.1f}")
                CTX["macro"](f"NTt{tg}ObsFpr", f"{100 * np.mean(obs):.1f}")
                if len(rows) >= 3:
                    CTX["macro"](f"NTt{tg}PredCorr", f"{np.corrcoef(pred, obs)[0, 1]:.2f}")
        T = TT["summary"]
        for ds, tg in (("nf_crossday_A", "Nf"), ("unsw_chrono_A", "Chr")):
            s = T.get(ds)
            if not s:
                continue
            for k, v in s.items():
                if v is None:
                    continue
                nm = k.replace("c1.5", "Conefive").replace("c2", "Ctwo").replace("c3", "Cthree")
                CTX["macro"](f"NTt{tg}{alpha(nm)}", f"{100 * v:.1f}")


# ---------------------------------------------------------------------------
def cic():
    S = CTX["S"].get("cic_crossday")
    if not S:
        return
    order = sorted(S, key=lambda m: -S[m]["static"]["auprc"]["mean"])
    lines = []
    for m in order:
        st, ar = S[m]["static"], S[m].get("arc", {})
        g = lambda d, k: ms(d[k]["values"]) if d and k in d else "--"  # noqa: E731
        lines.append(" & ".join([disp(m) if "no context" not in m else "ARC-Field $-$ context",
                                 g(st, "auroc"), g(st, "auprc"), g(st, "fpr"), g(ar, "recall"),
                                 g(ar, "fpr"), g(ar, "mcc")]) + r" \\")
        for k, kg in (("auroc", "Auroc"), ("auprc", "Auprc"), ("fpr", "StFpr")):
            CTX["macro"](f"NCic{alpha(m)}{kg}", f"{100 * st[k]['mean']:.2f}")
        for k, kg in (("f1", "Fone"), ("fpr", "Fpr"), ("mcc", "Mcc"), ("recall", "Rec")):
            if ar:
                CTX["macro"](f"NCic{alpha(m)}Ar{kg}", f"{100 * ar[k]['mean']:.2f}")
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Protocol~IV, CIC-IDS2017: fit on Monday, Tuesday--Friday streamed as separate days, each "
           r"recalibrated on its first window (mean\,$\pm$\,SD over three seeds, \%). Ranking metrics are averages of "
           r"the per-day values; operating-point metrics pool the four days.}" "\n"
           r"\label{tab:cic}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{2pt}" "\n"
           r"\begin{tabular}{@{}lccc|ccc@{}}" "\n" r"\toprule" "\n"
           r" & \multicolumn{2}{c}{Ranking} & Static & \multicolumn{3}{c}{ARC recalibration} \\" "\n"
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(l){5-7}" "\n"
           r"Detector & AUROC & AUPRC & FPR & Recall & FPR & MCC \\" "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    CTX["write"]("tab_cic.tex", tex)


# ---------------------------------------------------------------------------
def complexity():
    f = os.path.join(RESULTS_RAW, "complexity", "complexity.json")
    if not os.path.exists(f):
        return
    C = json.load(open(f))
    a, c = C["arcfield"], C["context"]
    CTX["macro"]("NCxArcParams", f"{a['params'] / 1e6:.2f}")
    CTX["macro"]("NCxArcMacs", f"{a['macs_per_record'] / 1e6:.2f}")
    # records per second, rounded to hundreds and printed in full (the text
    # previously appended ",000" to a value in thousands, giving "1301,000")
    rate = lambda v: f"{int(round(v, -2)):,}".replace(",", "{,}")  # noqa: E731
    CTX["macro"]("NCxArcGpu", rate(a.get("records_per_s_cuda", float("nan"))))
    CTX["macro"]("NCxArcCpu", rate(a.get("records_per_s_cpu", float("nan"))))
    CTX["macro"]("NCxCtxVec", rate(c["vectorised_records_per_s"]))
    CTX["macro"]("NCxCtxStream", rate(c["streaming_records_per_s_python"]))
    CTX["macro"]("NCxCtxState", f"{c['streaming_peak_state_entries']:,}".replace(",", "{,}"))
    CTX["macro"]("NCxCal", f"{1000 * C['calibration']['arc_policy_seconds_per_20k_stream']:.0f}")
    S = CTX["S"].get("unsw_chrono_A", {})
    rows = []
    for m in ("ARC-Field", "Anomal-E", "DIF", "MemAE", "Dense Autoencoder", "Deep SVDD", "Diffusion (DDPM) detector",
              "KitNET", "MCM", "LSTM Autoencoder", "Transformer Autoencoder", "Isolation Forest", "One-Class SVM"):
        if m in S and "train_seconds" in S[m]:
            rows.append(f"{disp(m)} & {S[m]['train_seconds']:.0f} \\\\")
            CTX["macro"](f"NCxTrain{alpha(m)}", f"{S[m]['train_seconds']:.0f}")
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Fitting time on 64{,}000 records (s, mean over Protocol~II seeds). ARC-Field, Anomal-E and "
           r"the added baselines train on the GPU; the reused baselines run on the CPU in their original form, so their times are "
           r"indicative only.}" "\n"
           r"\label{tab:fit_time}" "\n" r"\scriptsize" "\n" r"\begin{tabular}{@{}lc@{}}" "\n" r"\toprule" "\n"
           r"Detector & Fit time \\" "\n" r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n"
           r"\end{tabular}" "\n" r"\end{table}" "\n")
    CTX["write"]("tab_fit_time.tex", tex)


# ---------------------------------------------------------------------------
def stationarity():
    f = os.path.join(RESULTS_PROC, "stationarity_attribution.json")
    if not os.path.exists(f):
        return
    D = json.load(open(f))
    sv = D.get("stationarity_vs_fpr_dispersion")
    if sv:
        CTX["macro"]("NStatRho", f"{sv['spearman']:.2f}")
        CTX["macro"]("NStatP", f"{sv['p']:.1e}")
        CTX["macro"]("NStatN", str(sv["n_detectors"]))
    for m, v in D.get("stationarity_summary", {}).items():
        CTX["macro"](f"NStat{alpha(m)}Ks", f"{v['mean_ks']:.3f}")
    col = D.get("collateral", {})
    if col:
        CTX["macro"]("NCollStatic", f"{np.mean([v['spearman_static'][0] for v in col.values()]):.2f}")
        CTX["macro"]("NCollArc", f"{np.mean([v['spearman_arc'][0] for v in col.values()]):.2f}")
    att = D.get("attribution", {})
    if att:
        fams = sorted({k for v in att.values() for k in v})
        groups = list(next(iter(att.values()))[fams[0]].keys())
        lines = []
        for fam in fams:
            vals = [np.mean([v[fam][g] for v in att.values() if fam in v]) for g in groups]
            lines.append(fam + " & " + " & ".join(f"{100 * x:.0f}" for x in vals) + r" \\")
            for g, x in zip(groups, vals):
                CTX["macro"](f"NAtt{alpha(fam)}{alpha(g)}", f"{100 * x:.0f}")
        tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
               r"\caption{Share (\%) of each observability group among the $\kappa$ features that make up the "
               r"ARC-Field score of alarmed records, by attack family and for benign false alarms (Protocol~II, "
               r"mean over seeds).}" "\n" r"\label{tab:attribution}" "\n" r"\scriptsize" "\n"
               r"\begin{tabular}{@{}l" + "c" * len(groups) + "@{}}\n" r"\toprule" "\n"
               "Alarmed records & " + " & ".join(g.capitalize() for g in groups) + r" \\" "\n" r"\midrule" "\n"
               + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
        CTX["write"]("tab_attribution.tex", tex)


def build_all():
    for fn in (family_matched, ablation_model, calibration_ablation, poisoning, churn, featuresets, cic,
               complexity, stationarity):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("tables_extra:", fn.__name__, "skipped:", repr(e))


def headline_counts():
    """Detector counts and the static-threshold FPR range across published
    detectors on Protocol II (Introduction motivation)."""
    import data_unsw  # noqa: F401  (ensures src path)
    sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
    from common import REUSED_BASELINES, NEW_BASELINES, GRAPHINR_BASELINE
    base = REUSED_BASELINES + NEW_BASELINES + GRAPHINR_BASELINE
    CTX["macro"]("NnBaselines", str(len(base)))
    CTX["macro"]("NnDetectors", str(len(base)))
    S = CTX["S"].get("unsw_chrono_A", {})
    nd = len([m for m in S if not m.endswith("+ctx")])
    CTX["macro"]("NChrNdetWord", {16: "sixteen", 17: "seventeen"}.get(nd, str(nd)))
    f = [S[m]["static"]["fpr"]["mean"] for m in base if m in S]
    if f:
        CTX["macro"]("NChrStFprMin", f"{100 * min(f):.0f}")
        CTX["macro"]("NChrStFprMax", f"{100 * max(f):.0f}")
        CTX["macro"]("NChrStFprNdet", str(len(f)))
        ar = {m: S[m]["arc"]["fpr"]["mean"] for m in base if m in S and "arc" in S[m]}
        over = {m: v for m, v in ar.items() if v > 0.05}
        CTX["macro"]("NChrArOverN", str(len(over)))
        CTX["macro"]("NChrArOverMin", f"{100 * min(over.values()):.0f}")
        CTX["macro"]("NChrArOverMax", f"{100 * max(over.values()):.0f}")
        CTX["macro"]("NChrArUnderN", str(len(ar) - len(over) + 1))   # + ARC-Field
        # worst half hour of the carried-over threshold above 40 %, and the exceptions
        bm = {m: S[m]["static"]["fpr_bin_max"]["mean"] for m in base if m in S and "fpr_bin_max" in S[m]["static"]}
        over40 = [m for m, v in bm.items() if v > 0.40]
        words = {13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen"}
        CTX["macro"]("NChrStBinOverFortyWord", words.get(len(over40), str(len(over40))))
        CTX["macro"]("NChrStBinExceptions", ", ".join(disp(m) for m in sorted(set(bm) - set(over40))))
        ew = [S[m]["ema"]["recall"]["mean"] for m in base + ["ARC-Field"] if m in S and "ema" in S[m]]
        if ew:
            CTX["macro"]("NChrEwRecMax", f"{100 * max(ew):.1f}")


_old_build_all = build_all


def build_all():  # noqa: F811
    _old_build_all()
    try:
        headline_counts()
    except Exception as e:  # noqa: BLE001
        print("tables_extra: headline_counts skipped:", repr(e))


def _fr_macros(fr, pref):
    """Average ranks and derived counts of one Friedman-Nemenyi file."""
    if os.path.exists(fr):
        F = json.load(open(fr))
        for metric, d in F.items():
            mg = alpha(metric.replace("mcc_arc", "mccarc")).capitalize()
            for m, r in d["avg_rank"].items():
                CTX["macro"](f"{pref}{mg}Rank{alpha(m)}", f"{r:.2f}")
            order = sorted(d["avg_rank"].items(), key=lambda kv: kv[1])
            CTX["macro"](f"{pref}{mg}First", disp(order[0][0]))
            CTX["macro"](f"{pref}{mg}ArcPos", str([m for m, _ in order].index("ARC-Field") + 1)
                         if "ARC-Field" in d["avg_rank"] else "--")
            CTX["macro"](f"{pref}{mg}K", str(d["k"]))
            words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
            if "ARC-Field" in d["avg_rank"]:
                pos = [m for m, _ in order].index("ARC-Field")
                CTX["macro"](f"{pref}{mg}ArcPosWord", words[pos] if pos < len(words) else f"{pos + 1}th")
            # detectors not significantly different from the best (within CD)
            best = order[0][1]
            CTX["macro"](f"{pref}{mg}TopGroup", str(sum(1 for _, r in order if r - best <= d["critical_difference"])))


def stats_macros():
    """Every Friedman average rank and every paired-test statistic as a macro.
    NFr*: primary ranking (Protocols II and V); NFrAll*: Protocols I-III and V."""
    for fn, pref in (("friedman_nemenyi.json", "NFr"), ("friedman_nemenyi_all.json", "NFrAll")):
        _fr_macros(os.path.join(os.path.dirname(RESULTS_PROC), "statistics", fn), pref)
    pt = os.path.join(os.path.dirname(RESULTS_PROC), "statistics", "paired_tests.json")
    if os.path.exists(pt):
        T = json.load(open(pt))
        for tag, tg in (("unsw_chrono_A", "Chr"), ("unsw_hostdisjoint_A", "Host"), ("unsw_random_A", "Rand"),
                        ("nf_crossday_A", "Nf")):
            for pol in ("static", "arc"):
                for metric, rows in T.get(tag, {}).get(pol, {}).items():
                    for r in rows:
                        base = f"NPt{tg}{alpha(pol).capitalize()}{alpha(metric).capitalize()}{alpha(r['method'])}"
                        CTX["macro"](base + "Diff", f"{100 * r['mean_diff']:+.2f}")
                        p = r["t_p_holm"]
                        CTX["macro"](base + "P", f"{p:.3f}" if p >= 0.001 else "<0.001")
                        CTX["macro"](base + "Dz", f"{r['cohen_dz']:.2f}")
                        CTX["macro"](base + "Wins", f"{r['wins']}/{r['n']}")


_prev_build_all = build_all


def build_all():  # noqa: F811
    _prev_build_all()
    try:
        stats_macros()
    except Exception as e:  # noqa: BLE001
        print("tables_extra: stats_macros skipped:", repr(e))


def all_method_macros():
    """N<Proto><Method><Policy><Metric> for every detector (incl. +ctx) and policy."""
    keys = (("auroc", "Auroc"), ("auprc", "Auprc"), ("f1", "Fone"), ("fpr", "Fpr"), ("recall", "Rec"),
            ("mcc", "Mcc"), ("fpr_bin_max", "Fprmax"), ("tpr_at_1fpr", "Tpr"), ("precision", "Prec"),
            ("fpr_day_max", "Fprday"))
    pols = (("static", "St"), ("arc", "Ar"), ("burnin_full", "Bf"), ("burnin_thr", "Bt"),
            ("burnin_oracle", "Bo"), ("ema", "Ew"))
    for tag, tg in (("unsw_chrono_A", "Chr"), ("unsw_hostdisjoint_A", "Host"), ("unsw_random_A", "Rand"),
                    ("nf_crossday_A", "Nf")):
        for m, v in CTX["S"].get(tag, {}).items():
            mg = "M" + alpha(m.replace("+ctx", "Ctx"))
            for pol, pg in pols:
                d = v.get(pol)
                if not isinstance(d, dict):
                    continue
                for k, kg in keys:
                    if k in d:
                        CTX["macro"](f"N{tg}{mg}{pg}{kg}", f"{100 * d[k]['mean']:.2f}")
                        CTX["macro"](f"N{tg}{mg}{pg}{kg}SD", f"{100 * d[k]['sd']:.2f}")


_prev2 = build_all


def build_all():  # noqa: F811
    _prev2()
    try:
        all_method_macros()
    except Exception as e:  # noqa: BLE001
        print("tables_extra: all_method_macros skipped:", repr(e))


def poisoning_quota_macros():
    f = os.path.join(RESULTS_RAW, "poisoning", "poisoning_quota.json")
    if not os.path.exists(f):
        return
    R = json.load(open(f))["results"]
    from collections import defaultdict
    a = defaultdict(list)
    for r in R:
        a[(r["kind"], r["rate"], r["spread"], r["quota"])].append(r)
    for (kind, rate, spread, q), rs in a.items():
        tag = f"NPq{alpha(kind)}{numtag(rate)}{alpha(spread)}{numtag(q)}"
        CTX["macro"](tag + "Rec", f"{100 * np.mean([r['recall'] for r in rs]):.1f}")
        CTX["macro"](tag + "Fpr", f"{100 * np.mean([r['fpr'] for r in rs]):.2f}")


_prev3 = build_all


def build_all():  # noqa: F811
    _prev3()
    try:
        poisoning_quota_macros()
    except Exception as e:  # noqa: BLE001
        print("tables_extra: poisoning_quota_macros skipped:", repr(e))


# ---------------------------------------------------------------------------
# Supplementary tables (tab_s_*) and the remaining calibration macros
_SENS_METHODS = ("ARC-Field", "MemAE", "Dense_Autoencoder", "KitNET", "Deep_SVDD", "DIF", "Isolation_Forest")
_SENS_PARAMS = (("trim_c", "trim constant $c$"), ("burnin", "window $B$"), ("alpha", r"target $\alpha$"),
                ("quota", "host cap $q$"))


def _calib_table(prot, label, fname, pname):
    f = os.path.join(RESULTS_ABL, f"calibration_{prot}.json")
    if not os.path.exists(f):
        return
    A = json.load(open(f))["ablation"]
    meths = [m for m in ("ARC-Field", "MemAE", "Dense_Autoencoder", "DIF") if m in A]
    tex = _calib_tex(A, meths, pname, label, r" Rows as in Table~\ref{tab:calib_ablation}." + NONINDEP_III)
    CTX["write"](fname, tex)


def calibration_extra():
    for prot, tag in (("chrono", ""), ("hostdisjoint", "Host")):
        f = os.path.join(RESULTS_ABL, f"calibration_{prot}.json")
        if not os.path.exists(f):
            continue
        D = json.load(open(f))
        for m, rows in D["ablation"].items():
            for n, rs in rows.items():
                base = f"NCal{tag}{alpha(m)}{alpha(n)}"
                CTX["macro"](base + "Auroc", f"{100 * np.mean([r['auroc'] for r in rs]):.2f}")
                if tag:
                    CTX["macro"](base + "Fpr", f"{100 * np.mean([r['fpr'] for r in rs]):.2f}")
                    CTX["macro"](base + "Mcc", f"{100 * np.mean([r['mcc'] for r in rs]):.2f}")
                    CTX["macro"](base + "Rec", f"{100 * np.mean([r['recall'] for r in rs]):.2f}")
        for m, pars in D["sensitivity"].items():
            for par, vals in pars.items():
                for v, rs in vals.items():
                    base = f"NSens{tag}{alpha(m)}{alpha(par)}{numtag(v)}"
                    CTX["macro"](base + "Rec", f"{100 * np.mean([r['recall'] for r in rs]):.1f}")
                    CTX["macro"](base + "Mcc", f"{100 * np.mean([r['mcc'] for r in rs]):.1f}")
                    if tag:
                        CTX["macro"](base + "Fpr", f"{100 * np.mean([r['fpr'] for r in rs]):.2f}")
    _calib_table("hostdisjoint", "tab:s_calib_host", "tab_s_calib_host.tex", "Protocol~III")
    for prot, pname, label in (("chrono", "Protocol~II", "tab:s_sens_chrono"),
                               ("hostdisjoint", "Protocol~III", "tab:s_sens_host")):
        f = os.path.join(RESULTS_ABL, f"calibration_{prot}.json")
        if not os.path.exists(f):
            continue
        S = json.load(open(f))["sensitivity"]
        meths = [m for m in _SENS_METHODS if m in S]
        lines = []
        for par, pdisp in _SENS_PARAMS:
            vals = list(S[meths[0]].get(par, {}).keys())
            for i, v in enumerate(vals):
                cells = [pdisp if i == 0 else "", v]
                for m in meths:
                    rs = S[m].get(par, {}).get(v)
                    cells += (["--", "--"] if not rs else
                              [f"{100 * np.mean([r['fpr'] for r in rs]):.2f}",
                               f"{100 * np.mean([r['recall'] for r in rs]):.1f}"])
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
        short = {"ARC-Field": "ARC-Field", "Isolation_Forest": "IF", "Deep_SVDD": "Deep SVDD", "DIF": "Deep IF",
                 "Dense_Autoencoder": "Dense AE"}
        head = " & ".join(r"\multicolumn{2}{c}{" + short.get(m, disp(m)) + "}" for m in meths)
        cm = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(meths)))
        tex = (r"\begin{table*}[t]" "\n" r"\centering" "\n"
               r"\caption{Sensitivity of the ARC recalibration to its parameters on " + pname +
               r" (realised FPR and recall, \%, mean over five seeds). One parameter is varied at a time; the "
               r"others stay at the frozen values $c = 1.5$, $B = 5000$, $\alpha = 0.05$, $q = 0.2$. IF: Isolation "
               r"Forest; Deep IF: Deep Isolation Forest." + (NONINDEP_III if prot == "hostdisjoint" else "") + "}\n"
               r"\label{" + label + "}\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{2.2pt}" "\n"
               r"\begin{tabular}{@{}ll" + "rr" * len(meths) + "@{}}\n" r"\toprule" "\n"
               " & & " + head + r" \\" "\n" + cm + "\n"
               "Parameter & Value & " + " & ".join(["FPR & Rec."] * len(meths)) + r" \\" "\n" r"\midrule" "\n"
               + "\n".join(lines[:-1]) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table*}" "\n")
        CTX["write"](f"tab_s_sens_{prot}.tex", tex)


def quota_table():
    f = os.path.join(RESULTS_RAW, "poisoning", "poisoning_quota.json")
    if not os.path.exists(f):
        return
    R = json.load(open(f))["results"]
    from collections import defaultdict
    a = defaultdict(list)
    for r in R:
        a[(r["kind"], r["rate"], r["spread"], r["quota"])].append(r)
    qs = sorted({k[3] for k in a})
    settings = sorted({k[:3] for k in a}, key=lambda k: (k[0] != "none", k[0], k[2], k[1]))
    lines = []
    for kind, rate, spread in settings:
        name = "clean window" if kind == "none" else f"{kind}, {100 * rate:.0f}\\,\\%, {spread}"
        cells = [name]
        for q in qs:
            rs = a.get((kind, rate, spread, q), [])
            cells.append(ms([r["recall"] for r in rs], nd=1) if rs else "--")
        lines.append(" & ".join(cells) + r" \\")
    fmax = 100 * max(np.mean([r["fpr"] for r in rs]) for rs in a.values())
    CTX["macro"]("NPqFprMax", f"{fmax:.2f}")
    tex = (r"\begin{table}[t]" "\n" r"\centering" "\n"
           r"\caption{Recall (\%) of ARC-Field after ARC recalibration as a function of the per-host cap $q$ "
           r"(Protocol~II, mean\,$\pm$\,SD over five seeds). Injections come from one host (single) or are "
           r"spread over twenty hosts (distributed). The mean realised FPR did not exceed " + f"{fmax:.2f}" +
           r"\,\% in any setting.}" "\n"
           r"\label{tab:s_quota}" "\n" r"\scriptsize" "\n" r"\setlength{\tabcolsep}{3pt}" "\n"
           r"\begin{tabular}{@{}l" + "c" * len(qs) + "@{}}\n" r"\toprule" "\n"
           "Window & " + " & ".join(f"$q = {q:g}$" for q in qs) + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(lines) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table}" "\n")
    CTX["write"]("tab_s_quota.tex", tex)


_prev4 = build_all


def range_macros():
    """Ranges quoted in the text: Protocol I static FPR, feature-set collapse."""
    S = CTX["S"]
    R = S.get("unsw_random_A", {})
    base = {m: d for m, d in R.items() if "ctx" not in m and m != "Anomal-E"}
    if base:
        v = [100 * d["static"]["fpr"]["mean"] for d in base.values()]
        CTX["macro"]("NRandStFprMin", f"{min(v):.1f}")
        CTX["macro"]("NRandStFprMax", f"{max(v):.1f}")
        order = sorted((m for m in R if "ctx" not in m), key=lambda m: -R[m]["static"]["auprc"]["mean"])
        words = ["first", "second", "third", "fourth", "fifth", "sixth"]
        if "ARC-Field" in order and order.index("ARC-Field") < len(words):
            CTX["macro"]("NRandArcRankWord", words[order.index("ARC-Field")])
    meths = ["ARC-Field", "MemAE", "Dense Autoencoder", "Deep SVDD", "KitNET", "MCM", "DIF"]
    for k in ("A_noTTL", "AB", "ABC"):
        F = S.get(f"unsw_chrono_{k}", {})
        have = [m for m in meths if m in F and "arc" in F[m]]
        if not have:
            continue
        au = [100 * F[m]["static"]["auprc"]["mean"] for m in have]
        tp = [100 * F[m]["static"]["tpr_at_1fpr"]["mean"] for m in have]
        fp = [100 * F[m]["arc"]["fpr"]["mean"] for m in have]
        CTX["macro"](f"NFs{alpha(k)}AuprcMin", f"{min(au):.1f}")
        CTX["macro"](f"NFs{alpha(k)}AuprcMax", f"{max(au):.1f}")
        CTX["macro"](f"NFs{alpha(k)}TprMax", f"{max(tp):.1f}")
        CTX["macro"](f"NFs{alpha(k)}ArFprMin", f"{min(fp):.1f}")
        CTX["macro"](f"NFs{alpha(k)}ArFprMax", f"{max(fp):.1f}")
        for m in have:
            CTX["macro"](f"NFs{alpha(k)}{alpha(m)}ArFprmax", f"{100 * F[m]['arc']['fpr_bin_max']['mean']:.1f}")


def ttl_macros():
    f = os.path.join(RESULTS_PROC, "ttl_collateral.json")
    if not os.path.exists(f):
        return
    D = json.load(open(f))
    sep = D.get("ttl_separation")
    if sep:
        CTX["macro"]("NTtlAttShare", f"{100 * sep['attack_share_in_top']:.1f}")
        CTX["macro"]("NTtlBenShare", f"{100 * sep['benign_share_in_top']:.1f}")
        CTX["macro"]("NTtlValues", " and ".join(str(v) for v in sep["attack_top_sttl"]))
    for fs, tag in (("A", "A"), ("A_noTTL", "AnoTTL")):
        d = D.get(fs)
        if not d:
            continue
        CTX["macro"](f"NTtl{tag}Rho", f"{d['collateral_spearman'][0]:.2f}")
        CTX["macro"](f"NTtl{tag}FprHigh", f"{100 * d['fpr_high_attack_bins']:.1f}")
        CTX["macro"](f"NTtl{tag}FprLow", f"{100 * d['fpr_low_attack_bins']:.1f}")
        top = d["feature_shift"][:3]
        CTX["macro"](f"NTtl{tag}TopFeats", ", ".join(r"\texttt{" + r["feature"].replace("_", r"\_") + "}"
                                                     for r in top))
        for g, v in d["false_alarm_group_share_high_bins"].items():
            CTX["macro"](f"NTtl{tag}Group{alpha(g)}", f"{100 * v:.0f}")


def nf_profile_macros():
    """Corpus statistics of NF-CSE-CIC-IDS2018-v3 (Protocol V)."""
    from paths import DATA_DIR
    f = os.path.join(DATA_DIR, "nf_cicids2018_profile.json")
    if not os.path.exists(f):
        return
    P = json.load(open(f))
    big = lambda v: f"{v:,}".replace(",", "{,}")  # noqa: E731
    CTX["macro"]("NNfFlows", big(P["n_flows"]))
    CTX["macro"]("NNfIps", big(P["n_ips"]))
    CTX["macro"]("NNfAttShare", f"{100 * P['attack_share']:.1f}")
    CTX["macro"]("NNfDays", str(len(P["days"])))
    CTX["macro"]("NNfTestDays", str(len(P["days"]) - 1))
    src = [d["n_src"] for d in P["days"]]
    CTX["macro"]("NNfSrcMin", big(min(src)))
    CTX["macro"]("NNfSrcMax", big(max(src)))
    import pandas as pd
    span = (pd.Timestamp(P["days"][-1]["start_utc"]) - pd.Timestamp(P["days"][0]["start_utc"])).days
    CTX["macro"]("NNfSpanDays", str(span))
    CTX["macro"]("NNfFitDayFlows", big(P["days"][0]["n"]))
    CTX["macro"]("NNfFitDayAtt", f"{100 * P['days'][0]['attack_share']:.1f}")
    test = P["days"][1:]
    CTX["macro"]("NNfTestAttMin", f"{100 * min(d['attack_share'] for d in test):.1f}")
    CTX["macro"]("NNfTestAttMax", f"{100 * max(d['attack_share'] for d in test):.1f}")
    CTX["macro"]("NNfCleanDays", str(sum(1 for d in test if d["attack_share"] < 0.005)))
    sep = P["ttl_separation"]
    CTX["macro"]("NNfTtlAtt", f"{100 * sep['attack_share_in_top']:.1f}")
    CTX["macro"]("NNfTtlBen", f"{100 * sep['benign_share_in_top']:.1f}")
    CTX["macro"]("NNfTtlValues", " and ".join(str(v) for v in sep["attack_top_min_ttl"]))


CB_PROT = (("chrono", "Chr", "II"), ("hostdisjoint", "Host", "III"), ("nf", "Nf", "V"))
CB_ENC = (("none", "None", "None"), ("spectral", "Spec", "Spectral, robust-scaled"),
          ("afterimage", "Ai", "AfterImage~\\cite{Mirsky2018}"), ("rolx", "Rolx", "RolX roles~\\cite{Henderson2012RolX}"),
          ("roles", "Roles", "Role coordinates (ours)"))


def context_baselines():
    """Table and macros of the context-baseline study
    (results/processed/context_baselines.json)."""
    R = json.load(open(os.path.join(RESULTS_PROC, "context_baselines.json")))
    m = CTX["macro"]
    rows = []
    for cx, cg, cname in CB_ENC:
        cells, dims = [], []
        for prot, tg, _ in CB_PROT:
            P = R.get(prot, {})
            s = P.get("summary", {}).get("ARC-Field", {}).get(cx)
            g = P.get("conditioning", {}).get(cx, {})
            if s:
                dims.append(s["dim"])
                for key, kg in (("auprc", "Auprc"), ("arc_mcc", "Mcc"), ("tpr_at_1fpr", "Tpr"), ("auroc", "Auroc")):
                    m(f"NCb{tg}{cg}{kg}", f"{s[key]['mean']:.2f}")
                    m(f"NCb{tg}{cg}{kg}SD", f"{s[key]['sd']:.2f}")
                if "ARC-Field" in g:
                    m(f"NCb{tg}{cg}Cond", f"{g['ARC-Field']['auprc']:+.2f}")
                if "appended_mean" in g:
                    m(f"NCb{tg}{cg}App", f"{g['appended_mean']['auprc']:+.2f}")
                    m(f"NCb{tg}{cg}AppImp", str(g["appended_n_improved_auprc"]))
                    m(f"NCb{tg}AppN", str(g["appended_n"]))
                app = f"{g['appended_mean']['auprc']:+.1f}" if "appended_mean" in g else "--"
                cells += [f"{s['auprc']['mean']:.2f}", f"{s['arc_mcc']['mean']:.2f}", app]
            else:
                cells += ["--", "--", "--"]
        dim = "--" if not dims else (f"{dims[0]:.0f}" if max(dims) - min(dims) < 0.5 else
                                     f"{min(dims):.0f}--{max(dims):.0f}")
        rows.append([cname, dim] + cells)
    for prot, tg, _ in CB_PROT:
        k = R.get(prot, {}).get("summary", {}).get("Kitsune", {}).get("afterimage-only")
        if k:
            m(f"NCb{tg}KitsuneAuprc", f"{k['auprc']['mean']:.2f}")
        best = R.get(prot, {}).get("best_by_context", {})
        for cx, cg, _ in CB_ENC:
            if cx in best:
                m(f"NCb{tg}{cg}BestAuprc", f"{best[cx]['auprc']:.2f}")
    kit = ["Kitsune (KitNET on AfterImage)", "100"]
    for prot, tg, _ in CB_PROT:
        k = R.get(prot, {}).get("summary", {}).get("Kitsune", {}).get("afterimage-only")
        kit += [f"{k['auprc']['mean']:.2f}" if k else "--", f"{k['arc_mcc']['mean']:.2f}" if k else "--", ""]
    for cx, cg, _ in CB_ENC:
        pr = R.get("pooled_roles_vs", {}).get(cx)
        if pr:
            for key, kg in (("auprc", "Auprc"), ("arc_mcc", "Mcc"), ("auroc", "Auroc"), ("tpr_at_1fpr", "Tpr")):
                m(f"NCbPool{cg}{kg}Diff", f"{pr[key]['diff']:+.2f}")
                m(f"NCbPool{cg}{kg}Wins", str(pr[key]["wins"]))
                m(f"NCbPool{cg}{kg}N", str(pr[key]["n"]))
    # The ARC-Field runs of this study are independent refits of the frozen
    # configuration: largest |AUPRC difference| (points) against the same
    # configuration in the main tables (roles) and in the ablation / Table V
    # no-context rows (none). Computed, never typed.
    S = CTX["S"]
    pairs = []
    for prot, tag in (("chrono", "unsw_chrono_A"), ("hostdisjoint", "unsw_hostdisjoint_A"), ("nf", "nf_crossday_A")):
        s = R.get(prot, {}).get("summary", {}).get("ARC-Field", {})
        main = S.get(tag, {}).get("ARC-Field", {}).get("static", {}).get("auprc")
        if "roles" in s and main:
            pairs.append(abs(s["roles"]["auprc"]["mean"] - 100 * main["mean"]))
        if "none" in s:
            if prot == "nf":
                ref = S.get(tag, {}).get("ARC-Field (no context)", {}).get("static", {}).get("auprc", {}).get("mean")
            else:
                fs = glob.glob(os.path.join(RESULTS_ABL, f"model_{prot}", "no_context_s*.json"))
                ref = np.mean([json.load(open(f))["policies"]["static"]["auprc"] for f in fs]) if fs else None
            if ref is not None:
                pairs.append(abs(s["none"]["auprc"]["mean"] - 100 * ref))
    if pairs:
        m("NCbRefitMaxDiff", f"{max(pairs):.1f}")
    spec = [100 * json.load(open(f))["policies"]["static"]["auprc"] for f in
            glob.glob(os.path.join(os.path.dirname(RESULTS_ABL), "context_baselines", "hostdisjoint",
                                   "ARC-Field__spectral_s*.json"))]
    if spec:
        m("NCbHostSpecAuprcMin", f"{min(spec):.2f}")
        m("NCbHostSpecAuprcMax", f"{max(spec):.2f}")
    # bold the best conditioned AUPRC and MCC per protocol
    for j in range(len(CB_PROT)):
        for off in (0, 1):
            col = 2 + 3 * j + off
            vals = [float(r[col]) if r[col] not in ("--", "") else -1 for r in rows]
            b = int(np.argmax(vals))
            if vals[b] >= 0:
                rows[b][col] = "\\textbf{" + rows[b][col] + "}"
    lines = [" & ".join(r) + " \\\\" for r in rows] + ["\\midrule", " & ".join(kit) + " \\\\"]
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{Prot.~{rn}}}" for _, _, rn in CB_PROT)
    rules = "".join(f"\\cmidrule(lr){{{3 + 3 * j}-{5 + 3 * j}}}" for j in range(len(CB_PROT)))
    sub = " & ".join(["AUPRC & MCC & $\\Delta_{\\mathrm{app}}$"] * len(CB_PROT))
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Endpoint-context encodings under identical detectors and "
           "protocols. AUPRC (static scores) and MCC after the ARC recalibration of ARC-Field with each encoding as "
           "the conditioning input of its field, in \\%, mean over seeds (five for Protocols~II and~III, three for "
           "Protocol~V). $\\Delta_{\\mathrm{app}}$: mean change in AUPRC when the same encoding is instead appended "
           "to the input columns of six detectors (MemAE, Deep SVDD, dense AE, Isolation Forest, KitNET, MCM), "
           "relative to the same detectors without context. Dim: encoding dimension (RolX: $2k+4$ with $k$ roles "
           "chosen by description length). Best conditioned value per column in bold. The ARC-Field runs of this "
           "study are independent refits of the frozen configuration; their AUPRC differs from the same "
           "configuration in the main comparison and ablation tables by at most \\NCbRefitMaxDiff\\ points. ``Spectral, robust-scaled'' scales the "
           "coordinate like every encoding here, whereas Table~\\ref{tab:ablation} feeds it unscaled; on "
           "Protocol~III the scaling moves the zero vector of unseen endpoints outside the fitted range, and its "
           "AUPRC ranges from \\NCbHostSpecAuprcMin\\ to \\NCbHostSpecAuprcMax\\,\\% across seeds." + NONINDEP_III
           + "}\n"
           "\\label{tab:ctxbase}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lc" + "ccc" * len(CB_PROT) + "@{}}\n\\toprule\n"
           " & & " + head + " \\\\\n" + rules + "\n"
           "Encoding & Dim & " + sub + " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\end{table}\n")
    CTX["write"]("tab_context_baselines.tex", tex)


def hikari_macros():
    """Corpus counts (from the derived cache) and test-stream sizes (from the
    Protocol VI result files)."""
    from paths import DATA_DIR
    import data_hikari as dh
    big = lambda v: f"{int(v):,}".replace(",", "{,}")  # noqa: E731
    d = np.load(os.path.join(DATA_DIR, "hikari2021_corpus.npz"), allow_pickle=False)
    cat, normal = d["category"], d["is_normal"]
    m = CTX["macro"]
    m("NHkFlows", big(len(cat)))
    m("NHkAttShare", f"{100 * (1 - normal.mean()):.1f}")
    m("NHkAttFlows", big((~normal).sum()))
    for c, tg in (("Background", "Bg"), ("Benign", "Syn"), ("Bruteforce", "Bf"), ("Bruteforce-XML", "Bfx"),
                  ("Probing", "Probe"), ("XMRIGCC CryptoMiner", "Miner")):
        m(f"NHk{tg}Flows", big((cat == c).sum()))
    m("NHkTierA", str(len(dh.TIER_A)))
    m("NHkTierB", str(len(dh.TIER_B)))
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(RESULTS_RAW, "hikari_shift_A", "ARC-Field_s*.json"))]
    if runs:
        m("NHkTest", big(round(np.mean([r["n_test"] for r in runs]), -3)))
        m("NHkTestAtt", f"{100 * np.mean([r['test_attack_frac'] for r in runs]):.1f}")
    for tag, tg in (("hikari_shift_A", "Hks"), ("hikari_random_A", "Hkr")):
        s = CTX["S"].get(tag, {})
        vals = [(v["static"]["auprc"]["mean"], k) for k, v in s.items() if "static" in v and "auprc" in v["static"]]
        if vals:
            best, who = max(vals)
            m(f"N{tg}BestAuprc", f"{100 * best:.1f}")
            m(f"N{tg}MaxAuroc", f"{100 * max(v['static']['auroc']['mean'] for v in s.values() if 'static' in v):.1f}")


RA_PROT = (("dev", "Dev", "D"), ("chrono", "Chr", "II"), ("hostdisjoint", "Host", "III"), ("nf", "Nf", "V"))
RA_VAR = (("M0", "role coordinates", "--"), ("M1", "role coordinates", "role coordinates"),
          ("M2", "role coordinates", "RolX"), ("M3", "roles + RolX", "--"), ("M4", "roles + RolX", "roles + RolX"))


def role_anomaly():
    """Supplementary table and macros of the pre-registered role-anomaly
    variants (results/role_anomaly/<protocol>/<variant>_s<seed>.json)."""
    from paths import PROJECT
    base = os.path.join(PROJECT, "results", "role_anomaly")
    V = {}
    for prot, tg, _ in RA_PROT:
        for f in glob.glob(os.path.join(base, prot, "*.json")):
            d = json.load(open(f))
            s, a = d["policies"]["static"], d["policies"]["arc"]
            s, a = s.get("pooled", s), a.get("pooled", a)
            V.setdefault((prot, d["variant"]), {})[d["seed"]] = (100 * s["auprc"], 100 * s["tpr_at_1fpr"],
                                                               100 * a["mcc"], 100 * a["fpr"])
    if not V:
        return
    m = CTX["macro"]
    vword = {"M0": "MZero", "M1": "MOne", "M2": "MTwo", "M3": "MThree", "M4": "MFour"}   # macro names: letters only
    rows = []
    for v, cond, anom in RA_VAR:
        cells = []
        for prot, tg, _ in RA_PROT:
            r = V.get((prot, v))
            if not r:
                cells += ["--", "--"]
                continue
            mean = np.mean(list(r.values()), axis=0)
            cells += [f"{mean[0]:.2f}", f"{mean[2]:.2f}"]
            for i, kg in enumerate(("Auprc", "Tpr", "Mcc", "Fpr")):
                m(f"NRa{tg}{vword[v]}{kg}", f"{mean[i]:.2f}")
            if v != "M0" and (prot, "M0") in V:
                s0 = V[(prot, "M0")]
                common = sorted(set(r) & set(s0))
                for i, kg in enumerate(("Auprc", "Tpr", "Mcc")):
                    dd = [r[s][i] - s0[s][i] for s in common]
                    m(f"NRa{tg}{vword[v]}{kg}Diff", f"{np.mean(dd):+.1f}")
                    m(f"NRa{tg}{vword[v]}{kg}Wins", str(sum(x > 0 for x in dd)))
                m(f"NRa{tg}N", str(len(common)))
        rows.append([v, cond, anom] + cells)
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{Prot.~{rn}}}" for _, _, rn in RA_PROT)
    rules = "".join(f"\\cmidrule(lr){{{4 + 2 * j}-{5 + 2 * j}}}" for j in range(len(RA_PROT)))
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Role-anomaly extension (exploratory: designed after the "
           "comparison of Section~\\ref{subsec:ctxbase} and fixed before its runs, but evaluated on the protocols "
           "that motivated it; Protocol~III also reuses the development attacks of Protocol~D). Conditional: context that "
           "conditions the field. Role test: context scored by a second, separately calibrated masked field; "
           "the two tests are recalibrated at $\\alpha/2$ each and their alarms are combined, so the "
           "false-positive rate stays below $\\alpha$ whenever each recalibration holds. AUPRC (static scores) "
           "and MCC after ARC recalibration, in \\%, mean over seeds. M4 was designated in advance; M0 is "
           "the primary ARC-Field configuration evaluated in the main text. None of the extensions is part of "
           "ARC-Field.}\n\\label{tab:s_role_anomaly}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lll" + "cc" * len(RA_PROT) + "@{}}\n\\toprule\n & & & " + head + " \\\\\n" + rules
           + "\nVariant & Conditional & Role test & " + " & ".join(["AUPRC & MCC"] * len(RA_PROT))
           + " \\\\\n\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in rows)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    CTX["write"]("tab_s_role_anomaly.tex", tex)


def dev_overlap_macros():
    """Development/test overlap on UNSW-NB15 (results/processed/dev_overlap.json,
    written by code/tools/audit/issue1_overlap.py): Protocol D streams the first
    two hours of campaign 1, which hold every campaign-1 attack."""
    f = os.path.join(RESULTS_PROC, "dev_overlap.json")
    if not os.path.exists(f):
        return
    D = json.load(open(f))
    big = lambda v: f"{int(v):,}".replace(",", "{,}")  # noqa: E731
    m = CTX["macro"]
    m("NOvlDevN", big(D["dev_stream"]["n"]))
    m("NOvlDevAtt", big(D["dev_stream"]["n_attack"]))
    m("NOvlCampAtt", big(D["campaign1_attacks"]))
    m("NOvlCampAttInDev", big(D["campaign1_attacks_in_dev_stream"]))
    hh = D.get("halfhour_attack_share", {})
    for k, tg in (("dev_stream", "Dev"), ("campaign2", "CampTwo")):
        if k in hh:
            m(f"NHalfHour{tg}AttMin", f"{100 * hh[k]['min']:.0f}")
            m(f"NHalfHour{tg}AttMax", f"{100 * hh[k]['max']:.0f}")
    ev = [r for r in D["rows"] if r["part"] == "evaluated"]
    for prot, tg in (("I (random)", "I"), ("III (host-disjoint)", "III")):
        rs = [r for r in ev if r["protocol"] == prot]
        if not rs:
            continue
        share = [100 * r["share_of_attacks_shared"] for r in rs]
        m(f"NOvl{tg}AttShareMin", f"{min(share):.0f}")
        m(f"NOvl{tg}AttMin", big(min(r["shared_attack"] for r in rs)))
        m(f"NOvl{tg}AttMax", big(max(r["shared_attack"] for r in rs)))
        m(f"NOvl{tg}BenMin", big(min(r["shared_benign"] for r in rs)))
        m(f"NOvl{tg}BenMax", big(max(r["shared_benign"] for r in rs)))
        m(f"NOvl{tg}Seeds", str(len(rs)))


def protocol_macros():
    """Protocol sizes, corpus counts and feature-tier sizes quoted in the text,
    read from the raw result files, the corpus profiles and the code, plus the
    Protocol II settings of the published GraphINR-IDS pipeline (parsed from
    the vendored GraphINR-IDS harness, not typed)."""
    import ast
    from paths import DATA_DIR
    import data_unsw as du
    import data_nf as dn
    import data_cic as dc
    m = CTX["macro"]
    big = lambda v: f"{int(v):,}".replace(",", "{,}")  # noqa: E731
    rnd = lambda v: big(round(float(v), -3))  # noqa: E731
    m("NUnswTierA", str(len(du.TIER_A)))
    m("NUnswTierB", str(len(du.TIER_B)))
    m("NUnswTierC", str(len(du.TIER_C)))
    m("NUnswFeatAll", str(len(du.ALL36)))
    m("NNfTierA", str(len(dn.FEATURE_SETS["A"])))
    m("NCicFeat", str(len(dc.FEATURES)))
    prof = json.load(open(os.path.join(DATA_DIR, "unsw_full_profile.json")))
    m("NUnswFlows", big(prof["n_flows"]))
    m("NUnswAttFlows", big(prof["n_attack"]))
    m("NUnswIps", str(prof["n_ips"]))
    from paths import PROJECT
    cf = os.path.join(PROJECT, "reproducibility", "causality_check.json")
    if os.path.exists(cf):
        cc = json.load(open(cf))
        e = max(cc["max_abs_error"].values())
        mant, ex = f"{e:.1e}".split("e")
        m("NCausalN", big(cc["n_flows_checked"]))
        m("NCausalMaxErr", f"{mant}\\times10^{{{int(ex)}}}")
    cic = np.load(os.path.join(DATA_DIR, "cic_arcfield_corpus.npz"), allow_pickle=True)
    m("NCicFlows", big(len(cic["is_normal"])))
    # columns of the MachineLearningCVE day files: destination port, flow statistics, label
    heads = {tuple(c.strip() for c in open(os.path.join(dc.CIC_RAW_DIR, f), encoding="utf-8",
                                          errors="replace").readline().split(","))
             for f, _ in dc.FILES if os.path.exists(os.path.join(dc.CIC_RAW_DIR, f))}
    if heads:
        assert len(heads) == 1, "CIC-IDS2017 day files differ in their columns"
        h = heads.pop()
        assert h[0] == "Destination Port" and h[-1] == "Label", h
        m("NCicStatCols", str(len(h) - 2))
    for tag, tg in (("unsw_random_A", "Rand"), ("unsw_chrono_A", "Chr"), ("unsw_hostdisjoint_A", "Host")):
        rs = [json.load(open(f)) for f in glob.glob(os.path.join(RESULTS_RAW, tag, "ARC-Field_s*.json"))]
        if not rs:
            continue
        n = [r["n_test"] for r in rs]
        m(f"N{tg}TestN", big(n[0]) if len(set(n)) == 1 else rnd(np.mean(n)))
        m(f"N{tg}TestAtt", f"{100 * np.mean([r['test_attack_frac'] for r in rs]):.{1 if tg == 'Chr' else 0}f}")
        m(f"N{tg}Stride", str(rs[0]["protocol_meta"]["stride"]))
        m("NFitN", big(rs[0]["n_train"]))
        m("NValN", big(rs[0]["n_val"]))
        m("NCalN", big(rs[0]["n_cal"]))
    words = {3: "three", 5: "five"}
    for tag, tg in (("unsw_random_A", "Rand"), ("unsw_chrono_A", "Chr"), ("unsw_hostdisjoint_A", "Host"),
                    ("cic_crossday", "Cic"), ("nf_crossday_A", "Nf"), ("hikari_shift_A", "Hk")):
        n = len(glob.glob(os.path.join(RESULTS_RAW, tag, "ARC-Field_s*.json")))
        if n:
            m(f"N{tg}SeedsWord", words.get(n, str(n)))
    for tag, tg in (("cic_crossday", "Cic"), ("nf_crossday_A", "Nf")):
        rs = [json.load(open(f)) for f in glob.glob(os.path.join(RESULTS_RAW, tag, "ARC-Field_s*.json"))]
        if rs:
            sizes = [v for r in rs for v in r["test_sizes"].values()]
            m(f"N{tg}DayMin", big(min(sizes)))
            m(f"N{tg}DayMax", big(max(sizes)))
    # GraphINR-IDS (published) Protocol II settings, from the vendored code
    src = open(os.path.join(HERE, "..", "src", "graphinr_ids_harness.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    fdefs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def defaults(fn):
        a = fdefs[fn].args
        return {p.arg: ast.literal_eval(d) for p, d in zip(a.args[-len(a.defaults):], a.defaults)}
    ch, ld = defaults("make_chronological_split"), defaults("load_unsw_corpus")
    feats = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "UNSW_FEATURES" for t in n.targets))
    m("NGiRowsPerFile", big(ld["rows_per_file"]))
    m("NGiFlows", big(4 * ld["rows_per_file"]))
    m("NGiFeat", str(len(feats)))
    m("NGiTestBen", big(ch["n_test_normal"]))
    m("NGiTestAtt", big(ch["n_test_attack"]))
    m("NGiTestAttShare", f"{100 * ch['n_test_attack'] / (ch['n_test_attack'] + ch['n_test_normal']):.0f}")
    for key, tg in (("spectral_k", "K"), ("max_nodes", "MaxNodes"), ("num_frequencies", "Freqs")):
        v = int(re.search(rf'"{key}":\s*(\d+)', src).group(1))
        m(f"NGi{tg}", big(v))
    sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
    from common import REUSED_BASELINES, NEW_BASELINES
    nw = {4: "Four", 11: "eleven"}
    m("NnReusedWord", nw.get(len(REUSED_BASELINES), str(len(REUSED_BASELINES))))
    m("NnNewWord", nw.get(len(NEW_BASELINES), str(len(NEW_BASELINES))))


def context_drift_macros():
    D = json.load(open(os.path.join(RESULTS_PROC, "context_drift.json")))
    for prot, tg, _ in CB_PROT:
        for enc, eg in (("flow features", "Flow"), ("roles", "Roles"), ("afterimage", "Ai"), ("rolx", "Rolx")):
            v = D.get(prot, {}).get(enc)
            if v:
                CTX["macro"](f"NDr{tg}{eg}Auc", f"{v['domain_auc']:.3f}")
                CTX["macro"](f"NDr{tg}{eg}Shift", f"{v['median_shift']:.2f}")


def build_all():  # noqa: F811
    _prev4()
    for fn in (calibration_extra, quota_table, range_macros, ttl_macros, nf_profile_macros, nf_extra_macros,
               context_baselines, context_drift_macros, hikari_macros, role_anomaly, dev_overlap_macros,
               protocol_macros):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print("tables_extra:", fn.__name__, "skipped:", repr(e))
