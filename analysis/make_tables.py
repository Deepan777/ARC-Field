"""Generate every manuscript table (manuscript/tables/*.tex, mirrored to
tables/) and the number macros (manuscript/tables/numbers.tex) from processed
results. Run after aggregate.py. No number in the manuscript is typed by hand.
"""
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import PROJECT, RESULTS_ABL, RESULTS_PROC, RESULTS_RAW, RESULTS_STATS, TAB_DIR, DATA_DIR  # noqa

MS_TAB = os.path.join(PROJECT, "manuscript", "tables")
os.makedirs(MS_TAB, exist_ok=True)
S = json.load(open(os.path.join(RESULTS_PROC, "summary.json")))
PT = json.load(open(os.path.join(RESULTS_STATS, "paired_tests.json")))
FR_PATH = os.path.join(RESULTS_STATS, "friedman_nemenyi.json")
FR = json.load(open(FR_PATH)) if os.path.exists(FR_PATH) else {}
MACROS = {}

ORDER = ["ARC-Field", "GraphINR-IDS", "Anomal-E", "DIF", "Isolation Forest", "Local Outlier Factor", "One-Class SVM",
         "Dense Autoencoder", "LSTM Autoencoder", "Deep SVDD", "DAGMM", "MemAE", "USAD",
         "Transformer Autoencoder", "Diffusion (DDPM) detector", "KitNET", "MCM"]
CTXV = ["Isolation Forest +ctx", "Dense Autoencoder +ctx", "Deep SVDD +ctx", "KitNET +ctx", "MCM +ctx"]
NAME = {"ARC-Field": "ARC-Field (ours)", "GraphINR-IDS": "GraphINR-IDS~\\cite{Deepanramkumar2026GraphINR}",
        "Anomal-E": "Anomal-E~\\cite{Caville2022}",
        "DIF": "Deep IF~\\cite{Xu2023DIF}",
        "Isolation Forest": "Isolation Forest", "Local Outlier Factor": "LOF", "One-Class SVM": "OC-SVM",
        "Dense Autoencoder": "Dense AE", "LSTM Autoencoder": "LSTM-AE", "Deep SVDD": "Deep SVDD",
        "DAGMM": "DAGMM", "MemAE": "MemAE", "USAD": "USAD", "Transformer Autoencoder": "Transformer AE",
        "Diffusion (DDPM) detector": "DDPM detector", "KitNET": "KitNET~\\cite{Mirsky2018}",
        "MCM": "MCM~\\cite{Yin2024MCM}"}
for v in CTXV:
    NAME[v] = NAME[v.replace(" +ctx", "")].split("~")[0] + " +ctx"
CTXV = CTXV + ["ARC-Field (no context)"]            # Protocol V ablation row
NAME["ARC-Field (no context)"] = "ARC-Field $-$ context"


def macro(name, value):
    name = re.sub(r"[^A-Za-z]", "", name)
    MACROS[name] = value


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}"


def ms(d, key, nd=2, scale=100):
    if d is None or key not in d:
        return "--"
    m, s, n = d[key]["mean"], d[key]["sd"], len(d[key]["values"])
    if n >= 3:
        return f"{scale * m:.{nd}f}\\,$\\pm$\\,{scale * s:.{nd}f}"
    return f"{scale * m:.{nd}f}"


def fit_width(tex):
    """Wrap every tabular in adjustbox{max width=\\linewidth}: a table that
    would run past the text width is scaled down to fit, any other table is
    left at its natural size."""
    if "adjustbox" in tex:
        return tex
    return re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})",
                  lambda m: "\\begin{adjustbox}{max width=\\linewidth}\n" + m.group(1) + "\n\\end{adjustbox}",
                  tex, flags=re.S)


def write(name, tex):
    tex = fit_width(tex)
    for d in (MS_TAB, TAB_DIR):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.write(tex)


def bold_best(rows, col, higher=True):
    vals = []
    for r in rows:
        try:
            vals.append(float(r[col].split("\\,")[0]))
        except (ValueError, AttributeError):
            vals.append(np.nan)
    if np.all(np.isnan(vals)):
        return
    j = int(np.nanargmax(vals) if higher else np.nanargmin(vals))
    rows[j][col] = "\\textbf{" + rows[j][col].split("\\,")[0] + "}" + (
        "\\," + "\\,".join(rows[j][col].split("\\,")[1:]) if "\\," in rows[j][col] else "")


# ---------------------------------------------------------------------------
# dataset statistics
# ---------------------------------------------------------------------------
prof = json.load(open(os.path.join(DATA_DIR, "unsw_full_profile.json")))
# Campaign gap = idle interval from the last record of campaign 1 to the first
# record of campaign 2 (largest gap between consecutive Stime values);
# start-to-start interval reported alongside so the definition is explicit.
macro("NgapHours", f"{prof['session_gap_hours']:.0f}")
macro("NgapDays", f"{prof['session_gap_hours'] / 24:.0f}")
macro("NgapStartDays", f"{(prof['session2']['t0'] - prof['session1']['t0']) / 86400:.1f}")
macro("NsOneFlows", f"{prof['session1']['n']:,}".replace(",", "{,}"))
macro("NsTwoFlows", f"{prof['session2']['n']:,}".replace(",", "{,}"))
macro("NsOneHours", f"{prof['session1']['hours']:.1f}")
macro("NsTwoHours", f"{prof['session2']['hours']:.1f}")


# ---------------------------------------------------------------------------
# main comparison tables (one per UNSW protocol)
# ---------------------------------------------------------------------------
def main_table(tag, label, caption):
    s = S.get(tag)
    if not s:
        return
    rows = []
    for meth in ORDER + CTXV:
        if meth not in s:
            continue
        st, ar = s[meth].get("static"), s[meth].get("arc")
        rows.append({"name": NAME[meth], "auroc": ms(st, "auroc"), "auprc": ms(st, "auprc"),
                     "tpr1": ms(st, "tpr_at_1fpr"), "fpr_static": ms(st, "fpr"),
                     "f1": ms(ar, "f1"), "rec": ms(ar, "recall"), "fpr": ms(ar, "fpr"), "mcc": ms(ar, "mcc"),
                     "ctx": meth in CTXV})
    main = [r for r in rows if not r["ctx"]]
    for c in ("auroc", "auprc", "tpr1", "f1", "mcc"):
        bold_best(main, c)
    lines = []
    for r in rows:
        if r["ctx"] and not any("midrule" in l for l in lines[-1:]) and r is rows[[x["ctx"] for x in rows].index(True)]:
            lines.append("\\midrule")
        lines.append(" & ".join([r["name"], r["auroc"], r["auprc"], r["tpr1"], r["fpr_static"], r["f1"], r["rec"],
                                 r["fpr"], r["mcc"]]) + " \\\\")
    tex = ("\\begin{sidewaystable}\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{@{}lcccc|cccc@{}}\n\\toprule\n"
           " & \\multicolumn{3}{c}{Ranking (threshold-free)} & Static & \\multicolumn{4}{c}{ARC recalibration} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}\\cmidrule(l){6-9}\n"
           "Detector & AUROC & AUPRC & TPR@1\\% & FPR & F1 & Recall & FPR & MCC \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{sidewaystable}\n")
    write(f"tab_main_{tag}.tex", tex)


main_table("unsw_chrono_A", "tab:main_chrono",
           "Protocol~II (cross-campaign, \\NgapDays-day gap), Tier~A features. Mean\\,$\\pm$\\,SD over seeds, in \\%. "
           "Ranking metrics are computed on the scores of the static policy; the operating point uses the "
           "ARC recalibration of Algorithm~\\ref{alg:recal}, which is applied identically to every detector. "
           "Best per column in bold (``+ctx'' rows excluded). The static FPR column shows the realised false-positive "
           "rate of the threshold carried over from the reference period (target 5\\,\\%).")
# Protocols I and III evaluate on campaign-1 attacks that the development
# protocol D streamed (results/processed/dev_overlap.json): non-independent
NONINDEP = ("Not an independent test: every evaluated attack record of this protocol was also in the stream of "
            "the development protocol on which the hyper-parameters were selected. ")
main_table("unsw_hostdisjoint_A", "tab:main_host",
           "Protocol~III (host-disjoint): half of the benign client hosts are never seen during fitting. "
           + NONINDEP + "Layout as in Table~\\ref{tab:main_chrono}.")
main_table("unsw_random_A", "tab:main_random",
           "Protocol~I (in-distribution reference, campaign~1). " + NONINDEP
           + "Layout as in Table~\\ref{tab:main_chrono}.")
main_table("nf_crossday_A", "tab:main_nf",
           "Protocol~V (NF-CSE-CIC-IDS2018-v3, fit on the benign flows of the first capture day, nine later "
           "days streamed and recalibrated separately), Tier~A features. Mean\\,$\\pm$\\,SD over three seeds, "
           "in \\%. Ranking metrics are averages of the per-day values; operating-point metrics pool the "
           "decisions of all days. Layout otherwise as in Table~\\ref{tab:main_chrono}; ``ARC-Field $-$ "
           "context'' is the same field without role coordinates.")
main_table("hikari_shift_A", "tab:main_hikari",
           "Protocol~VI (HIKARI-2021, encrypted HTTPS attacks; flow level): fit on the real background traffic, "
           "test on the remaining background, the synthetic benign browsing and all attacks in a seeded random "
           "order. Tier~A features; no endpoint context is available (the public flow table has no timestamps). "
           "Mean\\,$\\pm$\\,SD over three seeds, in \\%. Layout as in Table~\\ref{tab:main_chrono}.")
main_table("hikari_random_A", "tab:s_main_hikari_random",
           "Protocol~VI-a (HIKARI-2021, random split of both benign sources). Layout as in "
           "Table~\\ref{tab:main_hikari}.")


# ---------------------------------------------------------------------------
# calibration policy table
# ---------------------------------------------------------------------------
POLNAME = {"static": "Static (reference period)", "burnin_thr": "Burn-in threshold",
           "burnin_full": "Burn-in mean/SD", "burnin_oracle": "Burn-in mean/SD, benign-labelled$^\\dagger$",
           "ema": "EWMA streaming", "arc": "ARC (Alg.~\\ref{alg:recal})"}


def policy_table(tag="unsw_chrono_A", label="tab:policies", fname="tab_policies.tex",
                 head="Calibration policies on Protocol~II", span="12-hour test stream"):
    s = S.get(tag)
    if not s:
        return
    dets = [m for m in ["ARC-Field", "GraphINR-IDS", "MemAE", "Dense Autoencoder", "Deep SVDD",
                        "Isolation Forest"] if m in s]
    lines = []
    for m in dets:
        pols = [p for p in ["static", "burnin_thr", "burnin_full", "burnin_oracle", "ema", "arc"] if p in s[m]]
        for i, p in enumerate(pols):
            d = s[m][p]
            lines.append(" & ".join([NAME[m].split("~")[0] if i == 0 else "", POLNAME[p], ms(d, "fpr"),
                                     ms(d, "fpr_bin_max"), ms(d, "recall"), ms(d, "f1"), ms(d, "mcc"),
                                     ms(d, "auroc")]) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    tex = ("\\begin{sidewaystable}\n\\centering\n\\caption{" + head + " (mean\\,$\\pm$\\,SD over "
           "seeds, \\%). FPR$_{\\max}$ is the largest realised false-positive rate over half-hour bins of the "
           + span + ". $^\\dagger$Uses benign labels of the window; reference only. AUROC changes "
           "between policies only through restandardisation.}\n\\label{" + label + "}\n\\scriptsize\n"
           "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
           "Detector & Policy & FPR & FPR$_{\\max}$ & Recall & F1 & MCC & AUROC \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{sidewaystable}\n")
    write(fname, tex)


policy_table()
policy_table("nf_crossday_A", "tab:s_policies_nf", "tab_s_policies_nf.tex",
             "Calibration policies on Protocol~V (NF-CSE-CIC-IDS2018-v3)",
             "nine test days (each day recalibrated on its own first window)")

# ---------------------------------------------------------------------------
# headline macros
# ---------------------------------------------------------------------------
def put(prefix, tag, meth, pol, key, nd=2, scale=100):
    try:
        v = S[tag][meth][pol][key]
        macro(prefix, f"{scale * v['mean']:.{nd}f}")
        macro(prefix + "SD", f"{scale * v['sd']:.{nd}f}")
    except KeyError:
        pass


for tag, tg in (("unsw_chrono_A", "Chr"), ("unsw_hostdisjoint_A", "Host"), ("unsw_random_A", "Rand"),
                ("nf_crossday_A", "Nf"), ("hikari_shift_A", "Hks"), ("hikari_random_A", "Hkr")):
    for meth, mg in (("ARC-Field", "Arc"), ("MemAE", "Mem"), ("Anomal-E", "Anom"), ("DIF", "Dif"),
                     ("Deep SVDD", "Dsvdd"), ("Diffusion (DDPM) detector", "Ddpm"), ("Dense Autoencoder", "Dae"),
                     ("KitNET", "Kit"), ("MCM", "Mcm"), ("Isolation Forest", "If"),
                     ("ARC-Field (no context)", "ArcNoctx")):
        for pol, pg in (("static", "St"), ("arc", "Ar"), ("burnin_full", "Bf"), ("burnin_thr", "Bt")):
            for key, kg in (("auroc", "Auroc"), ("auprc", "Auprc"), ("f1", "Fone"), ("fpr", "Fpr"),
                            ("recall", "Rec"), ("mcc", "Mcc"), ("fpr_bin_max", "Fprmax"), ("tpr_at_1fpr", "Tpr"),
                            ("fpr_day_max", "Fprday")):
                put(f"N{tg}{mg}{pg}{kg}", tag, meth, pol, key)

# statistics macros
for tag, tg in (("unsw_chrono_A", "Chr"), ("unsw_hostdisjoint_A", "Host"), ("nf_crossday_A", "Nf"),
                ("hikari_shift_A", "Hks"), ("hikari_random_A", "Hkr")):
    for metric, mg in (("auroc", "Auroc"), ("auprc", "Auprc"), ("mcc", "Mcc")):
        rows = PT.get(tag, {}).get("static" if metric != "mcc" else "arc", {}).get(metric, [])
        if rows:
            wins = sum(1 for r in rows if r["mean_diff"] > 0)
            sig = sum(1 for r in rows if r["mean_diff"] > 0 and r["t_p_holm"] < 0.05)
            macro(f"N{tg}{mg}Wins", str(wins))
            macro(f"N{tg}{mg}Sig", str(sig))
            macro(f"N{tg}{mg}Ncomp", str(len(rows)))
# NFr*: primary ranking over the independent Protocols II and V; NFrAll*:
# ranking over Protocols I, II, III and V (supplement; I and III reuse the
# development attacks of Protocol D, see aggregate.FR_SCOPES)
_FR_ALL_PATH = os.path.join(RESULTS_STATS, "friedman_nemenyi_all.json")
_FR_ALL = json.load(open(_FR_ALL_PATH)) if os.path.exists(_FR_ALL_PATH) else {}
for _pref, _F in (("NFr", FR), ("NFrAll", _FR_ALL)):
    for metric, mg in (("auroc", "Auroc"), ("auprc", "Auprc"), ("mcc_arc", "Mcc")):
        if metric in _F:
            f = _F[metric]
            macro(f"{_pref}{mg}P", f"{f['p']:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
                  if f["p"] < 1e-3 else f"{f['p']:.3f}")
            macro(f"{_pref}{mg}Cd", f"{f['critical_difference']:.2f}")
            macro(f"{_pref}{mg}Blocks", str(f["n_blocks"]))
            macro(f"{_pref}{mg}ArcRank", f"{f['avg_rank'].get('ARC-Field', float('nan')):.2f}")



import tables_extra  # noqa: E402  (further tables; shares MACROS/write via set_context)
tables_extra.set_context(macro, write, NAME, S)
tables_extra.build_all()

# development-protocol selection macros (logs/dev)
DEV = os.path.join(PROJECT, "logs", "dev")
try:
    v1 = json.load(open(os.path.join(DEV, "variant_selection.json")))["summary"]
    v2 = json.load(open(os.path.join(DEV, "variant_selection2.json")))["summary"]
    cs = json.load(open(os.path.join(DEV, "calibrator_selection_FINAL.json")))
    macro("NdevSelAuprc", pct(v2["homo_ctx_h256"]["auprc"]))
    macro("NdevHeteroAuprc", pct(v1["z2"]["auprc"]))
    macro("NdevDaeAuprc", pct(v1["DenseAE"]["auprc"]))
    macro("NdevArcMcc", f"{cs['selected']['mcc']:.3f}")
    macro("NdevBfMcc", f"{[r for r in cs['all'] if r['policy'] == 'burnin_full'][0]['mcc']:.3f}")
    rm = [r for r in cs["all"] if r["policy"] == "arc" and r["cfg"].get("eta_tau", 0) > 0]
    macro("NdevRmMcc", f"{max(r['mcc'] for r in rm):.3f}")
except Exception as e:
    print("dev macros skipped", e)

with open(os.path.join(MS_TAB, "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% generated by code/analysis/make_tables.py -- do not edit\n")
    for k in sorted(MACROS):
        f.write(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}\n")
shutil.copy(os.path.join(MS_TAB, "numbers.tex"), os.path.join(TAB_DIR, "numbers.tex"))
print(len(MACROS), "macros;", "tables:", sorted(os.listdir(MS_TAB)))
