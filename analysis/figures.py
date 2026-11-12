"""Publication figures (PDF + PNG) from raw and processed results.

Palette: validated reference categorical palette (blue, orange, aqua, yellow;
light mode) with every series also carrying a marker shape and a direct label,
so identity never rests on colour alone. Other detectors are drawn in grey
(emphasis form). One y-axis per panel throughout.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import FIG_DIR, RESULTS_ABL, RESULTS_PROC, RESULTS_RAW, RESULTS_STATS, PROJECT  # noqa: E402


def _gap_days():
    """Idle interval between the two UNSW-NB15 campaigns, in whole days (as NgapDays)."""
    from paths import DATA_DIR
    prof = json.load(open(os.path.join(DATA_DIR, "unsw_full_profile.json")))
    return f"{prof['session_gap_hours'] / 24:.0f}"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, INK2 = "#b9b8b3", "#0b0b0b", "#52514e"
# Figures are drawn at their printed width (double column 7.0 in), in a Times
# face to match the text, so that 8 pt in the figure is 8 pt on the page.
WIDE = 7.0
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.4})
HL = {"ARC-Field": (BLUE, "o", "ARC-Field"), "MemAE": (ORANGE, "s", "MemAE"),
      "Dense Autoencoder": (AQUA, "^", "Dense AE"), "KitNET": (YELLOW, "D", "KitNET")}
SHORT = {"Isolation Forest": "IF", "Local Outlier Factor": "LOF", "One-Class SVM": "OC-SVM",
         "Dense Autoencoder": "Dense AE", "LSTM Autoencoder": "LSTM-AE", "Deep SVDD": "Deep SVDD",
         "DAGMM": "DAGMM", "MemAE": "MemAE", "USAD": "USAD", "Transformer Autoencoder": "Transf. AE",
         "Diffusion (DDPM) detector": "DDPM", "KitNET": "KitNET", "MCM": "MCM", "ARC-Field": "ARC-Field",
         "Anomal-E": "Anomal-E", "DIF": "Deep IF", "GraphINR-IDS": "GraphINR-IDS"}
os.makedirs(FIG_DIR, exist_ok=True)
MS_FIG = os.path.join(PROJECT, "manuscript", "figures")
os.makedirs(MS_FIG, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300 if ext == "png" else None)
    fig.savefig(os.path.join(MS_FIG, f"{name}.pdf"))
    plt.close(fig)
    print("figure", name)


def runs(tag):
    out = {}
    for f in glob.glob(os.path.join(RESULTS_RAW, tag, "*.json")):
        r = json.load(open(f))
        out.setdefault(r["method"], []).append(r)
    return out


# ---------------------------------------------------------------------------
def fig_time_resolved():
    R = runs("unsw_chrono_A")
    if not R:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5), sharey=True)
    for ax, pol, title in zip(axes, ("static", "arc"),
                              ("(a) Static threshold from the reference period",
                               "(b) ARC recalibration on the first 5,000 records")):
        order = [m for m in R if m not in HL and not m.endswith("+ctx")] + list(HL)
        for meth in order:
            if meth not in R:
                continue
            curves = []
            for r in R[meth]:
                tr = r["policies"][pol].get("time_resolved")
                if not tr:
                    continue
                curves.append([b["fpr"] if b["fpr"] is not None and b["n_benign"] >= 200 else np.nan
                               for b in tr["bins"]])
            L = min(len(c) for c in curves)
            C = np.array([c[:L] for c in curves]) * 100
            x = np.arange(L) * 0.5
            if meth in HL:
                col, mk, lab = HL[meth]
                ax.plot(x, np.nanmean(C, 0), color=col, marker=mk, markersize=3.2, label=lab, zorder=3)
            else:
                ax.plot(x, np.nanmean(C, 0), color=GREY, lw=0.7, zorder=1)
        ax.axhline(5, color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.text(0.1, 5, "target 5 %", fontsize=7, color=INK, va="center", zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"Hours into campaign 2 (+{_gap_days()} days)")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Realised FPR per half hour (%)")
    axes[0].plot([], [], color=GREY, lw=0.8, label="other detectors")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), handlelength=2.2,
               columnspacing=1.6)
    save(fig, "fig_time_resolved_fpr")


# ---------------------------------------------------------------------------
def fig_ranking():
    S = json.load(open(os.path.join(RESULTS_PROC, "summary.json")))
    tags = [("unsw_chrono_A", "Protocol II\ncross-campaign"), ("unsw_hostdisjoint_A", "Protocol III\nhost-disjoint"),
            ("unsw_random_A", "Protocol I\nin-distribution"), ("cic_crossday", "Protocol IV\nCIC-IDS2017")]
    tags = [t for t in tags if t[0] in S]
    meths = [m for m in ["ARC-Field", "MemAE", "Dense Autoencoder", "KitNET", "MCM", "Deep SVDD", "Anomal-E", "DIF",
                         "Diffusion (DDPM) detector", "One-Class SVM", "Isolation Forest", "LSTM Autoencoder",
                         "DAGMM", "USAD", "Transformer Autoencoder", "Local Outlier Factor"]
             if any(m in S[t] for t, _ in tags)]
    fig, axes = plt.subplots(1, len(tags), figsize=(WIDE, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(meths))[::-1]
    for ax, (tag, title) in zip(axes, tags):
        for yi, m in zip(y, meths):
            d = S[tag].get(m, {}).get("static", {}).get("auprc")
            if not d:
                continue
            col, mk = (HL[m][0], HL[m][1]) if m in HL else (GREY, "o")
            ax.errorbar(100 * d["mean"], yi, xerr=100 * d["sd"], fmt=mk, color=col, ms=4, capsize=0,
                        elinewidth=0.8, zorder=3 if m in HL else 2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUPRC (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([SHORT.get(m, m) for m in meths])
    save(fig, "fig_ranking_auprc")


# ---------------------------------------------------------------------------
def fig_poisoning():
    f = os.path.join(RESULTS_RAW, "poisoning", "poisoning_results.json")
    if not os.path.exists(f):
        return
    P = json.load(open(f))["results"]
    polstyle = {"burnin_full": (ORANGE, "s", "mean/SD restandardisation"),
                "burnin_thr": (AQUA, "^", "plain quantile threshold"),
                "arc": (BLUE, "o", "ARC (host cap + median/MAD + trimming)")}
    panels = [("ARC-Field", "replay", "rate"), ("ARC-Field", "chaff", "rate"), ("ARC-Field", "blind", "n"),
              ("MemAE", "blind", "n")]
    fig, axes = plt.subplots(1, 4, figsize=(WIDE, 2.3), sharey=True)
    for pi, (ax, (meth, kind, xk)) in enumerate(zip(axes, panels)):
        for pol, (col, mk, lab) in polstyle.items():
            rows = [r for r in P if r["method"] == meth and r["policy"] == pol and r["spread"] == "single"
                    and (r["kind"] == kind or r["n_injected"] == 0)]
            if not rows:
                continue
            xs = sorted(set((r["n_injected"] if xk == "n" else r["rate"]) for r in rows))
            if xk == "n":
                xs = [x for x in xs if x == 0 or any(r["kind"] == kind and r["n_injected"] == x for r in rows)]
            ys, es = [], []
            for x in xs:
                v = [r["recall"] for r in rows if (r["n_injected"] if xk == "n" else r["rate"]) == x
                     and (r["kind"] == kind or x == 0)]
                ys.append(100 * np.mean(v))
                es.append(100 * np.std(v))
            xp = np.array(xs, float) * (1 if xk == "n" else 100)
            ax.errorbar(xp, ys, yerr=es, color=col, marker=mk, ms=3.5, capsize=0, elinewidth=0.7, label=lab)
        ax.set_xscale("symlog", linthresh=1 if xk == "n" else 0.2)
        ax.set_xlabel("Injected records" if xk == "n" else "Injected share (%)")
        ax.set_title(f"({'abcd'[pi]}) {meth}, {kind}", loc="left")
    axes[0].set_ylabel("Recall after recalibration (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6)
    save(fig, "fig_poisoning")


# ---------------------------------------------------------------------------
def fig_churn():
    f = os.path.join(RESULTS_RAW, "churn", "churn_results.json")
    if not os.path.exists(f):
        return
    C = json.load(open(f))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    for ax, (key, pol, lab) in zip(axes, (("fpr", "static", "Realised FPR, static threshold (%)"),
                                          ("auroc", "static", "AUROC (%)"))):
        for meth, (col, mk, name) in (("ARC-Field", (BLUE, "o", "role coordinates")), ("Spectral coordinates", (ORANGE, "s", "spectral coordinates"))):
            for mode, ls in (("renumber", "-"), ("lease", (0, (3, 2)))):
                rows = [r for r in C["results"] if r["method"] == meth and (r["mode"] == mode or r["rho"] == 0)]
                rhos = sorted(set(r["rho"] for r in rows))
                ys = [100 * np.mean([r["policies"][pol][key] for r in rows if r["rho"] == rho]) for rho in rhos]
                ax.plot(np.array(rhos) * 100, ys, color=col, marker=mk, ms=3.5, ls=ls,
                        label=f"{name}, {mode}")
        ax.set_xlabel("Endpoints given fresh identifiers (%)")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=6.3, loc="upper left")
    save(fig, "fig_churn")


# ---------------------------------------------------------------------------
def fig_ablation():
    out = {}
    for prot in ("chrono", "hostdisjoint"):
        for f in glob.glob(os.path.join(RESULTS_ABL, f"model_{prot}", "*.json")):
            r = json.load(open(f))
            out.setdefault(prot, {}).setdefault(r["variant"], []).append(r)
    if not out:
        return
    names = {"full": "ARC-Field (full)", "no_context": "- role coordinates", "src_context": "source roles only",
             "dst_context": "destination roles only", "unmasked": "- masking (reconstruct all)",
             "heteroscedastic": "+ heteroscedastic head", "relu": "SIREN -> ReLU MLP",
             "spectral_coord": "spectral (Laplacian) coords", "abs_time": "+ absolute-time encoding"}
    order = [k for k in names if any(k in v for v in out.values())]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, prot, title in zip(axes, ("chrono", "hostdisjoint"), ("Protocol II", "Protocol III")):
        if prot not in out:
            continue
        for yi, k in zip(y, order):
            rs = out[prot].get(k)
            if not rs:
                continue
            v = [100 * r["policies"]["static"]["auprc"] for r in rs]
            ax.barh(yi, np.mean(v), xerr=np.std(v), color=BLUE if k == "full" else GREY, height=0.62,
                    error_kw={"elinewidth": 0.7, "ecolor": INK2})
            ax.text(np.mean(v) + np.std(v) + 0.5, yi, f"{np.mean(v):.1f}", va="center", fontsize=6.3, color=INK2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUPRC (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[k] for k in order])
    save(fig, "fig_ablation_model")


# ---------------------------------------------------------------------------
def fig_sensitivity():
    f = os.path.join(RESULTS_ABL, "calibration_chrono.json")
    if not os.path.exists(f):
        return
    D = json.load(open(f))["sensitivity"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.4))
    for ax, par, xl in zip(axes, ("alpha", "trim_c", "burnin"),
                           ("target FPR alpha (%)", "trim constant c", "window size B")):
        for meth in ("ARC-Field", "MemAE", "Dense_Autoencoder", "KitNET"):
            key = meth.replace("_", " ")
            if meth not in D or par not in D[meth]:
                continue
            col, mk, lab = HL.get(key, (GREY, "o", key))
            xs = sorted(D[meth][par], key=float)
            yk = "fpr"
            ys = [100 * np.mean([r[yk] for r in D[meth][par][x]]) for x in xs]
            xv = [100 * float(x) if par == "alpha" else float(x) for x in xs]
            ax.plot(xv, ys, color=col, marker=mk, ms=3.5, label=lab)
        if par == "alpha":
            lim = [0, 10.5]
            ax.plot(lim, lim, color=INK, lw=0.7, ls=(0, (3, 2)))
            ax.text(6.2, 7.6, "realised = target", fontsize=6.3, color=INK2, rotation=30)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("Realised FPR (%)")
    axes[0].legend(fontsize=6.3)
    save(fig, "fig_calibration_sensitivity")


# ---------------------------------------------------------------------------
def fig_families():
    R = runs("unsw_chrono_A")
    meths = [m for m in ("ARC-Field", "MemAE", "Dense Autoencoder", "Deep SVDD", "Anomal-E", "Isolation Forest")
             if m in R]
    if not meths:
        return
    fams = sorted({f for m in meths for r in R[m] for f in r["policies"]["arc"]["per_family"]})
    M = np.array([[100 * np.mean([r["policies"]["arc"]["per_family"].get(f, {}).get("tpr", np.nan)
                                  for r in R[m]]) for f in fams] for m in meths])
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f2f6fc", "#9cc1ee", "#2a78d6", "#123a6b"])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.3,
                    color="white" if M[i, j] > 60 else INK)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=30, ha="right")
    ax.set_yticks(range(len(meths)))
    ax.set_yticklabels([SHORT.get(m, m) for m in meths])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Detection rate (%)", fontsize=7)
    save(fig, "fig_per_family")


# ---------------------------------------------------------------------------
def fig_cd(src="friedman_nemenyi.json", out="fig_critical_difference"):
    """Average ranks (Friedman-Nemenyi) as one ranked dot row per detector,
    so no label can collide; the shaded band spans one critical difference
    from the best average rank (detectors inside it are not significantly
    worse than the best). Main text: Protocols II and V; supplement
    (friedman_nemenyi_all.json): Protocols I-III and V."""
    f = os.path.join(RESULTS_STATS, src)
    if not os.path.exists(f):
        return
    F = json.load(open(f))
    metrics = [m for m in ("auprc", "mcc_arc") if m in F]
    fig, axes = plt.subplots(1, len(metrics), figsize=(WIDE, 3.1))
    axes = np.atleast_1d(axes)
    for pi, (ax, metric) in enumerate(zip(axes, metrics)):
        d = F[metric]
        items = sorted(d["avg_rank"].items(), key=lambda kv: kv[1])
        k, cd = d["k"], d["critical_difference"]
        best = items[0][1]
        y = np.arange(len(items))[::-1]
        ax.axvspan(best, best + cd, color="#e8f0fb", zorder=0, lw=0)
        ax.axvline(best + cd, color=BLUE, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for yi, (m, r) in zip(y, items):
            col = HL[m][0] if m in HL else INK2
            ax.plot([1, r], [yi, yi], color="#e1e0dc", lw=0.8, zorder=1)
            ax.plot(r, yi, "o", color=col, ms=4.2, zorder=3)
            ax.text(r + 0.25, yi, f"{r:.1f}", va="center", fontsize=6.8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(m, m) for m, _ in items])
        for t, (m, _) in zip(ax.get_yticklabels(), items):
            if m == "ARC-Field":
                t.set_fontweight("bold")
        ax.set_xlim(1, k + 0.8)
        ax.set_ylim(-0.7, len(items) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Average rank (lower is better)")
        p = d["p"]
        ptxt = f"{p:.1e}".replace("e-", r"\times10^{-") + "}" if p < 1e-3 else f"{p:.3f}"
        ax.set_title(f"({'ab'[pi]}) " + {"auprc": "AUPRC", "mcc_arc": "MCC after ARC recalibration"}[metric]
                     + f"; Friedman $p={ptxt}$, CD = {cd:.2f}", loc="left")
    fig.tight_layout(w_pad=2.0)
    save(fig, out)


def fig_cd_all():
    fig_cd("friedman_nemenyi_all.json", "fig_critical_difference_all")


if __name__ == "__main__":
    for fn in (fig_time_resolved, fig_ranking, fig_poisoning, fig_churn, fig_ablation, fig_sensitivity,
               fig_families, fig_cd, fig_cd_all):
        try:
            fn()
        except Exception as e:  # keep going; report
            print("FAILED", fn.__name__, repr(e))
