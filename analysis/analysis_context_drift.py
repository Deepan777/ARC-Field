"""How much does each endpoint-context encoding drift between the fitting
period and deployment, compared with the flow features themselves?

For every protocol of the context-baseline study (seed 0 partitions), benign
fitting rows are compared with benign test rows, after the robust scaling
fitted on the fitting rows (the scaling every detector receives):
  domain AUC   cross-validated AUROC of a classifier that tells fitting rows
               from test rows (0.5: no drift; 1.0: fully separable)
  median shift median over columns of |median(test) - median(fit)| in
               robust-scaled units
The study's question is whether the encodings that hurt detectors when they
are appended as input columns are the ones that drift most.
Output: results/processed/context_drift.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import DATA_DIR, PROJECT, RESULTS_PROC  # noqa: E402
from scaling import RobustScaler, log_mask_for  # noqa: E402

N = 20000
CACHE = os.path.join(PROJECT, "results", "context_baselines", "cache")


def domain_auc(A, B, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, y, cv=5, scoring="roc_auc")))


def summarise(name, fit, test, log_mask=None):
    sc = RobustScaler(log_mask).fit(fit)
    a, b = sc(fit), sc(test)
    shift = float(np.median(np.abs(np.median(b, 0) - np.median(a, 0))))
    out = {"domain_auc": domain_auc(a, b), "median_shift": shift, "dim": int(fit.shape[1])}
    print(f"  {name:11s} dim {out['dim']:4d}  domain AUC {out['domain_auc']:.3f}  median shift {shift:.3f}", flush=True)
    return out


def main(protocols=("chrono", "hostdisjoint", "nf")):
    rng = np.random.default_rng(0)
    alt = os.path.join(DATA_DIR, "ctxalt")
    out_path = os.path.join(RESULTS_PROC, "context_drift.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for prot in protocols:
        if prot == "nf":
            import data_nf as dn
            corpus = dn.load_corpus(verbose=False)
            tr, va, ca_, tests = dn.cross_day(corpus, 0)
            te = np.concatenate([ix for ix in tests.values()])
            feats, heavy, X = dn.FEATURE_SETS["A"], dn.HEAVY, corpus["X"][:, [dn.ALL.index(f) for f in dn.FEATURE_SETS["A"]]]
            tag = "nf"
        else:
            import data_unsw as du
            corpus = du.load_corpus(verbose=False)
            P = {"chrono": du.chronological, "hostdisjoint": du.host_disjoint}[prot](corpus, seed=0)
            tr, te = P.train, P.test
            feats, heavy = du.FEATURE_SETS["A"], du.HEAVY
            X = corpus["X"][:, [du.ALL36.index(f) for f in feats]]
            tag = "unsw"
        te_b = te[corpus["is_normal"][te]]
        a_rows = np.sort(rng.choice(tr, min(N, len(tr)), replace=False))
        b_pos = np.sort(rng.choice(len(te_b), min(N, len(te_b)), replace=False))
        b_rows = te_b[b_pos]
        A_all = np.load(os.path.join(alt, f"{tag}_afterimage.npy"), mmap_mode="r")
        names = json.load(open(os.path.join(alt, f"{tag}_ctxalt_names.json")))["afterimage"]
        print(f"== {prot}: {len(a_rows):,} fitting vs {len(b_rows):,} benign test rows", flush=True)
        r = {"flow features": summarise("flow", X[a_rows], X[b_rows], log_mask_for(feats, heavy)),
             "roles": summarise("roles", corpus["C"][a_rows], corpus["C"][b_rows]),
             "afterimage": summarise("afterimage", np.asarray(A_all[a_rows]), np.asarray(A_all[b_rows]),
                                     np.array(["_w_" in n for n in names]))}
        rc = os.path.join(CACHE, f"{prot}_s0_rolx.npz")
        if os.path.exists(rc):
            d = np.load(rc)
            test_keys = ["test"] if prot != "nf" else [k for k in d.files if k.startswith("day")]
            test_idx = te if prot != "nf" else np.concatenate([tests[int(k[3:])] for k in test_keys])
            Rt = np.vstack([d[k] for k in test_keys])
            Rtr = d["train"]
            pos_tr = np.searchsorted(tr, a_rows)
            ben = corpus["is_normal"][test_idx]
            Rt_b = Rt[ben]
            r["rolx"] = summarise("rolx", Rtr[pos_tr], Rt_b[b_pos])
        res[prot] = r
        os.makedirs(RESULTS_PROC, exist_ok=True)
        json.dump(res, open(out_path, "w"), indent=1)
    print("written", out_path)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("chrono", "hostdisjoint", "nf"))
