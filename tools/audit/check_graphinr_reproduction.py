"""Re-run GraphINR-IDS on one UNSW-NB15 protocol/seed with the code path of
run_unsw.py and compare with the stored result file. Writes nothing under
results/; prints stored vs recomputed metrics (GPU non-determinism allows small
differences).
usage: python code/tools/audit/check_graphinr_reproduction.py [random|chrono|hostdisjoint] [seed]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))
from common import MATRIX_POLICIES, RESULTS_RAW, device, evaluate_policies  # noqa: E402
import data_unsw as du  # noqa: E402
import graphinr_baseline as gb  # noqa: E402

prot = sys.argv[1] if len(sys.argv) > 1 else "random"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
PROT = {"random": du.random_split, "chrono": du.chronological, "hostdisjoint": du.host_disjoint}
corpus = du.load_corpus(verbose=False)
feats = du.FEATURE_SETS["A"]
fi = [du.ALL36.index(f) for f in feats]
kappa = max(1, round(len(feats) / 4))
P = PROT[prot](corpus, seed=seed)
ctx = {"y": ~corpus["is_normal"][P.test], "labels": corpus["label"][P.test], "host": corpus["src"][P.test],
       "t": corpus["ltime"][P.test].astype(float), "time_resolved": prot == "chrono"}
g = gb.run(corpus, P, fi, seed, device=device())
pol = evaluate_policies("matrix", {k: g[k] for k in ("val", "cal", "test")}, MATRIX_POLICIES, ctx,
                        {"kappa": kappa, "scoring": "raw_meansd"})
stored = json.load(open(os.path.join(RESULTS_RAW, f"unsw_{prot}_A", f"GraphINR-IDS_s{seed}.json")))
print(f"GraphINR-IDS unsw_{prot}_A seed {seed}: graph nodes {g['graph_nodes']} (stored {stored['graph_nodes']}), "
      f"OOV test flows {g['oov_test_flows']} (stored {stored['oov_test_flows']})")
for p in ("static", "burnin_full", "arc"):
    for k in ("auroc", "auprc", "fpr", "recall"):
        print(f"  {p:12s} {k:7s} stored {100 * stored['policies'][p][k]:7.2f}  recomputed {100 * pol[p][k]:7.2f}")
