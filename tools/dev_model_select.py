"""Model hyper-parameter selection on development Protocol D (AUPRC under the
static robust scorer, mean of 2 seeds). Output logs/dev/model_selection.json"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import LOG_DIR
import arcfield as af, data_unsw as du, calibration as cal, metrics as mt
from scaling import RobustScaler, log_mask_for
c = du.load_corpus(verbose=False)
feats = du.FEATURE_SETS["A"]; fi = [du.ALL36.index(f) for f in feats]; groups = du.groups_for(feats)
res = []
for epochs, hidden, layers in [(10, 256, 4), (20, 256, 4), (40, 256, 4), (20, 128, 4), (20, 256, 6)]:
    sc = []
    for seed in (0, 1):
        P = du.development(c, seed=seed)
        Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(c["X"][P.train][:, fi])
        Cs = RobustScaler(None).fit(c["C"][P.train])
        M = lambda idx: Xs(c["X"][idx][:, fi]); C = lambda idx: Cs(c["C"][idx])
        m = af.ARCField(len(feats), c["C"].shape[1], hidden=hidden, layers=layers)
        af.train_field(m, M(P.train), C(P.train), groups, epochs=epochs, seed=seed)
        R = {k: af.residuals(m, M(getattr(P, k)), C(getattr(P, k)), groups) for k in ("val", "cal", "test")}
        o = cal.run_policy("static", cal.CalConfig(), R_val=R["val"], R_cal=R["cal"], R_test=R["test"])
        y = ~c["is_normal"][P.test][o["eval_index"]]
        sc.append(mt.ranking_metrics(o["score"], y))
    r = {"epochs": epochs, "hidden": hidden, "layers": layers,
         "auroc": float(np.mean([s["auroc"] for s in sc])), "auprc": float(np.mean([s["auprc"] for s in sc]))}
    res.append(r); print(r, flush=True)
best = max(res, key=lambda r: r["auprc"])
json.dump({"rule": "max mean AUPRC on Protocol D (2 seeds)", "selected": best, "all": res},
          open(os.path.join(LOG_DIR, "dev", "model_selection.json"), "w"), indent=1)
print("SELECTED", best)
