"""Protocol D follow-up: homoscedastic field with/without context and masking,
and width. Output logs/dev/variant_selection2.json"""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import LOG_DIR
import arcfield as af, calibration as cal, data_unsw as du, metrics as mt
from scaling import RobustScaler, log_mask_for
c = du.load_corpus(verbose=False)
feats = du.FEATURE_SETS["A"]; fi = [du.ALL36.index(f) for f in feats]; groups = du.groups_for(feats)
rows = {}
for seed in (0, 1):
    P = du.development(c, seed=seed)
    Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(c["X"][P.train][:, fi])
    Cs = RobustScaler(None).fit(c["C"][P.train])
    M = {k: Xs(c["X"][getattr(P, k)][:, fi]) for k in ("train", "val", "cal", "test")}
    C = {k: Cs(c["C"][getattr(P, k)]) for k in ("train", "val", "cal", "test")}
    Z = {k: np.zeros((len(M[k]), 0), np.float32) for k in M}
    y = ~c["is_normal"][P.test]
    for name, kw in {"homo_ctx": dict(ctx=True), "homo_noctx": dict(ctx=False),
                     "homo_unmasked": dict(masked=False), "homo_ctx_h256": dict(hidden=256),
                     "homo_ctx_e40": dict(epochs=40)}.items():
        ctx = kw.get("ctx", True); masked = kw.get("masked", True)
        m = af.ARCField(len(feats), c["C"].shape[1] if ctx else 0, hidden=kw.get("hidden", 128),
                        layers=4, hetero=False)
        af.train_field(m, M["train"], C["train"] if ctx else Z["train"], groups,
                       epochs=kw.get("epochs", 20), seed=seed, masked=masked)
        CC = C if ctx else Z
        R = {k: af.residuals(m, M[k], CC[k], groups, masked=masked) for k in ("val", "cal", "test")}
        o = cal.run_policy("static", cal.CalConfig(), R_val=R["val"], R_cal=R["cal"], R_test=R["test"])
        rows.setdefault(name, []).append(mt.ranking_metrics(o["score"], y[o["eval_index"]]))
    print(seed, {k: (round(v[-1]["auroc"], 4), round(v[-1]["auprc"], 4)) for k, v in rows.items()}, flush=True)
summary = {k: {"auroc": float(np.mean([r["auroc"] for r in v])), "auprc": float(np.mean([r["auprc"] for r in v]))}
           for k, v in rows.items()}
json.dump({"summary": summary, "per_seed": rows}, open(os.path.join(LOG_DIR, "dev", "variant_selection2.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1))
