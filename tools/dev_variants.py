"""Development-protocol study of the field's scoring rule and inputs.

Variants (all on Protocol D, seeds 0-1):
  z2        heteroscedastic field, score from z^2 (initial design)
  nll       heteroscedastic field, score from the conditional NLL z^2 + log s^2
  homo      homoscedastic field (Huber), squared error
  noctx     nll without context
  unmasked  nll-type field that sees every feature (identity-prone reference)
  DenseAE   reference baseline on the same inputs
Reports AUROC/AUPRC with the static robust scorer (per-feature median/MAD on
validation). Output: logs/dev/variant_selection.json
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "third_party"))
from paths import LOG_DIR  # noqa: E402
import arcfield as af  # noqa: E402
import calibration as cal  # noqa: E402
import data_unsw as du  # noqa: E402
import metrics as mt  # noqa: E402
from scaling import RobustScaler, log_mask_for  # noqa: E402
import baselines as B0  # noqa: E402

c = du.load_corpus(verbose=False)
feats = du.FEATURE_SETS["A"]
fi = [du.ALL36.index(f) for f in feats]
groups = du.groups_for(feats)


@torch.no_grad()
def nll_residuals(model, M, C, masked=True, bs=8192):
    """z^2 + log sigma^2 per feature (conditional negative log-likelihood up to
    constants), from the pass in which each feature is hidden."""
    model.eval()
    d = M.shape[1]
    G = len(groups)
    gm = torch.zeros(G, d, device="cuda")
    for i, g in enumerate(groups):
        gm[i, g] = 1.0
    out = []
    for s in range(0, len(M), bs):
        m = torch.from_numpy(M[s:s + bs]).cuda()
        cc = torch.from_numpy(C[s:s + bs]).cuda() if C.shape[1] else torch.zeros(len(m), 0, device="cuda")
        if masked:
            B = len(m)
            mask = gm.repeat_interleave(B, 0)
            mu, lv = model(cc.repeat(G, 1), m.repeat(G, 1), mask)
            u = ((m.repeat(G, 1) - mu) ** 2 / torch.exp(lv) + lv) * mask
            u = u.view(G, B, d).sum(0)
        else:
            mu, lv = model(cc, m, torch.zeros_like(m))
            u = (m - mu) ** 2 / torch.exp(lv) + lv
        out.append(u.cpu().numpy())
    U = np.concatenate(out)
    return U - U.min() + 1e-3     # shift to positive; ranking unchanged per feature


def score_static(Rv, Rc, Rt, y):
    o = cal.run_policy("static", cal.CalConfig(), R_val=Rv, R_cal=Rc, R_test=Rt)
    return mt.ranking_metrics(o["score"], y[o["eval_index"]])


rows = {}
for seed in (0, 1):
    P = du.development(c, seed=seed)
    Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(c["X"][P.train][:, fi])
    Cs = RobustScaler(None).fit(c["C"][P.train])
    M = {k: Xs(c["X"][getattr(P, k)][:, fi]) for k in ("train", "val", "cal", "test")}
    C = {k: Cs(c["C"][getattr(P, k)]) for k in ("train", "val", "cal", "test")}
    Z = {k: np.zeros((len(M[k]), 0), np.float32) for k in M}
    y = ~c["is_normal"][P.test]

    def fit(hetero=True, masked=True, ctx=True):
        m = af.ARCField(len(feats), c["C"].shape[1] if ctx else 0, hidden=128, layers=4, hetero=hetero)
        af.train_field(m, M["train"], C["train"] if ctx else Z["train"], groups, epochs=20,
                       seed=seed, masked=masked)
        return m

    m = fit()
    R = {k: af.residuals(m, M[k], C[k], groups) for k in ("val", "cal", "test")}
    rows.setdefault("z2", []).append(score_static(R["val"], R["cal"], R["test"], y))
    R = {k: nll_residuals(m, M[k], C[k]) for k in ("val", "cal", "test")}
    rows.setdefault("nll", []).append(score_static(R["val"], R["cal"], R["test"], y))
    m = fit(hetero=False)
    R = {k: af.residuals(m, M[k], C[k], groups) for k in ("val", "cal", "test")}
    rows.setdefault("homo", []).append(score_static(R["val"], R["cal"], R["test"], y))
    m = fit(ctx=False)
    R = {k: nll_residuals(m, M[k], Z[k]) for k in ("val", "cal", "test")}
    rows.setdefault("noctx_nll", []).append(score_static(R["val"], R["cal"], R["test"], y))
    m = fit(masked=False)
    R = {k: nll_residuals(m, M[k], C[k], masked=False) for k in ("val", "cal", "test")}
    rows.setdefault("unmasked_nll", []).append(score_static(R["val"], R["cal"], R["test"], y))
    U = {k: ((M[k] / 6 + 1) / 2).astype(np.float32) for k in M}
    sf = B0.dense_ae(U["train"], seed)
    s = {k: sf(U[k]) for k in ("val", "cal", "test")}
    o = cal.run_policy("static", cal.CalConfig(), s_val=s["val"], s_cal=s["cal"], s_test=s["test"])
    rows.setdefault("DenseAE", []).append(mt.ranking_metrics(o["score"], y[o["eval_index"]]))
    print(seed, {k: (round(v[-1]["auroc"], 4), round(v[-1]["auprc"], 4)) for k, v in rows.items()}, flush=True)

summary = {k: {"auroc": float(np.mean([r["auroc"] for r in v])),
               "auprc": float(np.mean([r["auprc"] for r in v]))} for k, v in rows.items()}
json.dump({"summary": summary, "per_seed": rows}, open(os.path.join(LOG_DIR, "dev", "variant_selection.json"), "w"),
          indent=1, default=float)
print(json.dumps(summary, indent=1))
