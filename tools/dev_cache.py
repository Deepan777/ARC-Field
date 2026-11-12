"""Fit ARC-Field on the development protocol and cache residuals so that
calibrator hyper-parameters can be selected without touching test streams."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import LOG_DIR
import arcfield as af, data_unsw as du
from scaling import RobustScaler, log_mask_for
c = du.load_corpus(verbose=False)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
P = du.development(c, seed=seed)
feats = du.FEATURE_SETS["A"]; fi = [du.ALL36.index(f) for f in feats]; groups = du.groups_for(feats)
Xs = RobustScaler(log_mask_for(feats, du.HEAVY)).fit(c["X"][P.train][:, fi])
Cs = RobustScaler(None).fit(c["C"][P.train])
M = lambda idx: Xs(c["X"][idx][:, fi]); C = lambda idx: Cs(c["C"][idx])
model = af.ARCField(len(feats), c["C"].shape[1], hidden=256, layers=4, hetero=False)
af.train_field(model, M(P.train), C(P.train), groups, epochs=20, seed=seed)
R = {k: af.residuals(model, M(getattr(P, k)), C(getattr(P, k)), groups) for k in ("val", "cal", "test")}
y = ~c["is_normal"][P.test]
print("dev test n", len(P.test), "attack frac", y.mean())
os.makedirs(os.path.join(LOG_DIR, "dev"), exist_ok=True)
np.savez_compressed(os.path.join(LOG_DIR, "dev", f"dev_resid_s{seed}.npz"), **R, y=y,
                    host=c["src"][P.test], t=c["ltime"][P.test], label=c["label"][P.test])
