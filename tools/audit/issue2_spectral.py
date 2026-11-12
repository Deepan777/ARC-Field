"""Issue 2: what robust (median/IQR, 6*tanh) scaling does to the spectral
coordinate, in particular to the zero vector of unseen endpoints. Read-only."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "third_party"))
import data_unsw as du  # noqa: E402
import graphinr_ids_harness as rc  # noqa: E402
from scaling import RobustScaler  # noqa: E402

c = du.load_corpus(verbose=False)
ips = c["ips"]
for prot, fn, seeds in (("III", du.host_disjoint, (0, 1, 2, 3, 4)), ("II", du.chronological, (0, 1, 2, 3, 4))):
    for seed in seeds:
        P = fn(c, seed)
        g = rc.RevGraph(spectral_k=32).fit(ips[c["src"][P.train]], ips[c["dst"][P.train]], verbose=False)

        def pe(idx):
            ps = g.eigenvectors[g.index(ips[c["src"][idx]])]
            pd_ = g.eigenvectors[g.index(ips[c["dst"][idx]])]
            return np.concatenate([ps, pd_, ps * pd_], 1).astype(np.float32)
        tr, te = pe(P.train), pe(P.test)
        sc = RobustScaler(None).fit(tr)
        z_tr, z_te = sc(tr), sc(te)
        unseen = np.all(te[:, :32] == 0, 1) | np.all(te[:, 32:64] == 0, 1)
        y = ~c["is_normal"][P.test]
        print(f"Protocol {prot} seed {seed}: test records with an unseen endpoint (zero vector): "
              f"{unseen.mean()*100:.1f}% (benign {unseen[~y].mean()*100:.1f}%, attack {unseen[y].mean()*100:.1f}%)")
        print(f"   raw  |x| max: train {np.abs(tr).max():.3f}, test {np.abs(te).max():.3f}")
        print(f"   scaled |z| 99th pct: train {np.percentile(np.abs(z_tr),99):.2f}, "
              f"test unseen rows {np.percentile(np.abs(z_te[unseen]),99) if unseen.any() else float('nan'):.2f}, "
              f"test seen rows {np.percentile(np.abs(z_te[~unseen]),99):.2f}")
        frac_sat = (np.abs(z_te[unseen]) > 5).mean() if unseen.any() else float("nan")
        print(f"   share of scaled entries of unseen rows beyond |5| (tanh bound 6): {frac_sat*100:.1f}%; "
              f"same for training rows: {(np.abs(z_tr) > 5).mean()*100:.2f}%")
