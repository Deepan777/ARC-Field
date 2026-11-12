"""Issue 1: overlap between the Protocol D stream (and its fitting partitions)
and the Protocol I / III test streams, per seed, by record index. Read-only."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import data_unsw as du  # noqa: E402
from paths import RESULTS_PROC  # noqa: E402

B = 5000
c = du.load_corpus(verbose=False)
normal = c["is_normal"]
lt = c["ltime"]
cut = du.session_cut(c)
s1 = np.where(c["stime"] <= cut)[0]
t1 = lt[s1]
split = t1.min() + 2 * 3600
att1 = s1[~normal[s1]]
print(f"campaign 1: {len(s1):,} records; attacks {len(att1):,}; "
      f"attacks with ltime < split: {(lt[att1] < split).sum():,}; "
      f"attack ltime range [{lt[att1].min()-t1.min()}, {lt[att1].max()-t1.min()}] s after start")

D = du.development(c, seed=0)
dev_stream = D.test
dev_fit = np.concatenate([D.train, D.val, D.cal])
print(f"Protocol D stream: {len(dev_stream):,} records "
      f"(benign {normal[dev_stream].sum():,}, attack {(~normal[dev_stream]).sum():,})")
for s in (0, 1):
    Ds = du.development(c, seed=s)
    print(f"  D seed {s}: stream identical to seed 0: {np.array_equal(Ds.test, dev_stream)}; "
          f"fit/val/cal sizes {len(Ds.train)}/{len(Ds.val)}/{len(Ds.cal)}")

rows = []


def report(name, P):
    test = P.test
    ev = test[B:]                       # evaluated records (window excluded)
    for label, part in (("stream", test), ("evaluated", ev)):
        sh = np.intersect1d(part, dev_stream, assume_unique=False)
        rows.append({
            "protocol": name, "seed": P.meta["seed"], "part": label, "n": int(len(part)),
            "n_benign": int(normal[part].sum()), "n_attack": int((~normal[part]).sum()),
            "shared_with_D_stream": int(len(sh)),
            "shared_benign": int(normal[sh].sum()), "shared_attack": int((~normal[sh]).sum()),
            "share_of_attacks_shared": float((~normal[sh]).sum() / max(1, (~normal[part]).sum())),
            "shared_with_D_fit_partitions": int(len(np.intersect1d(part, dev_fit))),
        })


for s in (0, 1, 2):
    report("I (random)", du.random_split(c, s))
for s in (0, 1, 2, 3, 4):
    report("III (host-disjoint)", du.host_disjoint(c, s))

hdr = f"{'protocol':20s} {'seed':>4s} {'part':10s} {'records':>8s} {'benign':>8s} {'attack':>7s} | {'shared':>7s} {'sh.ben':>7s} {'sh.att':>7s} {'att.share':>9s} | {'in D fit':>8s}"
print(hdr)
for r in rows:
    print(f"{r['protocol']:20s} {r['seed']:4d} {r['part']:10s} {r['n']:8,d} {r['n_benign']:8,d} {r['n_attack']:7,d} | "
          f"{r['shared_with_D_stream']:7,d} {r['shared_benign']:7,d} {r['shared_attack']:7,d} {100*r['share_of_attacks_shared']:8.1f}% | "
          f"{r['shared_with_D_fit_partitions']:8,d}")
def halfhour_shares(idx, min_n=1000):
    """Attack share per half hour (bins on emission time from the first record),
    over bins with at least `min_n` records."""
    t = lt[idx] - lt[idx].min()
    b = (t // 1800).astype(int)
    out = []
    for k in np.unique(b):
        sel = idx[b == k]
        if len(sel) >= min_n:
            out.append(float((~normal[sel]).mean()))
    return out


s2 = np.where(c["stime"] > cut)[0]
hh = {"dev_stream": halfhour_shares(dev_stream), "campaign2": halfhour_shares(s2)}
for k, v in hh.items():
    print(f"half-hour attack share, {k}: {100 * min(v):.1f}-{100 * max(v):.1f} % over {len(v)} bins")
out = os.path.join(RESULTS_PROC, "dev_overlap.json")
summary = {"dev_stream": {"n": int(len(dev_stream)), "n_benign": int(normal[dev_stream].sum()),
                          "n_attack": int((~normal[dev_stream]).sum())},
           "campaign1_attacks": int(len(att1)),
           "campaign1_attacks_in_dev_stream": int((lt[att1] < split).sum()),
           "halfhour_attack_share": {k: {"min": min(v), "max": max(v), "n_bins": len(v)} for k, v in hh.items()},
           "rows": rows}
json.dump(summary, open(out, "w"), indent=1)
print("saved", out)
