"""Compute the alternative endpoint-context encodings (AfterImage, ReFeX) for
the UNSW-NB15 and NF-CICIDS2018-v3 corpora, aligned row for row with the
cached ARC-Field corpora.

usage: python build_ctxalt.py [--corpus unsw nf]

Outputs (data_processed/ctxalt/, memory-mappable .npy):
  <corpus>_afterimage.npy   float32 (n, 100)
  <corpus>_refex.npy        float16 (n, 110)
  <corpus>_ctxalt_names.json
NF-CICIDS2018-v3: the encodings are computed causally over the complete,
unthinned stream of each capture day (exactly as the role coordinates) and
only the corpus rows are kept. Alignment is verified by recomputing the role
coordinates of every day and comparing them with the cached corpus.
Each output is written to a temporary file and renamed when complete, so an
interrupted run resumes by recomputing only what is missing.
"""
import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import DATA_DIR  # noqa: E402
import context as ctx  # noqa: E402
import context_alt as ca  # noqa: E402

OUT = os.path.join(DATA_DIR, "ctxalt")
os.makedirs(OUT, exist_ok=True)


def target(name):
    return os.path.join(OUT, name)


def memmap(name, shape, dtype):
    return np.lib.format.open_memmap(target(name) + ".tmp.npy", mode="w+", dtype=dtype, shape=shape)


def finish(name, mm):
    mm.flush()
    mm._mmap.close()            # Windows keeps a mapped file locked until it is closed
    os.replace(target(name) + ".tmp.npy", target(name))


def build_unsw():
    import data_unsw as du
    c = du.load_corpus(verbose=True)
    n = len(c["label"])
    t = c["ltime"].astype(np.float64)
    sb = c["X"][:, du.ALL36.index("sbytes")].astype(np.float64)
    db = c["X"][:, du.ALL36.index("dbytes")].astype(np.float64)
    names = {}
    if not os.path.exists(target("unsw_afterimage.npy")):
        t0 = time.time()
        mm = memmap("unsw_afterimage.npy", (n, 100), np.float32)
        _, names["afterimage"] = ca.afterimage(c["src"], c["dst"], c["dport"], t, sb, db, out=mm)
        finish("unsw_afterimage.npy", mm)
        print(f"[unsw] AfterImage {n:,} rows in {time.time() - t0:.0f}s", flush=True)
    if not os.path.exists(target("unsw_refex.npy")):
        t0 = time.time()
        mm = memmap("unsw_refex.npy", (n, 2 * ca.N_REFEX + 2), np.float16)
        ca.refex_context(c["src"], c["dst"], t, out=mm, verbose=True)
        finish("unsw_refex.npy", mm)
        print(f"[unsw] ReFeX {n:,} rows in {time.time() - t0:.0f}s", flush=True)
    write_names("unsw", names)


def write_names(corpus, names):
    if "afterimage" not in names:
        z = np.zeros(4, np.int64)
        _, names["afterimage"] = ca.afterimage(z, z, z, np.arange(4.0), z + 1.0, z + 1.0)
    names["refex"] = ([f"src_refex_{i}" for i in range(ca.N_REFEX)] + [f"dst_refex_{i}" for i in range(ca.N_REFEX)]
                      + ["src_present", "dst_present"])
    with open(target(f"{corpus}_ctxalt_names.json"), "w") as f:
        json.dump(names, f, indent=0)


# ---------------------------------------------------------------------------
# NF-CICIDS2018-v3
# ---------------------------------------------------------------------------
STREAM = ("t", "src", "dst", "dport", "proto", "ib", "ob", "dur", "lab")


RAW = {"t0": ("FLOW_START_MILLISECONDS", np.int64), "t1": ("FLOW_END_MILLISECONDS", np.int64),
       "src": ("IPV4_SRC_ADDR", np.int32), "dst": ("IPV4_DST_ADDR", np.int32), "dport": ("L4_DST_PORT", np.int32),
       "proto": ("PROTOCOL", np.int16), "ib": ("IN_BYTES", np.float64), "ob": ("OUT_BYTES", np.float64),
       "dur": ("FLOW_DURATION_MILLISECONDS", np.float64), "lab": ("Label", np.int8)}


def nf_stream(chunksize=400_000):
    """Full emission-ordered stream (memory-mapped), built once from the zip.

    Same content and order as data_nf._pass1 + build_corpus (addresses coded
    in order of first appearance, emission order (t1, t0, file position)),
    but streamed in small chunks straight to disk: the machine this runs on
    cannot hold the 20 M-row pass in memory while other work is running."""
    import json
    import pandas as pd
    import data_nf as dn
    paths = {k: target(f"nf_stream_{k}.npy") for k in STREAM}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    n = json.load(open(os.path.join(DATA_DIR, "nf_cicids2018_profile.json")))["n_flows"]
    raw = {k: np.lib.format.open_memmap(target(f"nf_raw_{k}.npy"), mode="w+", dtype=dt, shape=(n,))
           for k, (_, dt) in RAW.items()}
    ip_index = {}
    z = zipfile.ZipFile(dn.NF_ZIP)
    a = 0
    with z.open(dn._member(z)) as f:
        for ch in pd.read_csv(f, usecols=[c for c, _ in RAW.values()], chunksize=chunksize, low_memory=False):
            for col in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
                for u in pd.unique(ch[col]):
                    if u not in ip_index:
                        ip_index[u] = len(ip_index)
            b = a + len(ch)
            for k, (c, dt) in RAW.items():
                v = ch[c].map(ip_index) if k in ("src", "dst") else ch[c]
                if k == "dur":
                    raw[k][a:b] = v.to_numpy(np.float64) / 1000.0
                else:
                    raw[k][a:b] = v.to_numpy(dt)
            a = b
            if (a // chunksize) % 10 == 0:
                print(f"[nf] stream pass: {a:,} rows", flush=True)
    assert a == n, (a, n)
    np.save(target("nf_stream_ips.npy"), np.array(sorted(ip_index, key=ip_index.get)))
    order = np.lexsort((np.arange(n), np.asarray(raw["t0"]), np.asarray(raw["t1"])))
    cols = {"t": lambda: np.asarray(raw["t1"])[order].astype(np.float64) / 1000.0,
            "proto": lambda: np.where(np.asarray(raw["proto"])[order] == 6, 0,
                                      np.where(np.asarray(raw["proto"])[order] == 17, 1, 2)).astype(np.int8)}
    for k in STREAM:
        v = cols[k]() if k in cols else np.asarray(raw[k])[order]
        np.save(paths[k] + ".tmp.npy", v)
        os.replace(paths[k] + ".tmp.npy", paths[k])
        del v
    for k in list(raw):
        raw[k]._mmap.close()
        os.remove(target(f"nf_raw_{k}.npy"))
    del order
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def nf_selection(s):
    """Rows of the cached corpus, in emission order (identical to
    data_nf.build_corpus)."""
    import data_nf as dn
    t, lab = s["t"], np.asarray(s["lab"])
    n = len(t)
    cuts = np.where(np.diff(t) > dn.DAY_GAP_S)[0] + 1
    bounds = np.concatenate([[0], cuts, [n]])
    day = np.zeros(n, np.int16)
    for i in range(len(bounds) - 1):
        day[bounds[i]:bounds[i + 1]] = i
    is_normal = lab == 0
    rng = np.random.default_rng(20180214)
    ben0 = np.where((day == 0) & is_normal)[0]
    cal = ben0[-dn.N_CAL:]
    pool = ben0[:-dn.N_CAL]
    pool = np.sort(rng.choice(pool, size=min(dn.N_POOL, len(pool)), replace=False))
    role = np.full(n, -1, np.int8)
    role[pool] = 0
    role[cal] = 1
    rank = np.zeros(n, np.int64)
    for i in range(1, len(bounds) - 1):
        idx = np.arange(bounds[i], bounds[i + 1])
        stride = max(1, len(idx) // dn.PER_DAY_TEST)
        rank[idx] = np.arange(len(idx))
        role[idx[(rank[idx] % stride) < dn.N_SEEDS]] = 2
    return np.where(role >= 0)[0], bounds


def build_nf():
    import data_nf as dn
    corpus = dn.load_corpus(verbose=True)
    s = nf_stream()
    sel, bounds = nf_selection(s)
    assert len(sel) == len(corpus["X"]), (len(sel), len(corpus["X"]))
    ips = np.load(target("nf_stream_ips.npy"))
    for side in ("src", "dst"):          # address codes may differ; the addresses must not
        assert np.array_equal(ips[np.asarray(s[side])[sel]], corpus["ips"][corpus[side]]), f"row alignment ({side})"
    assert np.array_equal(np.asarray(s["lab"])[sel] == 0, corpus["is_normal"]), "row alignment (label)"
    print(f"[nf] stream {len(s['t']):,} flows, {len(bounds) - 1} days, {len(sel):,} corpus rows", flush=True)
    todo = [k for k in ("afterimage", "refex") if not os.path.exists(target(f"nf_{k}.npy"))]
    mm = {}
    if "afterimage" in todo:
        mm["afterimage"] = memmap("nf_afterimage.npy", (len(sel), 100), np.float32)
    if "refex" in todo:
        mm["refex"] = memmap("nf_refex.npy", (len(sel), 2 * ca.N_REFEX + 2), np.float16)
    names = {}
    max_err = 0.0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        rows = np.where((sel >= a) & (sel < b))[0]
        want = sel[rows] - a
        sl = {k: np.asarray(s[k][a:b]) for k in STREAM}
        src, dst = sl["src"].astype(np.int64), sl["dst"].astype(np.int64)
        t0 = time.time()
        Cd, _ = ctx.unsw_context(src=src, dst=dst, dport=sl["dport"].astype(np.int64), t=sl["t"],
                                 sbytes=sl["ib"], dbytes=sl["ob"], dur=sl["dur"],
                                 proto=sl["proto"].astype(np.int64))
        err = float(np.abs(Cd[want] - corpus["C"][rows]).max())
        max_err = max(max_err, err)
        assert err < 1e-4, f"day {i}: role coordinates do not match the corpus (max err {err})"
        del Cd
        if "afterimage" in mm:
            A, names["afterimage"] = ca.afterimage(src, dst, sl["dport"].astype(np.int64), sl["t"], sl["ib"],
                                                   sl["ob"], want=want)
            mm["afterimage"][rows] = A
            del A
        if "refex" in mm:
            mm["refex"][rows] = ca.refex_context(src, dst, sl["t"], want=want)
        print(f"[nf] day {i}: {b - a:,} flows, {len(rows):,} corpus rows, role check err {err:.1e} "
              f"({time.time() - t0:.0f}s)", flush=True)
    for k, m in mm.items():
        finish(f"nf_{k}.npy", m)
    write_names("nf", names)
    print(f"[nf] done; role-coordinate alignment max err {max_err:.1e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=["unsw", "nf"])
    args = ap.parse_args()
    if "unsw" in args.corpus:
        build_unsw()
    if "nf" in args.corpus:
        build_nf()
