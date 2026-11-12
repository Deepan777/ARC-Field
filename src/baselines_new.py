"""Additional baselines (the eleven one-class baselines of GraphINR-IDS are in
third_party/baselines.py, unchanged).

* KitNET  - the anomaly detector of Kitsune (Mirsky et al., NDSS 2018):
            correlation-based feature mapping into small clusters, an ensemble
            of compact autoencoders, and an output autoencoder over the
            ensemble's RMSEs. Trained in mini-batch epochs rather than one
            online pass so that it sees the same data budget as every other
            detector.
* MCM     - Masked Cell Modeling (Yin et al., ICLR 2024): a learnable mask
            generator produces K diverse soft masks; an autoencoder
            reconstructs the full record from each masked view; the anomaly
            score is the mean reconstruction error over the K views.

Both return a scoring function, following the reused baseline interface.
The detectors are re-implemented from their papers' descriptions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _fit(model, X, epochs, lr, step, bs=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X).float().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(DEV)
        for b in range(0, n - bs + 1, bs):
            loss = step(model, Xt[perm[b:b + bs]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _apply(fn, X, bs=8192):
    out = []
    for s in range(0, len(X), bs):
        out.append(fn(torch.from_numpy(X[s:s + bs]).float().to(DEV)).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# KitNET
# ---------------------------------------------------------------------------

class _AE(nn.Module):
    def __init__(self, d, ratio=0.75):
        super().__init__()
        h = max(1, int(np.ceil(d * ratio)))
        self.enc = nn.Linear(d, h)
        self.dec = nn.Linear(h, d)

    def forward(self, x):
        return torch.sigmoid(self.dec(torch.sigmoid(self.enc(x))))


class KitNETModel(nn.Module):
    def __init__(self, clusters, ratio=0.75):
        super().__init__()
        self.clusters = [torch.tensor(c, dtype=torch.long) for c in clusters]
        self.ens = nn.ModuleList([_AE(len(c), ratio) for c in clusters])
        self.out = _AE(len(clusters), ratio)

    def ens_rmse(self, x):
        r = []
        for c, ae in zip(self.clusters, self.ens):
            xc = x[:, c.to(x.device)]
            r.append(torch.sqrt(((ae(xc) - xc) ** 2).mean(1) + 1e-12))
        return torch.stack(r, 1)


def kitnet(Xtr, seed, epochs=20, max_cluster=10):
    torch.manual_seed(seed)
    # Feature mapper: agglomerative clustering on correlation distance.
    Z = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(20000, len(Xtr)), replace=False)]
    C = np.corrcoef(Z.T)
    C = np.nan_to_num(C, nan=0.0)
    D = 1 - np.abs(C)
    np.fill_diagonal(D, 0)
    L = linkage(squareform(np.clip(D, 0, None), checks=False), method="complete")
    # smallest number of clusters such that none exceeds max_cluster
    for k in range(1, Xtr.shape[1] + 1):
        lab = fcluster(L, k, criterion="maxclust")
        if np.bincount(lab).max() <= max_cluster:
            break
    clusters = [np.where(lab == u)[0].tolist() for u in np.unique(lab)]
    m = KitNETModel(clusters).to(DEV)
    lo, hi = Xtr.min(0), Xtr.max(0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    N = lambda X: np.clip((X - lo) / rng, 0, 1).astype(np.float32)
    Xn = N(Xtr)
    _fit(m.ens, Xn, epochs, 1e-3, lambda mo, xb: sum(
        F.mse_loss(ae(xb[:, c.to(DEV)]), xb[:, c.to(DEV)]) for c, ae in zip(m.clusters, mo)), seed=seed)
    with torch.no_grad():
        R = _apply(m.ens_rmse, Xn)
    rlo, rhi = R.min(0), R.max(0)
    rr = np.where(rhi - rlo > 0, rhi - rlo, 1.0)
    RN = lambda R_: np.clip((R_ - rlo) / rr, 0, 1).astype(np.float32)
    _fit(m.out, RN(R), epochs, 1e-3, lambda mo, xb: F.mse_loss(mo(xb), xb), seed=seed)

    def score(X):
        R_ = _apply(m.ens_rmse, N(X))
        Rn = RN(R_)
        return _apply(lambda t: torch.sqrt(((m.out(t) - t) ** 2).mean(1)), Rn)
    return score


# ---------------------------------------------------------------------------
# MCM
# ---------------------------------------------------------------------------

class MCMModel(nn.Module):
    def __init__(self, d, K=15, h=128, z=32):
        super().__init__()
        self.K = K
        self.maskgen = nn.Sequential(nn.Linear(d, h), nn.LeakyReLU(0.2), nn.Linear(h, d * K))
        self.enc = nn.Sequential(nn.Linear(d, h), nn.LeakyReLU(0.2), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.LeakyReLU(0.2), nn.Linear(h, d))

    def forward(self, x):
        B, d = x.shape
        M = torch.sigmoid(self.maskgen(x)).view(B, self.K, d)
        xm = (x.unsqueeze(1) * M).view(B * self.K, d)
        xr = self.dec(self.enc(xm)).view(B, self.K, d)
        return xr, M


def mcm(Xtr, seed, epochs=30, K=15, lam=5.0):
    torch.manual_seed(seed)
    m = MCMModel(Xtr.shape[1], K=K).to(DEV)

    def step(mo, xb):
        xr, M = mo(xb)
        rec = ((xr - xb.unsqueeze(1)) ** 2).mean()
        Mn = F.normalize(M, dim=2)
        sim = torch.bmm(Mn, Mn.transpose(1, 2))                 # (B, K, K)
        off = sim - torch.diag_embed(torch.diagonal(sim, dim1=1, dim2=2))
        div = off.sum((1, 2)).mean() / (K * (K - 1))
        return rec + lam * div / 100.0

    _fit(m, Xtr, epochs, 1e-3, step, seed=seed)
    return lambda X: _apply(lambda t: ((m(t)[0] - t.unsqueeze(1)) ** 2).mean((1, 2)), X)


NEW_BASELINES = {"KitNET": kitnet, "MCM": mcm}


# ---------------------------------------------------------------------------
# Deep Isolation Forest (Xu et al., IEEE TKDE 2023)
# ---------------------------------------------------------------------------

def dif(Xtr, seed, n_reps=50, n_trees=6, max_samples=256, hidden=(500, 100), out_dim=128):
    """Deep Isolation Forest: an ensemble of isolation forests, each built on a
    representation produced by a randomly initialised (untrained) MLP. The
    anomaly score is the mean isolation score over the representations."""
    from sklearn.ensemble import IsolationForest
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    nets, forests = [], []
    g = torch.Generator().manual_seed(seed)
    for r in range(n_reps):
        dims = (d,) + tuple(hidden) + (out_dim,)
        Ws = [torch.randn(dims[i], dims[i + 1], generator=g) / np.sqrt(dims[i]) for i in range(len(dims) - 1)]
        nets.append(Ws)

    def embed(Ws, X):
        H = torch.from_numpy(np.asarray(X, np.float32))
        for i, W in enumerate(Ws):
            H = H @ W
            if i < len(Ws) - 1:
                H = torch.tanh(H)
        return H.numpy()

    for r, Ws in enumerate(nets):
        f = IsolationForest(n_estimators=n_trees, max_samples=max_samples,
                            random_state=int(rng.integers(1 << 31)), n_jobs=-1)
        f.fit(embed(Ws, Xtr))
        forests.append(f)

    def score(X):
        s = np.zeros(len(X))
        for Ws, f in zip(nets, forests):
            s += -f.score_samples(embed(Ws, X))
        return s / len(nets)
    return score


# ---------------------------------------------------------------------------
# Anomal-E (Caville et al., Knowledge-Based Systems 2022)
# ---------------------------------------------------------------------------

class _EGraphSAGE(nn.Module):
    """One E-GraphSAGE layer: a node aggregates the features of its incident
    edges (mean), concatenates them with its own state, and the embedding of
    an edge is the concatenation of its endpoints' embeddings."""

    def __init__(self, d_edge, d_node=128):
        super().__init__()
        self.lin = nn.Linear(1 + d_edge, d_node)
        self.d_node = d_node

    def forward(self, src, dst, E, n_nodes):
        agg = torch.zeros(n_nodes, E.shape[1], device=E.device)
        cnt = torch.zeros(n_nodes, 1, device=E.device)
        ones = torch.ones(len(src), 1, device=E.device)
        for idx in (src, dst):
            agg.index_add_(0, idx, E)
            cnt.index_add_(0, idx, ones)
        agg = agg / cnt.clamp(min=1)
        h = torch.relu(self.lin(torch.cat([torch.ones(n_nodes, 1, device=E.device), agg], 1)))
        return torch.cat([h[src], h[dst]], 1)


class AnomalE:
    """Anomal-E: E-GraphSAGE trained with Deep Graph Infomax on the benign flow
    graph (corruption = permuted edge features), then an Isolation Forest on
    the learned edge embeddings. Graphs are built over consecutive chunks of
    the record stream (emission order), with source hosts and destination
    service endpoints (host:port) as nodes."""

    def __init__(self, seed, chunk=10000, d_node=128, epochs=20, lr=1e-3):
        self.seed, self.chunk, self.d_node, self.epochs, self.lr = seed, chunk, d_node, epochs, lr

    def _graphs(self, X, src, dstsvc):
        for s in range(0, len(X), self.chunk):
            sl = slice(s, s + self.chunk)
            keys = np.concatenate([src[sl].astype(str), dstsvc[sl].astype(str)])
            _, inv = np.unique(keys, return_inverse=True)
            n = len(src[sl])
            yield (sl, torch.from_numpy(inv[:n]).long().to(DEV), torch.from_numpy(inv[n:]).long().to(DEV),
                   torch.from_numpy(np.asarray(X[sl], np.float32)).to(DEV), int(inv.max()) + 1)

    def fit(self, X, src, dstsvc):
        from sklearn.ensemble import IsolationForest
        torch.manual_seed(self.seed)
        self.enc = _EGraphSAGE(X.shape[1], self.d_node).to(DEV)
        self.W = nn.Parameter(torch.empty(2 * self.d_node, 2 * self.d_node, device=DEV))
        nn.init.xavier_uniform_(self.W)
        opt = torch.optim.Adam(list(self.enc.parameters()) + [self.W], lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        graphs = list(self._graphs(X, src, dstsvc))
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            for _, s_, d_, E, n in graphs:
                pos = self.enc(s_, d_, E, n)
                perm = torch.randperm(len(E), generator=g).to(DEV)
                neg = self.enc(s_, d_, E[perm], n)
                summ = torch.sigmoid(pos.mean(0))
                lp = pos @ (self.W @ summ)
                ln = neg @ (self.W @ summ)
                loss = bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        Z = self.embed(X, src, dstsvc)
        self.iforest = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(Z)
        return self

    @torch.no_grad()
    def embed(self, X, src, dstsvc):
        out = np.zeros((len(X), 2 * self.d_node), np.float32)
        for sl, s_, d_, E, n in self._graphs(X, src, dstsvc):
            out[sl] = self.enc(s_, d_, E, n).cpu().numpy()
        return out

    def score(self, X, src, dstsvc):
        return -self.iforest.score_samples(self.embed(X, src, dstsvc))


NEW_BASELINES["DIF"] = dif
