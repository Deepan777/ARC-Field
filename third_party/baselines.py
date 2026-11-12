"""Standard one-class baselines (Isolation Forest, LOF, OC-SVM, dense/LSTM/
Transformer autoencoders, Deep SVDD, DAGMM, MemAE, USAD, denoising-diffusion
detector). Each factory fits on benign training inputs and returns a scoring
function; all detectors receive identical inputs and partitions. Reused
unchanged from P. Deepanramkumar and A. Helen Sharmila, GraphINR-IDS, J. King Saud
Univ. - Comput. Inf. Sci. 38(8):916, 2026, doi:10.1007/s44443-026-01260-x."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _loader(X: np.ndarray, bs: int = 1024, shuffle: bool = False):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                       drop_last=shuffle)


def _train(model, X, epochs, lr, step_fn, wd=1e-5, bs=1024, verbose=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    dl = _loader(X, bs=bs, shuffle=True)
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for (xb,) in dl:
            loss = step_fn(model, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(xb)
            n += len(xb)
        if verbose and (ep + 1) % 10 == 0:
            print(f"      ep{ep+1} loss={tot/max(1,n):.5f}")
    return model


@torch.no_grad()
def _apply(model, X, fn, bs=4096) -> np.ndarray:
    model.eval()
    out = []
    for (xb,) in _loader(X, bs=bs):
        out.append(fn(model, xb).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# classical
# ---------------------------------------------------------------------------

def iso_forest(Xtr, seed):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xtr)
    return lambda X: -m.score_samples(X)


def lof(Xtr, seed, n_fit=20000):
    from sklearn.neighbors import LocalOutlierFactor
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xtr), min(n_fit, len(Xtr)), replace=False)
    m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(Xtr[sub])
    return lambda X: -m.score_samples(X)


def ocsvm(Xtr, seed, n_fit=5000):
    from sklearn.svm import OneClassSVM
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xtr), min(n_fit, len(Xtr)), replace=False)
    Z = Xtr[sub]
    # median heuristic for gamma, as in the original configuration
    idx = rng.choice(len(Z), min(1000, len(Z)), replace=False)
    d = np.linalg.norm(Z[idx][:, None, :] - Z[idx][None, :, :], axis=-1)
    med = np.median(d[d > 0]) if (d > 0).any() else 1.0
    m = OneClassSVM(kernel="rbf", gamma=1.0 / (2 * med ** 2), nu=0.05).fit(Z)
    return lambda X: -m.decision_function(X)


# ---------------------------------------------------------------------------
# deep one-class models
# ---------------------------------------------------------------------------

class DenseAE(nn.Module):
    def __init__(self, d, h=(64, 32)):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h[0]), nn.ReLU(),
                                 nn.Linear(h[0], h[1]))
        self.dec = nn.Sequential(nn.Linear(h[1], h[0]), nn.ReLU(),
                                nn.Linear(h[0], d))

    def forward(self, x):
        return self.dec(self.enc(x))


def dense_ae(Xtr, seed, epochs=50, d=None):
    torch.manual_seed(seed)
    m = DenseAE(Xtr.shape[1])
    _train(m, Xtr, epochs, 1e-3, lambda mo, xb: F.mse_loss(mo(xb), xb))
    return lambda X: _apply(m, X, lambda mo, xb: ((mo(xb) - xb) ** 2).mean(1))


class LSTMAE(nn.Module):
    def __init__(self, d, h=64, L=16):
        super().__init__()
        self.L = L
        self.enc = nn.LSTM(d, h, num_layers=2, batch_first=True)
        self.dec = nn.LSTM(h, h, num_layers=2, batch_first=True)
        self.out = nn.Linear(h, d)

    def forward(self, x):                       # x: (B, L, d)
        _, (hn, _) = self.enc(x)
        z = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(z)
        return self.out(y)


def lstm_ae(Xtr, seed, epochs=30, L=16):
    """Sequence autoencoder over fixed-length windows of consecutive flows.

    Scores are assigned per flow by averaging the reconstruction error of every
    window the flow belongs to, using a stride-1 sliding window. This removes
    the window-boundary dependence that made the original submission's LSTM-AE
    numbers depend on the composition of the evaluated subset.
    """
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    m = LSTMAE(d, L=L)

    def windows(X):
        n = len(X)
        if n < L:
            X = np.pad(X, ((0, L - n), (0, 0)), mode="edge")
            n = L
        s = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)  # (n-L+1, d, L)
        return np.ascontiguousarray(s.transpose(0, 2, 1))           # (n-L+1, L, d)

    Wtr = windows(Xtr)
    sel = np.random.default_rng(seed).choice(len(Wtr), min(len(Wtr), 40000), replace=False)
    _train(m, Wtr[sel], epochs, 1e-3, lambda mo, xb: F.mse_loss(mo(xb), xb), bs=256)

    def score(X):
        W = windows(X)
        err = _apply(m, W, lambda mo, xb: ((mo(xb) - xb) ** 2).mean(2), bs=512)  # (nw, L)
        n = len(X)
        acc = np.zeros(max(n, L))
        cnt = np.zeros(max(n, L))
        for j in range(err.shape[1]):
            acc[j:j + err.shape[0]] += err[:, j]
            cnt[j:j + err.shape[0]] += 1
        return (acc / np.maximum(cnt, 1))[:n]
    return score


class DeepSVDD(nn.Module):
    def __init__(self, d, h=(64, 32)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h[0], bias=False), nn.ReLU(),
                                 nn.Linear(h[0], h[1], bias=False))

    def forward(self, x):
        return self.net(x)


def deep_svdd(Xtr, seed, epochs=50):
    """One-class Deep SVDD (Ruff et al., 2018), soft-boundary omitted.

    Centre c is fixed from an initial forward pass, as prescribed; bias terms
    are removed to prevent the trivial constant solution.
    """
    torch.manual_seed(seed)
    m = DeepSVDD(Xtr.shape[1])
    with torch.no_grad():
        c = _apply(m, Xtr, lambda mo, xb: mo(xb)).mean(0)
    c = torch.from_numpy(c).float()
    c[c.abs() < 1e-6] = 1e-6
    _train(m, Xtr, epochs, 1e-3, lambda mo, xb: ((mo(xb) - c) ** 2).sum(1).mean(), wd=1e-6)
    return lambda X: _apply(m, X, lambda mo, xb: ((mo(xb) - c) ** 2).sum(1))


class DAGMM(nn.Module):
    def __init__(self, d, zc=2, K=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 60), nn.Tanh(), nn.Linear(60, 30),
                                 nn.Tanh(), nn.Linear(30, zc))
        self.dec = nn.Sequential(nn.Linear(zc, 30), nn.Tanh(), nn.Linear(30, 60),
                                 nn.Tanh(), nn.Linear(60, d))
        self.est = nn.Sequential(nn.Linear(zc + 2, 10), nn.Tanh(), nn.Dropout(0.5),
                                 nn.Linear(10, K), nn.Softmax(dim=1))
        self.K, self.zdim = K, zc + 2

    def latent(self, x):
        zc = self.enc(x)
        xr = self.dec(zc)
        eu = (x - xr).norm(2, dim=1, keepdim=True) / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        cs = F.cosine_similarity(x, xr, dim=1).unsqueeze(1)
        return torch.cat([zc, eu, cs], dim=1), xr

    def forward(self, x):
        z, xr = self.latent(x)
        return z, xr, self.est(z)


def _dagmm_params(z, gamma):
    N = gamma.size(0)
    phi = gamma.sum(0) / N
    mu = (gamma.unsqueeze(-1) * z.unsqueeze(1)).sum(0) / (gamma.sum(0).unsqueeze(-1) + 1e-8)
    dz = z.unsqueeze(1) - mu.unsqueeze(0)
    cov = ((gamma.unsqueeze(-1).unsqueeze(-1) * dz.unsqueeze(-1) * dz.unsqueeze(-2)).sum(0)
           / (gamma.sum(0).unsqueeze(-1).unsqueeze(-1) + 1e-8))
    return phi, mu, cov


def _dagmm_energy(z, phi, mu, cov):
    K, D = mu.shape
    eye = torch.eye(D).unsqueeze(0) * 1e-4
    cov = cov + eye
    dz = z.unsqueeze(1) - mu.unsqueeze(0)                        # (N,K,D)
    inv = torch.linalg.inv(cov)
    m = torch.einsum("nkd,kde,nke->nk", dz, inv, dz)
    det = torch.linalg.det(cov).clamp(min=1e-12)
    logp = -0.5 * m - 0.5 * torch.log(det).unsqueeze(0) - 0.5 * D * np.log(2 * np.pi)
    return -torch.logsumexp(logp + torch.log(phi.clamp(min=1e-8)).unsqueeze(0), dim=1)


def dagmm(Xtr, seed, epochs=50):
    torch.manual_seed(seed)
    m = DAGMM(Xtr.shape[1])

    def step(mo, xb):
        z, xr, g = mo(xb)
        phi, mu, cov = _dagmm_params(z, g)
        e = _dagmm_energy(z, phi, mu, cov)
        pen = torch.diagonal(cov, dim1=1, dim2=2).clamp(min=1e-12).reciprocal().sum()
        return F.mse_loss(xr, xb) + 0.1 * e.mean() + 0.005 * pen

    _train(m, Xtr, epochs, 1e-4, step)
    with torch.no_grad():
        zs, gs = [], []
        for (xb,) in _loader(Xtr, bs=4096):
            z, _, g = m(xb)
            zs.append(z)
            gs.append(g)
        phi, mu, cov = _dagmm_params(torch.cat(zs), torch.cat(gs))
    return lambda X: _apply(m, X, lambda mo, xb: _dagmm_energy(mo.latent(xb)[0], phi, mu, cov))


class MemAE(nn.Module):
    """Memory-augmented autoencoder (Gong et al., 2019)."""

    def __init__(self, d, h=32, N=100, shrink=0.0025):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, h))
        self.dec = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, d))
        self.mem = nn.Parameter(torch.randn(N, h) * 0.05)
        self.shrink = shrink

    def forward(self, x):
        z = self.enc(x)
        att = F.softmax(F.normalize(z, dim=1) @ F.normalize(self.mem, dim=1).t(), dim=1)
        att = F.relu(att - self.shrink) * att / (torch.abs(att - self.shrink) + 1e-12)
        att = att / (att.sum(1, keepdim=True) + 1e-12)
        return self.dec(att @ self.mem), att


def memae(Xtr, seed, epochs=50):
    torch.manual_seed(seed)
    m = MemAE(Xtr.shape[1])

    def step(mo, xb):
        xr, att = mo(xb)
        ent = (-att * torch.log(att + 1e-12)).sum(1).mean()
        return F.mse_loss(xr, xb) + 0.0002 * ent

    _train(m, Xtr, epochs, 1e-3, step)
    return lambda X: _apply(m, X, lambda mo, xb: ((mo(xb)[0] - xb) ** 2).mean(1))


class USAD(nn.Module):
    """Two-decoder adversarially trained autoencoder (Audibert et al., 2020)."""

    def __init__(self, d, h=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, h), nn.ReLU())
        self.d1 = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, d), nn.Sigmoid())
        self.d2 = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, d), nn.Sigmoid())


def usad(Xtr, seed, epochs=50, alpha=0.5):
    torch.manual_seed(seed)
    m = USAD(Xtr.shape[1])
    o1 = torch.optim.AdamW(list(m.enc.parameters()) + list(m.d1.parameters()), lr=1e-3)
    o2 = torch.optim.AdamW(list(m.enc.parameters()) + list(m.d2.parameters()), lr=1e-3)
    dl = _loader(Xtr, bs=1024, shuffle=True)
    for ep in range(1, epochs + 1):
        m.train()
        n = ep
        for (xb,) in dl:
            z = m.enc(xb)
            w1, w2 = m.d1(z), m.d2(z)
            w3 = m.d2(m.enc(w1))
            l1 = (1 / n) * F.mse_loss(w1, xb) + (1 - 1 / n) * F.mse_loss(w3, xb)
            o1.zero_grad(set_to_none=True)
            l1.backward()
            o1.step()

            z = m.enc(xb)
            w1, w2 = m.d1(z), m.d2(z)
            w3 = m.d2(m.enc(w1))
            l2 = (1 / n) * F.mse_loss(w2, xb) - (1 - 1 / n) * F.mse_loss(w3, xb)
            o2.zero_grad(set_to_none=True)
            l2.backward()
            o2.step()

    def sc(mo, xb):
        z = mo.enc(xb)
        w1 = mo.d1(z)
        w3 = mo.d2(mo.enc(w1))
        return alpha * ((w1 - xb) ** 2).mean(1) + (1 - alpha) * ((w3 - xb) ** 2).mean(1)
    return lambda X: _apply(m, X, sc)


class TransformerAE(nn.Module):
    """Transformer encoder-decoder over the feature axis.

    Each of the 36 flow features is treated as a token with a learned embedding,
    so self-attention models inter-feature dependence without imposing a time
    window on the flow stream.
    """

    def __init__(self, d, dm=32, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(1, dm)
        self.pos = nn.Parameter(torch.randn(1, d, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(dm, heads, dm * 4, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dm, 1)

    def forward(self, x):
        h = self.proj(x.unsqueeze(-1)) + self.pos
        return self.out(self.tf(h)).squeeze(-1)


def transformer_ae(Xtr, seed, epochs=40):
    torch.manual_seed(seed)
    m = TransformerAE(Xtr.shape[1])
    _train(m, Xtr, epochs, 5e-4, lambda mo, xb: F.mse_loss(mo(xb), xb), bs=512)
    return lambda X: _apply(m, X, lambda mo, xb: ((mo(xb) - xb) ** 2).mean(1))


class DDPM(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


def diffusion_ad(Xtr, seed, epochs=50, T=100, eval_t=20):
    """Denoising-diffusion anomaly detector.

    A DDPM is fitted to the benign feature distribution; the anomaly score is
    the expected noise-prediction error at a fixed diffusion step, averaged over
    several noise draws. Benign flows lie on the learned density and are
    denoised accurately; off-manifold flows are not.
    """
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    m = DDPM(d)
    betas = torch.linspace(1e-4, 0.02, T)
    ab = torch.cumprod(1.0 - betas, dim=0)

    def step(mo, xb):
        ti = torch.randint(0, T, (len(xb),))
        a = ab[ti].unsqueeze(1)
        eps = torch.randn_like(xb)
        xt = a.sqrt() * xb + (1 - a).sqrt() * eps
        return F.mse_loss(mo(xt, ti.float().unsqueeze(1) / T), eps)

    _train(m, Xtr, epochs, 1e-3, step)

    @torch.no_grad()
    def sc(mo, xb):
        a = ab[eval_t]
        tot = 0.0
        g = torch.Generator().manual_seed(seed)
        for _ in range(4):
            eps = torch.randn(xb.shape, generator=g)
            xt = a.sqrt() * xb + (1 - a).sqrt() * eps
            tt = torch.full((len(xb), 1), eval_t / T)
            tot = tot + ((mo(xt, tt) - eps) ** 2).mean(1)
        return tot / 4
    return lambda X: _apply(m, X, sc)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

BASELINES = {
    "Isolation Forest": iso_forest,
    "Local Outlier Factor": lof,
    "One-Class SVM": ocsvm,
    "Dense Autoencoder": dense_ae,
    "LSTM Autoencoder": lstm_ae,
    "Deep SVDD": deep_svdd,
    "DAGMM": dagmm,
    "MemAE": memae,
    "USAD": usad,
    "Transformer Autoencoder": transformer_ae,
    "Diffusion (DDPM) detector": diffusion_ad,
}


