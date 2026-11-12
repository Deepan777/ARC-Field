"""ARC-Field: a masked, context-conditioned, heteroscedastic neural field.

For a flow with observable metadata m in R^d (partitioned into observability
groups g_1..g_G) and identity-free context coordinate c, the field models the
conditional law of each group given everything else,

    (mu_g, log sigma^2_g) = f_theta( c, m_{-g}, 1_{g} ),

with a sinusoidal-activation network (SIREN). Training masks a random group
(plus, for half of the batch, a random 15 % of the remaining features) and
fits a beta-NLL objective on the masked entries only. Because the entries that
are scored are never part of the input, the identity map is not a solution of
the objective (Proposition 2 in the manuscript).

At test time every group is masked in turn (G forward passes, batched) and the
per-feature standardised residual z_k = (m_k - mu_k) / sigma_k is returned.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """sin(omega_0 (W x + b)) with the SIREN initialisation (Sitzmann et al.,
    2020); identical in form to the SIRENLayer of GraphINR-IDS."""

    def __init__(self, n_in: int, n_out: int, omega_0: float, is_first: bool):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(n_in, n_out)
        with torch.no_grad():
            b = 1.0 / n_in if is_first else math.sqrt(6.0 / n_in) / omega_0
            self.linear.weight.uniform_(-b, b)
            self.linear.bias.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ARCField(nn.Module):
    def __init__(self, d: int, dc: int, hidden: int = 256, layers: int = 4,
                 omega_0: float = 30.0, activation: str = "sine",
                 hetero: bool = True, input_scale: float = 1.0 / 3.0):
        super().__init__()
        self.d, self.dc, self.hetero = d, dc, hetero
        self.input_scale = input_scale
        n_in = dc + 2 * d
        blocks: List[nn.Module] = []
        if activation == "sine":
            blocks.append(SineLayer(n_in, hidden, omega_0, True))
            for _ in range(layers - 1):
                blocks.append(SineLayer(hidden, hidden, omega_0, False))
        else:  # ReLU ablation with the same widths
            blocks += [nn.Linear(n_in, hidden), nn.ReLU()]
            for _ in range(layers - 1):
                blocks += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 2 * d if hetero else d)
        if activation == "sine":
            with torch.no_grad():
                b = math.sqrt(6.0 / hidden) / omega_0
                self.head.weight.uniform_(-b, b)
                self.head.bias.zero_()

    def forward(self, c: torch.Tensor, m: torch.Tensor, mask: torch.Tensor):
        """mask: 1 where the entry is hidden from the input (to be predicted)."""
        vis = m * (1.0 - mask)
        x = torch.cat([c, vis, mask], dim=1) if self.dc > 0 else torch.cat([vis, mask], dim=1)
        h = self.trunk(x * self.input_scale)
        out = self.head(h)
        if self.hetero:
            mu, logvar = out[:, : self.d], out[:, self.d:]
            return mu, torch.clamp(logvar, -7.0, 5.0)
        return out, None


def _masks(batch: int, groups: List[List[int]], d: int, extra_p: float,
           gen: torch.Generator, device) -> torch.Tensor:
    G = len(groups)
    gm = torch.zeros(G, d, device=device)
    for i, g in enumerate(groups):
        gm[i, g] = 1.0
    gid = torch.randint(0, G, (batch,), generator=gen, device="cpu").to(device)
    M = gm[gid]
    if extra_p > 0:
        sel = (torch.rand(batch, 1, generator=gen, device="cpu") < 0.5).to(device)
        extra = (torch.rand(batch, d, generator=gen, device="cpu") < extra_p).to(device)
        M = torch.clamp(M + (extra & sel).float(), 0, 1)
    return M


def train_field(model: ARCField, M: np.ndarray, C: np.ndarray, groups: List[List[int]],
                epochs: int = 40, lr: float = 2e-4, batch: int = 1024, seed: int = 0,
                device: str = "cuda", masked: bool = True, beta: float = 0.5,
                extra_p: float = 0.15, verbose: bool = False) -> Dict:
    """Fit on benign flows only. `masked=False` gives the unmasked ablation
    (plain conditional reconstruction of every feature from all features)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(device)
    Mt = torch.from_numpy(M).float().to(device)
    Ct = torch.from_numpy(C).float().to(device) if C is not None and C.shape[1] else \
        torch.zeros(len(M), 0, device=device)
    n = len(Mt)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * (n // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1),
                                                pct_start=0.1)
    hist, t0 = [], time.time()
    d = M.shape[1]
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        tot, cnt = 0.0, 0
        model.train()
        for b in range(n // batch):
            idx = perm[b * batch:(b + 1) * batch]
            m, c = Mt[idx], Ct[idx]
            if masked:
                mask = _masks(len(idx), groups, d, extra_p, gen, device)
            else:
                mask = torch.zeros(len(idx), d, device=device)
            mu, logvar = model(c, m, mask)
            w = mask if masked else torch.ones_like(mask)
            if model.hetero:
                var = torch.exp(logvar)
                nll = 0.5 * ((m - mu) ** 2 / var + logvar)
                nll = nll * var.detach() ** beta
                loss = (nll * w).sum() / w.sum()
            else:
                loss = (nn.functional.huber_loss(mu, m, reduction="none", delta=1.0) * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        hist.append(tot / max(cnt, 1))
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs} loss={hist[-1]:.4f}")
    return {"loss_history": hist, "train_seconds": time.time() - t0}


@torch.no_grad()
def residuals(model: ARCField, M: np.ndarray, C: np.ndarray, groups: List[List[int]],
              device: str = "cuda", masked: bool = True, batch: int = 8192) -> np.ndarray:
    """Per-feature standardised squared residuals z_k^2, shape (n, d).

    With `masked=True`, feature k's residual comes from the pass in which its
    own group is hidden. Without heteroscedastic head z_k^2 is the squared
    error.
    """
    model.eval().to(device)
    n, d = M.shape
    out = np.empty((n, d), dtype=np.float32)
    G = len(groups)
    gm = torch.zeros(G, d, device=device)
    for i, g in enumerate(groups):
        gm[i, g] = 1.0
    for s in range(0, n, batch):
        m = torch.from_numpy(M[s:s + batch]).float().to(device)
        c = (torch.from_numpy(C[s:s + batch]).float().to(device)
             if C is not None and C.shape[1] else torch.zeros(len(m), 0, device=device))
        if masked:
            B = len(m)
            mm = m.repeat(G, 1)
            cc = c.repeat(G, 1)
            mask = gm.repeat_interleave(B, dim=0)
            mu, logvar = model(cc, mm, mask)
            z2 = (mm - mu) ** 2
            if logvar is not None:
                z2 = z2 / torch.exp(logvar)
            z2 = (z2 * mask).view(G, B, d).sum(0)       # each feature masked once
        else:
            mu, logvar = model(c, m, torch.zeros_like(m))
            z2 = (m - mu) ** 2
            if logvar is not None:
                z2 = z2 / torch.exp(logvar)
        out[s:s + batch] = z2.cpu().numpy()
    return out


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
