"""
GraphINR-IDS Loss Functions
================================================================
Total loss = Huber(δ) + λ_lip * Lipschitz_reg + λ_compact * Compactness

Compactness loss minimises the variance of per-sample reconstruction
errors within each batch.  Concentrating the normal-traffic error
distribution makes attacks easier to separate at evaluation time.
================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberReconstructionLoss(nn.Module):
    """Huber reconstruction loss (delta configurable)."""

    def __init__(self, delta: float = 0.5):
        super().__init__()
        self.delta = delta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(prediction, target, delta=self.delta, reduction="mean")


class LipschitzRegularization(nn.Module):
    """
    Approximate Lipschitz regularisation using the mean squared weight value.
    This keeps the penalty comparable across ablations with different
    parameter counts instead of automatically punishing larger variants.
    """

    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight

    def forward(self, model: nn.Module) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        total_elems = 0
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                reg = reg + torch.sum(param ** 2)
                total_elems += param.numel()
        if total_elems == 0:
            return reg
        return self.weight * (reg / total_elems)


class CompactnessLoss(nn.Module):
    """
    Variance compactness loss.

    Minimises the variance of per-sample reconstruction errors within
    each training batch.  This tightens the normal-traffic error
    distribution, increasing separation from attack flows at evaluation.

    L_compact = Var_batch[ mean_feat[ (pred_i - tgt_i)^2 ] ]
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_errors      = (prediction - target) ** 2    # (B, D)
        per_sample_err = sq_errors.mean(dim=1)         # (B,)
        var_loss       = per_sample_err.var()           # scalar
        return self.weight * var_loss


class GraphINRLoss(nn.Module):
    """
    Combined loss: Huber + Lipschitz + Compactness.
    """

    def __init__(self, huber_delta: float = 0.5,
                 lipschitz_weight: float = 0.001,
                 compactness_weight: float = 0.1):
        super().__init__()
        self.reconstruction_loss = HuberReconstructionLoss(delta=huber_delta)
        self.lipschitz_reg       = LipschitzRegularization(weight=lipschitz_weight)
        self.compactness_loss    = CompactnessLoss(weight=compactness_weight)

    def forward(self, output: dict, model: nn.Module) -> dict:
        recon    = self.reconstruction_loss(output["prediction"], output["target"])
        reg      = self.lipschitz_reg(model)
        compact  = self.compactness_loss(output["prediction"], output["target"])
        total    = recon + reg + compact

        return {
            "total":          total,
            "reconstruction": recon,
            "regularization": reg,
            "compactness":    compact,
        }
