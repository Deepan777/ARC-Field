"""
GraphINR-IDS Hierarchical Frequency Decomposition (HFD)
================================================================
Corresponds to: Section 5.5, Innovation 2

"A Hierarchical Frequency Decomposition (HFD) that splits the
SIREN into two branches — a low-frequency branch (ω₀ = 10)
capturing smooth baseline traffic patterns and a high-frequency
branch (ω₀ = 60) capturing rapid transient behaviors — whose
outputs are combined via learned attention weights."
================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .siren import SIRENNetwork


class HierarchicalFrequencyDecomposition(nn.Module):
    """
    Dual-branch SIREN architecture that decomposes network traffic state
    into low-frequency (baseline) and high-frequency (transient) components.

    Low-freq branch (ω₀ = 10): Captures smooth diurnal patterns, average
    flow characteristics, baseline traffic volume.

    High-freq branch (ω₀ = 60): Captures burst events, rapid traffic
    changes, and anomalous transient patterns.

    Branches are fused via learned attention weights.
    """

    def __init__(self, input_dim: int = 151, hidden_dim: int = 512,
                 hidden_layers: int = 6, output_dim: int = 23,
                 low_omega: float = 10.0, high_omega: float = 60.0,
                 attention_dim: int = 64,
                 gradient_checkpointing: bool = True):
        """
        Args:
            input_dim: 151-dim input coordinate
            hidden_dim: 512 neurons
            hidden_layers: 6 layers each branch
            output_dim: 23-dim output
            low_omega: ω₀ = 10 for baseline patterns
            high_omega: ω₀ = 60 for transient patterns
            attention_dim: 64 for attention fusion
            gradient_checkpointing: VRAM optimization
        """
        super().__init__()

        # Low-frequency branch (smooth baseline)
        self.low_freq_branch = SIRENNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            omega_0=low_omega,
            gradient_checkpointing=gradient_checkpointing,
        )

        # High-frequency branch (rapid transients)
        self.high_freq_branch = SIRENNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            omega_0=high_omega,
            gradient_checkpointing=gradient_checkpointing,
        )

        # Attention-based fusion network
        # Takes the input coordinate and produces per-feature attention weights
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, output_dim),
            nn.Sigmoid(),  # Weights in [0, 1]
        )

    def forward(self, x: torch.Tensor, low_modulation_params=None,
                high_modulation_params=None) -> torch.Tensor:
        """
        Forward pass through dual-branch HFD.

        Args:
            x: (B, 151) input coordinates

        Returns:
            output: (B, 23) fused prediction
            Also returns: (low_out, high_out, attention_weights) for analysis
        """
        # Low-frequency prediction
        low_out = self.low_freq_branch(
            x, modulation_params=low_modulation_params
        )     # (B, 23)

        # High-frequency prediction
        high_out = self.high_freq_branch(
            x, modulation_params=high_modulation_params
        )   # (B, 23)

        # Attention weights (per-feature, per-sample)
        alpha = self.attention_net(x)          # (B, 23) in [0, 1]

        # Weighted combination
        output = alpha * low_out + (1.0 - alpha) * high_out  # (B, 23)

        return output

    def forward_with_components(self, x: torch.Tensor, low_modulation_params=None,
                                high_modulation_params=None):
        """Forward returning all components for analysis/visualization."""
        low_out = self.low_freq_branch(x, modulation_params=low_modulation_params)
        high_out = self.high_freq_branch(x, modulation_params=high_modulation_params)
        alpha = self.attention_net(x)
        output = alpha * low_out + (1.0 - alpha) * high_out
        return output, low_out, high_out, alpha
