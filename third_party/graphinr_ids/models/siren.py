"""
GraphINR-IDS SIREN Network
================================================================
Corresponds to: Section 5.5 (SIREN-Based Implicit Neural Network)

Implements Sinusoidal Representation Networks (SIREN) with:
  - sin(ω₀ · x) activation functions
  - SIREN-specific weight initialization
  - Gradient checkpointing for VRAM savings (Section 6.2)
  - 6 hidden layers, 512 neurons, ω₀ = 30

Based on: Sitzmann et al., "Implicit Neural Representations with
Periodic Activation Functions," NeurIPS 2020 [Ref 1 in proposal].
================================================================
"""
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SIRENLayer(nn.Module):
    """
    Single SIREN layer: y = sin(ω₀ · (Wx + b))

    Uses SIREN-specific initialization per the proposal (Section 5.5):
    - First layer: W ~ U(-1/n, 1/n) where n = input_dim
    - Hidden layers: W ~ U(-√(6/n)/ω₀, √(6/n)/ω₀)
    """

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        """SIREN weight initialization (Section 5.5 of proposal)."""
        with torch.no_grad():
            if self.is_first:
                # First layer: W₀ ~ U(-1/n, 1/n)
                bound = 1.0 / self.in_features
            else:
                # Hidden layers: W ~ U(-√(6/n)/ω₀, √(6/n)/ω₀)
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            # Bias initialized similarly
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ω₀-scaled linear transform followed by sin activation."""
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENNetwork(nn.Module):
    """
    Full SIREN network for implicit neural representation.

    Architecture (Section 5.5):
        h₀ = sin(ω₀ · (W₀x + b₀))
        h_l = sin(ω₀ · (W_l h_{l-1} + b_l)),  l = 1, ..., 5
        s_pred = W₆ h₅ + b₆   (linear output)

    Supports gradient checkpointing (Section 6.2) for VRAM savings.
    """

    def __init__(self, input_dim: int = 151, hidden_dim: int = 512,
                 hidden_layers: int = 6, output_dim: int = 23,
                 omega_0: float = 30.0,
                 gradient_checkpointing: bool = True,
                 checkpoint_every: int = 2):
        """
        Args:
            input_dim: 151 (96 graph + 32 temporal + 23 flow)
            hidden_dim: 512 neurons per layer
            hidden_layers: 6 hidden layers
            output_dim: 23 (reconstruct normalized flow features)
            omega_0: 30.0 frequency parameter
            gradient_checkpointing: Enable to save ~60% VRAM
            checkpoint_every: Checkpoint every N layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.use_checkpointing = gradient_checkpointing
        self.checkpoint_every = checkpoint_every

        # Build layers
        layers = []

        # First SIREN layer
        layers.append(SIRENLayer(input_dim, hidden_dim, omega_0=omega_0, is_first=True))

        # Hidden SIREN layers
        for _ in range(hidden_layers - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0, is_first=False))

        self.siren_layers = nn.ModuleList(layers)

        # Final linear output layer (no sin activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # Initialize output layer
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.uniform_(-bound, bound)

    def _checkpointed_forward(self, layer, x):
        """Apply gradient checkpointing to a single layer."""
        # checkpoint requires the function to not be a lambda and
        # use_reentrant=False for modern PyTorch
        return checkpoint(layer, x, use_reentrant=False)

    def forward(self, x: torch.Tensor, modulation_params=None) -> torch.Tensor:
        """
        Forward pass through SIREN network.

        Args:
            x: (B, 151) input coordinates

        Returns:
            s_pred: (B, 23) predicted normal traffic state
        """
        h = x
        for i, layer in enumerate(self.siren_layers):
            if self.use_checkpointing and self.training and i % self.checkpoint_every == 0:
                h = self._checkpointed_forward(layer, h)
            else:
                h = layer(h)
            if modulation_params is not None and i < len(modulation_params):
                gamma, beta = modulation_params[i]
                h = gamma * h + beta

        # Linear output (no sin)
        return self.output_layer(h)

    def forward_with_intermediates(self, x: torch.Tensor, modulation_params=None):
        """Forward pass returning intermediate activations (for GCM modulation)."""
        intermediates = []
        h = x
        for i, layer in enumerate(self.siren_layers):
            h = layer(h)
            if modulation_params is not None and i < len(modulation_params):
                gamma, beta = modulation_params[i]
                h = gamma * h + beta
            intermediates.append(h)
        output = self.output_layer(h)
        return output, intermediates
