"""
GraphINR-IDS Temporal Encoder
================================================================
Corresponds to: Section 5.4 (Module 3: Temporal-Flow Coordinate Generator)

Multi-scale sinusoidal temporal encoding:
  TE(t) = [sin(2πf₁t), cos(2πf₁t), ..., sin(2πf_Lt), cos(2πf_Lt)]

Where {f₁, ..., f_L} are L=16 log-linearly spaced frequencies
from 1 Hz to 10 kHz, capturing temporal patterns from second-level
periodicities to sub-millisecond dynamics.

Output: 32-dimensional temporal encoding (2L = 32).
================================================================
"""
import math
import torch
import torch.nn as nn
import numpy as np


class TemporalEncoder(nn.Module):
    """
    Multi-scale sinusoidal temporal encoding for continuous time coordinates.

    Unlike discrete time binning, this encodes time as a continuous signal
    using multiple frequency bands, enabling the INR to learn periodic
    patterns at different time scales.
    """

    def __init__(self, num_frequencies: int = 16,
                 freq_min: float = 1.0, freq_max: float = 10000.0):
        """
        Args:
            num_frequencies: L = 16 (Section 5.4)
            freq_min: 1 Hz
            freq_max: 10 kHz (10000 Hz)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.encoding_dim = 2 * num_frequencies  # 32

        # Log-linearly spaced frequencies: f_l = freq_min * (freq_max/freq_min)^(l/(L-1))
        log_freqs = torch.linspace(
            math.log(freq_min),
            math.log(freq_max),
            num_frequencies
        )
        frequencies = torch.exp(log_freqs)  # (L,)

        # Store as buffer (not trainable, but moves with model to GPU)
        # Pre-multiply by 2π for efficiency
        self.register_buffer("omega", 2.0 * math.pi * frequencies)  # (L,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous timestamp into multi-scale sinusoidal features.

        TE(t) = [sin(2πf₁t), cos(2πf₁t), ..., sin(2πf_Lt), cos(2πf_Lt)]

        Args:
            t: (B,) or (B, 1) continuous timestamps

        Returns:
            te: (B, 2L) = (B, 32) temporal encodings
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # (B, 1)

        # (B, 1) * (L,) → (B, L) via broadcasting
        phase = t * self.omega.unsqueeze(0)  # (B, L)

        # Concatenate sin and cos
        te = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)  # (B, 2L)
        return te
