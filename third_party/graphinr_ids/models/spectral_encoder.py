"""
GraphINR-IDS Spectral Graph Encoder
================================================================
Corresponds to: Section 5.3 (Module 2: Spectral Graph Encoder)

Computes edge positional encodings from precomputed Laplacian
eigenvectors:
  PE(v_i, v_j) = [u_K(v_i) || u_K(v_j) || u_K(v_i) ⊙ u_K(v_j)] ∈ ℝ^(3K)

K = 32 eigenvectors → 96-dimensional encoding per edge.
================================================================
"""
import torch
import torch.nn as nn
import numpy as np


class SpectralGraphEncoder(nn.Module):
    """
    Converts precomputed spectral embeddings (Laplacian eigenvectors)
    into edge-level positional encodings.

    The eigenvectors are precomputed by NetworkGraphBuilder and stored
    as a fixed buffer. This module performs the concatenation and
    Hadamard product operations defined in Section 5.3.
    """

    def __init__(self, num_nodes: int, spectral_k: int = 32):
        """
        Args:
            num_nodes: Number of nodes in the network graph
            spectral_k: K = number of eigenvectors (default 32)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.spectral_k = spectral_k
        self.edge_encoding_dim = 3 * spectral_k  # 96

        # Register eigenvectors as a non-trainable buffer
        # Will be set via set_eigenvectors()
        self.register_buffer(
            "eigenvectors",
            torch.zeros(num_nodes, spectral_k, dtype=torch.float32)
        )

    def set_eigenvectors(self, eigvec: np.ndarray):
        """
        Set the Laplacian eigenvectors from the graph builder.

        Args:
            eigvec: (num_nodes, K) numpy array of eigenvectors
        """
        assert eigvec.shape == (self.num_nodes, self.spectral_k), \
            f"Expected ({self.num_nodes}, {self.spectral_k}), got {eigvec.shape}"
        self.eigenvectors.copy_(torch.from_numpy(eigvec).float())

    def forward(self, src_idx: torch.Tensor, dst_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute edge positional encoding.

        PE(v_i, v_j) = [u_K(v_i) || u_K(v_j) || u_K(v_i) ⊙ u_K(v_j)]

        Args:
            src_idx: (B,) source node indices
            dst_idx: (B,) destination node indices

        Returns:
            edge_enc: (B, 3K) = (B, 96) positional encodings
        """
        u_src = self.eigenvectors[src_idx]   # (B, K)
        u_dst = self.eigenvectors[dst_idx]   # (B, K)
        u_hadamard = u_src * u_dst           # (B, K) element-wise product

        # Concatenate: [src_spectral || dst_spectral || hadamard_product]
        edge_enc = torch.cat([u_src, u_dst, u_hadamard], dim=1)  # (B, 3K)
        return edge_enc

    def update_for_new_topology(self, new_eigvec: np.ndarray, new_num_nodes: int):
        """
        Update encoder for changed topology (Section 5.7: topology adaptation).

        Args:
            new_eigvec: (new_num_nodes, K) eigenvectors for new topology
            new_num_nodes: Number of nodes in new topology
        """
        self.num_nodes = new_num_nodes
        # Resize buffer
        device = self.eigenvectors.device
        self.eigenvectors = torch.zeros(
            new_num_nodes, self.spectral_k,
            dtype=torch.float32, device=device
        )
        self.set_eigenvectors(new_eigvec)
