"""
GraphINR-IDS Graph-Conditioned Modulation (GCM)
================================================================
Corresponds to: Section 5.5, Innovation 1

"A Graph-Conditioned Modulation (GCM) mechanism that modulates
intermediate SIREN activations using topology-specific scale and
shift parameters learned by a lightweight GNN operating on the
current graph snapshot."

This implements FiLM-style (Feature-wise Linear Modulation)
conditioning: h_modulated = γ * h + β, where γ and β are
produced by a 2-layer GNN from the graph structure.
================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphAttentionLayer(nn.Module):
    """Lightweight graph attention layer for GCM's GNN backbone."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) edge indices

        Returns:
            out: (N, out_dim) updated node features
        """
        h = self.W(x)  # (N, out_dim)
        src, dst = edge_index[0], edge_index[1]

        # Attention scores
        e_src = self.a_src(h)[src]  # (E, 1)
        e_dst = self.a_dst(h)[dst]  # (E, 1)
        e = self.leaky_relu(e_src + e_dst).squeeze(-1)  # (E,)

        # Softmax over neighbors
        alpha = torch.zeros(x.size(0), device=x.device)
        # Simple mean aggregation (stable, avoids scatter issues)
        # For each destination node, aggregate source features
        out = torch.zeros_like(h)
        counts = torch.zeros(x.size(0), 1, device=x.device)

        out.index_add_(0, dst, h[src] * torch.sigmoid(e).unsqueeze(1))
        counts.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))
        counts = counts.clamp(min=1.0)
        out = out / counts

        return F.elu(out + h)  # Residual connection


class GraphConditionedModulation(nn.Module):
    """
    GCM: Generates per-flow FiLM parameters (scale γ, shift β) from
    the network graph topology using a lightweight GNN.

    The GNN processes node features derived from spectral embeddings
    and produces modulation parameters that condition SIREN activations
    based on the local graph neighborhood structure.
    """

    def __init__(self, spectral_k: int = 32, gnn_hidden: int = 128,
                 gnn_layers: int = 2, siren_hidden: int = 512,
                 num_modulated_layers: int = 6):
        """
        Args:
            spectral_k: K eigenvector dimensions per node
            gnn_hidden: GNN hidden dimension (128)
            gnn_layers: Number of GNN layers (2)
            siren_hidden: SIREN layer width to modulate (512)
            num_modulated_layers: Number of SIREN layers to modulate
        """
        super().__init__()
        self.num_modulated_layers = num_modulated_layers

        # GNN backbone: 2-layer graph attention
        self.gnn_layers_list = nn.ModuleList()
        self.gnn_layers_list.append(GraphAttentionLayer(spectral_k, gnn_hidden))
        for _ in range(gnn_layers - 1):
            self.gnn_layers_list.append(GraphAttentionLayer(gnn_hidden, gnn_hidden))

        # Per-layer FiLM parameter generators
        # Each produces (γ, β) ∈ ℝ^(512) from concatenated src+dst GNN features
        self.film_generators = nn.ModuleList()
        for _ in range(num_modulated_layers):
            self.film_generators.append(
                nn.Sequential(
                    nn.Linear(2 * gnn_hidden, siren_hidden),
                    nn.ReLU(),
                    nn.Linear(siren_hidden, 2 * siren_hidden),  # γ and β
                )
            )

    def compute_node_features(self, spectral_embeddings: torch.Tensor,
                               edge_index: torch.Tensor) -> torch.Tensor:
        """
        Run GNN on spectral embeddings to get topology-aware node features.

        Args:
            spectral_embeddings: (N, K) Laplacian eigenvectors
            edge_index: (2, E) graph edges

        Returns:
            node_features: (N, gnn_hidden)
        """
        h = spectral_embeddings
        for gnn_layer in self.gnn_layers_list:
            h = gnn_layer(h, edge_index)
        return h

    def get_modulation_params(self, node_features: torch.Tensor,
                               src_idx: torch.Tensor,
                               dst_idx: torch.Tensor):
        """
        Generate per-flow, per-layer FiLM parameters.

        Args:
            node_features: (N, gnn_hidden) from compute_node_features
            src_idx: (B,) source node indices
            dst_idx: (B,) dest node indices

        Returns:
            list of (gamma, beta) tuples, one per SIREN layer
            Each gamma, beta: (B, siren_hidden)
        """
        # Concatenate source and dest node features
        src_feat = node_features[src_idx]  # (B, gnn_hidden)
        dst_feat = node_features[dst_idx]  # (B, gnn_hidden)
        edge_feat = torch.cat([src_feat, dst_feat], dim=1)  # (B, 2*gnn_hidden)

        modulation_params = []
        for film_gen in self.film_generators:
            params = film_gen(edge_feat)  # (B, 2 * siren_hidden)
            gamma, beta = params.chunk(2, dim=1)  # Each (B, siren_hidden)
            # Initialize gamma near 1 and beta near 0 for stability
            gamma = 1.0 + 0.1 * torch.tanh(gamma)
            beta = 0.1 * torch.tanh(beta)
            modulation_params.append((gamma, beta))

        return modulation_params

    @staticmethod
    def apply_modulation(h: torch.Tensor, gamma: torch.Tensor,
                         beta: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation: h_mod = γ ⊙ h + β

        Args:
            h: (B, D) SIREN activation
            gamma: (B, D) scale
            beta: (B, D) shift

        Returns:
            h_modulated: (B, D)
        """
        return gamma * h + beta
