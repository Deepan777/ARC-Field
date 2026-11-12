"""
GraphINR-IDS Full Model
================================================================
Corresponds to: Section 5.1 (System Architecture Overview)

Integrates all five modules:
  1. SDN Flow Telemetry Collector → handled by preprocessing
  2. Spectral Graph Encoder → SpectralGraphEncoder
  3. Temporal-Flow Coordinate Generator → TemporalEncoder
  4. SIREN-based Implicit Neural Network → SIRENNetwork + GCM + HFD
  5. Anomaly Scoring and Alert Engine → AnomalyScorer

The full coordinate is:
  x = [PE(v_i, v_j) || TE(t) || norm(m_f)] ∈ ℝ^151
================================================================
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

from .spectral_encoder import SpectralGraphEncoder
from .temporal_encoder import TemporalEncoder
from .siren import SIRENNetwork, SIRENLayer
from .gcm import GraphConditionedModulation
from .hfd import HierarchicalFrequencyDecomposition


class GraphINRIDS(nn.Module):
    """
    GraphINR-IDS: Graph-Topology-Aware Implicit Neural Representation
    for Zero-Day Intrusion Detection in Encrypted Network Traffic.

    This model learns a continuous neural field over the network topology
    that maps (spectral_graph_pos, time, flow_metadata) → normal_state.
    """

    def __init__(self, cfg: dict, num_nodes: int):
        """
        Args:
            cfg: Full configuration dict
            num_nodes: Number of nodes in the network graph
        """
        super().__init__()
        self.cfg = cfg

        # ── Module 2: Spectral Graph Encoder (Section 5.3) ──
        self.spectral_encoder = SpectralGraphEncoder(
            num_nodes=num_nodes,
            spectral_k=cfg["graph"]["spectral_k"],  # 32
        )

        # ── Module 3: Temporal Encoder (Section 5.4) ──
        self.temporal_encoder = TemporalEncoder(
            num_frequencies=cfg["temporal"]["num_frequencies"],  # 16
            freq_min=cfg["temporal"]["freq_min"],                # 1.0
            freq_max=cfg["temporal"]["freq_max"],                # 10000.0
        )

        # Dimensions:
        # graph_enc: 3 * spectral_k = 96
        # temporal_enc: 2 * num_frequencies = 32
        # flow_features: 23
        # Total input: 96 + 32 + 23 = 151
        siren_cfg = cfg["siren"]
        input_dim = siren_cfg["input_dim"]   # 151
        hidden_dim = siren_cfg["hidden_dim"]  # 512
        hidden_layers = siren_cfg["hidden_layers"]  # 6
        output_dim = siren_cfg["output_dim"]  # 23

        # ── Module 4: SIREN-based INR (Section 5.5) ──
        hfd_cfg = cfg["hfd"]
        gcm_cfg = cfg["gcm"]

        if hfd_cfg["enabled"]:
            # Use Hierarchical Frequency Decomposition (dual-branch SIREN)
            self.siren = HierarchicalFrequencyDecomposition(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=output_dim,
                low_omega=hfd_cfg["low_omega"],    # 10.0
                high_omega=hfd_cfg["high_omega"],  # 60.0
                attention_dim=hfd_cfg["attention_dim"],  # 64
                gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
            )
        else:
            # Use single SIREN network
            self.siren = SIRENNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=output_dim,
                omega_0=siren_cfg["omega_0"],  # 30.0
                gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
                checkpoint_every=cfg["training"]["checkpoint_layers"],
            )

        # ── Graph-Conditioned Modulation (Section 5.5, Innovation 1) ──
        self.use_gcm = gcm_cfg["enabled"]
        if self.use_gcm:
            self.gcm = GraphConditionedModulation(
                spectral_k=cfg["graph"]["spectral_k"],
                gnn_hidden=gcm_cfg["gnn_hidden_dim"],
                gnn_layers=gcm_cfg["gnn_layers"],
                siren_hidden=hidden_dim,
                num_modulated_layers=hidden_layers,
            )
            # Edge index buffer for GCM (set during setup)
            self.register_buffer("edge_index", torch.zeros(2, 1, dtype=torch.long))

        # Cache for GCM node features (recomputed when topology changes)
        self._gcm_node_features: Optional[torch.Tensor] = None

    def set_graph_data(self, eigenvectors: np.ndarray,
                        edge_index: Optional[np.ndarray] = None):
        """
        Initialize graph data for the model.

        Args:
            eigenvectors: (N, K) Laplacian eigenvectors
            edge_index: (2, E) graph edge indices (for GCM)
        """
        self.spectral_encoder.set_eigenvectors(eigenvectors)
        if edge_index is not None and self.use_gcm:
            self.edge_index = torch.from_numpy(edge_index).long().to(
                self.spectral_encoder.eigenvectors.device
            )
        self._gcm_node_features = None  # Reset cache

    def _build_coordinate(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build the 151-dimensional input coordinate from batch data.

        x = [PE(v_i, v_j) || TE(t) || norm(m_f)] ∈ ℝ^151

        Per Section 5.4.
        """
        # Spectral graph positional encoding: (B, 96)
        graph_enc = self.spectral_encoder(batch["src_idx"], batch["dst_idx"])

        # Temporal encoding: (B, 32)
        temporal_enc = self.temporal_encoder(batch["timestamp"])

        # Flow features (already normalized): (B, 23)
        flow_feat = batch["features"]

        # Concatenate: (B, 151)
        coordinate = torch.cat([graph_enc, temporal_enc, flow_feat], dim=1)
        return coordinate

    def _compute_gcm_features(self):
        """Compute GCM node features from graph.
        During training, do NOT cache them, because they depend on trainable
        GCM parameters and would otherwise reuse the old autograd graph across batches.
        During evaluation, caching is safe and faster.
        """
        if not self.use_gcm:
            return None

        if self.training:
            return self.gcm.compute_node_features(
                self.spectral_encoder.eigenvectors,
                self.edge_index
            )

        if self._gcm_node_features is None:
            self._gcm_node_features = self.gcm.compute_node_features(
                self.spectral_encoder.eigenvectors,
                self.edge_index
            )
        return self._gcm_node_features

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of GraphINR-IDS.

        Args:
            batch: Dict with keys: features, src_idx, dst_idx, timestamp

        Returns:
            Dict with:
              - "prediction": (B, 23) predicted normal state
              - "target": (B, 23) observed state (= input features)
              - "coordinate": (B, 151) input coordinate
        """
        # Build 151-dim coordinate
        coordinate = self._build_coordinate(batch)

        modulation_params = None
        if self.use_gcm:
            node_features = self._compute_gcm_features()
            if node_features is not None:
                modulation_params = self.gcm.get_modulation_params(
                    node_features, batch["src_idx"], batch["dst_idx"]
                )

        # SIREN forward pass (with HFD if enabled)
        if isinstance(self.siren, HierarchicalFrequencyDecomposition):
            prediction = self.siren(
                coordinate,
                low_modulation_params=modulation_params,
                high_modulation_params=modulation_params,
            )
        else:
            prediction = self.siren(
                coordinate,
                modulation_params=modulation_params,
            )

        return {
            "prediction": prediction,
            "target": batch["features"],  # Reconstruction target = input features
            "coordinate": coordinate,
        }

    def get_anomaly_scores(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute raw L2 reconstruction error for anomaly detection.

        Args:
            batch: Dict with flow data

        Returns:
            scores: (B,) L2 reconstruction errors
        """
        with torch.no_grad():
            output = self.forward(batch)
            scores = torch.norm(
                output["prediction"] - output["target"],
                p=2, dim=1
            )
        return scores

    def invalidate_gcm_cache(self):
        """Call when topology changes to recompute GCM features."""
        self._gcm_node_features = None
