"""Shared-feature-encoder + set-aggregator model (research plan section 10).

    values x_j (window)  --TemporalEncoder-->  h_j
    (h_j, e_j)           --FeatureEncoder F_theta-->  z_j        (SHARED across all features)
    {z_j}                --AttentionAggregator A-->  z           (permutation-invariant)
    z                    --Head G-->  logit

The same F_theta processes every feature, so knowledge is keyed on the semantic embedding e_j rather
than on column position; that is what lets an unseen feature (new e_j) reuse learned behaviour.
Conditions differ ONLY in how e_j is supplied (see experiment/conditions.py):
  C0 zero e_j; C2 trainable nn.Embedding by feature index; C1/C3/C4/C5/C6 fixed precomputed e_j.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Shared 1D-conv encoder over a value window -> h_j. window==1 degrades to a scalar MLP."""
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):                      # x: (B, F, W)
        B, F, W = x.shape
        h = self.net(x.reshape(B * F, 1, W))   # (B*F, hidden, 1)
        return h.reshape(B, F, -1)             # (B, F, hidden)


class FeatureEncoder(nn.Module):
    """Shared F_theta: (h_j, e_j) -> z_j."""
    def __init__(self, h_dim: int, e_dim: int, z_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + e_dim, z_dim), nn.ReLU(),
            nn.Linear(z_dim, z_dim), nn.ReLU(),
        )
    def forward(self, h, e):                    # h:(B,F,h_dim) e:(B,F,e_dim)
        return self.net(torch.cat([h, e], dim=-1))


class AttentionAggregator(nn.Module):
    """Permutation-invariant attention pooling over features -> z. Respects a feature mask so
    held-out / variable feature counts work without positional identities."""
    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.score = nn.Linear(z_dim, 1)
    def forward(self, z, mask=None):           # z:(B,F,z_dim) mask:(B,F) 1=present
        s = self.score(z).squeeze(-1)          # (B,F)
        if mask is not None:
            s = s.masked_fill(mask == 0, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return (a * z).sum(dim=1)              # (B,z_dim)


class SFTModel(nn.Module):
    def __init__(self, e_dim: int, n_features_for_id: int = 0,
                 h_hidden: int = 32, z_dim: int = 64):
        super().__init__()
        self.temporal = TemporalEncoder(h_hidden)
        self.encoder = FeatureEncoder(h_hidden, e_dim, z_dim)
        self.aggregator = AttentionAggregator(z_dim)
        self.head = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, 1))
        # condition C2 only: trainable per-feature-index embedding (cannot embed unseen features)
        self.id_embedding = (nn.Embedding(n_features_for_id, e_dim)
                             if n_features_for_id > 0 else None)

    def forward(self, x, e=None, feature_idx=None, mask=None):
        h = self.temporal(x)
        if self.id_embedding is not None:
            e = self.id_embedding(feature_idx)          # (B,F,e_dim)
        z = self.encoder(h, e)
        return self.head(self.aggregator(z, mask)).squeeze(-1)   # (B,) logit
