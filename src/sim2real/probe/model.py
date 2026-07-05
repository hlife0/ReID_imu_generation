"""Probe architecture — M3 (implemented). Torch module; import only in the venv.

A deliberately small, FIXED two-tower model: the probe is a measurement
instrument, not the research object — its capacity never changes across
matrix cells.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """(B, T, F) -> (B, D) L2-normalized embedding."""

    def __init__(self, in_features: int, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))          # (B, 128, T)
        pooled = torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)
        return F.normalize(self.head(pooled), dim=1)


class MotionTower(Tower):
    """(B, T, J, 3) -> flatten joints -> Tower."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[:2]
        return super().forward(x.reshape(b, t, -1))


def build_probe(imu_channels: int, num_joints: int, embed_dim: int = 128):
    return Tower(imu_channels, embed_dim), MotionTower(num_joints * 3, embed_dim)
