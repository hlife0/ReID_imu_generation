"""L1 embedding adapters.

``stats_v1``: a deterministic handcrafted feature encoder — per channel:
mean, std, min, max, RMS of the first difference, and four log-scaled rFFT
band energies (DC excluded). 9 features x C channels. Deliberately frozen and
model-free so L1 rankings are reproducible before the probe exists; the
probe's IMU tower can be added as a second encoder.
"""

from __future__ import annotations

import numpy as np

ENCODERS = ("stats_v1",)


def _stats_v1(imu_windows: np.ndarray) -> np.ndarray:
    """(N, T, C) -> (N, 9*C) deterministic features."""
    x = np.asarray(imu_windows, dtype=np.float64)
    n, t, c = x.shape

    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    diff_rms = np.sqrt((np.diff(x, axis=1) ** 2).mean(axis=1))

    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2   # (N, T//2+1, C)
    bins = spec.shape[1]
    edges = np.linspace(1, bins, 5, dtype=int)   # 4 bands, DC excluded
    bands = [np.log1p(spec[:, edges[i]:edges[i + 1]].sum(axis=1)) for i in range(4)]

    feats = np.concatenate([mean, std, mn, mx, diff_rms] + bands, axis=1)
    return feats.astype(np.float64)


def embed_windows(imu_windows: np.ndarray, encoder_id: str = "stats_v1") -> np.ndarray:
    if encoder_id == "stats_v1":
        return _stats_v1(imu_windows)
    raise ValueError(f"unknown encoder {encoder_id!r}, available: {ENCODERS}")
