"""Per-sequence IMU<->motion temporal alignment.

The corpus pairs a real Xsens stream (lxhong raw) with an AMASS-derived
motion stream of a *different* length (observed offsets: mostly 0-2 frames,
S5_freestyle3 is 44). The original code assumed a global convention —
sim2real tail-aligned (`[-n:]`), sim2pipe head-aligned (`[:n]`) — and a lag
scan on S5_freestyle3 shows neither is right in general (head-align there
gives acc-magnitude corr 0.02 vs 0.94 at lag=44; on S1_freestyle1 head wins).

Convention: ``imu_motion_lag = k`` means ``real[i] <-> motion[i + k]``.
Synthetic streams are generated ON the motion timebase, so their lag against
motion is always 0 (and their lag against real is the same ``k``).

Lag is estimated once per sequence by ``scripts/sim2real/01c_estimate_alignment.py``
(sliding-Pearson scan of acc magnitudes, using the deterministic naive synth
stream as the motion-timebase bridge) and stored in the sequence ``meta.json``
under ``alignment``. Consumers: sim2real windows/gate and sim2pipe export.

Pure numpy; runs under the repo python.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ALIGNMENT_METHOD = "naive_bridge_lagscan_v1"
DEFAULT_MAX_LAG = 60
MIN_BRIDGE_CORR = 0.3  # below this the lag estimate itself is suspect


def estimate_lag(imu_mag: np.ndarray, motion_mag: np.ndarray,
                 max_lag: int = DEFAULT_MAX_LAG) -> tuple[int, float]:
    """Best ``k`` (imu[i] <-> motion[i+k]) by sliding Pearson correlation.

    ``imu_mag`` / ``motion_mag`` are 1-D signals on the two timebases (acc
    magnitudes in practice). Returns (lag, correlation at that lag).
    """
    imu_mag = np.asarray(imu_mag, dtype=np.float64)
    motion_mag = np.asarray(motion_mag, dtype=np.float64)
    if imu_mag.ndim != 1 or motion_mag.ndim != 1:
        raise ValueError("estimate_lag expects 1-D magnitude signals")

    best_lag, best_corr = 0, -np.inf
    for k in range(-max_lag, max_lag + 1):
        sl_i, sl_m = aligned_slices(len(imu_mag), len(motion_mag), k)
        n = sl_i.stop - sl_i.start
        if n < 32:  # too little overlap for a meaningful correlation
            continue
        a, b = imu_mag[sl_i], motion_mag[sl_m]
        if a.std() == 0.0 or b.std() == 0.0:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_lag, best_corr = k, corr
    if not np.isfinite(best_corr):
        raise ValueError("no valid lag found (signals too short or constant)")
    return best_lag, best_corr


def aligned_slices(len_imu: int, len_motion: int, lag: int) -> tuple[slice, slice]:
    """Slices realizing ``imu[i] <-> motion[i + lag]`` with maximal overlap.

    Returns (imu_slice, motion_slice) of equal length. Raises if the overlap
    is empty.
    """
    start_i = max(0, -int(lag))
    start_m = max(0, int(lag))
    n = min(len_imu - start_i, len_motion - start_m)
    if n < 1:
        raise ValueError(
            f"empty overlap: len_imu={len_imu} len_motion={len_motion} lag={lag}"
        )
    return slice(start_i, start_i + n), slice(start_m, start_m + n)


def alignment_record(lag: int, corr: float, bridge_stream: str,
                     max_lag: int = DEFAULT_MAX_LAG) -> dict:
    """The dict stored under ``meta.json['alignment']``."""
    return {
        "method": ALIGNMENT_METHOD,
        "imu_motion_lag": int(lag),
        "acc_mag_pearson": round(float(corr), 4),
        "bridge_stream": bridge_stream,
        "search_range": int(max_lag),
    }


def lag_from_meta(meta: dict) -> int | None:
    """Extract the estimated lag from a sequence meta dict; None if absent.

    Pre-01c corpora carry the legacy string ``meta['alignment'] == 'tail'``
    (or nothing) — those return None and the caller decides whether to fall
    back or fail loudly.
    """
    alignment = meta.get("alignment")
    if isinstance(alignment, dict) and "imu_motion_lag" in alignment:
        return int(alignment["imu_motion_lag"])
    return None
