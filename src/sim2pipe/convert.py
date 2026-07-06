"""Pure conversions from sim2real corpus streams to the main project's tensors.

Target contracts (verified 2026-07-06 against Rd-id-Project):

* ``src/datasets/totalcapture.py::quat_to_rotmat`` — quaternions are **wxyz**.
* ``convert_imu_to_48`` — 48-D layout is **rotation-first by slot**: columns
  0:36 hold four 9-D row-major rotation matrices, columns 36:48 hold four
  3-D accelerations, slot order [L_LowLeg, R_LowLeg, L_LowArm, R_LowArm].
  Single-sensor mode replicates one sensor into all four slots
  (``repeat_single_sensor=4``), which is what we emit here.
* ``normalize_skeleton`` — root-relative (joint 0) then divided by the
  norm of joint 8 (Spine3/thorax) minus joint 0, per frame.

The TotalCapture path in the main repo consumes vicon + IMU 1:1 at the
native 60 Hz with no resampling, so no resampling happens here either.
Everything in this module is pure numpy and unit-tested.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Sensor slot order in the main project's 48-D vector.
PIPE_SENSOR_ORDER = ("L_LowLeg", "R_LowLeg", "L_LowArm", "R_LowArm")

# Channel names required from an ImuSequence (sim2real 13-channel contract).
QUAT_CHANNELS = ("quat0", "quat1", "quat2", "quat3")  # wxyz, same as xsens raw
ACC_CHANNELS = ("acc_x", "acc_y", "acc_z")


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(T, 4) wxyz quaternions -> (T, 3, 3) rotation matrices.

    Mirrors Rd-id-Project ``quat_to_rotmat`` exactly (including no
    normalization of the input quaternion).
    """
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected (T, 4) quaternions, got {q.shape}")
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.zeros((q.shape[0], 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _column(data: np.ndarray, channels: Sequence[str], name: str) -> np.ndarray:
    try:
        idx = list(channels).index(name)
    except ValueError:
        raise KeyError(f"channel {name!r} not found in {tuple(channels)}") from None
    return data[:, idx]


def imu_to_48(data: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    """(T, C) ImuSequence data -> (T, 48) main-project vector.

    Extracts quat wxyz + acc by channel name (column order independent),
    converts quat to a 9-D rotation matrix and replicates the single
    sensor into all four slots, rotation-first layout.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected (T, C) data, got {data.shape}")
    quat = np.stack([_column(data, channels, c) for c in QUAT_CHANNELS], axis=1)
    acc = np.stack([_column(data, channels, c) for c in ACC_CHANNELS], axis=1)
    rot9 = quat_wxyz_to_rotmat(quat).reshape(data.shape[0], 9)

    out = np.zeros((data.shape[0], 48), dtype=np.float32)
    for i in range(len(PIPE_SENSOR_ORDER)):
        out[:, i * 9 : (i + 1) * 9] = rot9
        out[:, 36 + i * 3 : 36 + (i + 1) * 3] = acc
    return out


def normalize_skeleton_pipe(skel17: np.ndarray) -> np.ndarray:
    """(T, 17, 3) -> root-relative, scale-normalized. Mirrors main repo."""
    skel17 = np.asarray(skel17, dtype=np.float32)
    if skel17.ndim != 3 or skel17.shape[1] != 17 or skel17.shape[2] != 3:
        raise ValueError(f"expected (T, 17, 3), got {skel17.shape}")
    root = skel17[:, 0:1, :]
    skel = skel17 - root
    scale = np.linalg.norm(skel[:, 8, :] - skel[:, 0, :], axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    return (skel / scale[:, None, :]).astype(np.float32)
