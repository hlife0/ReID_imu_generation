"""Repo-native geometry helpers for the IMU generation protocol.

Faithful copies of the small pure-numpy routines the generators need, so the
generation layer no longer sys.path-hacks into the external ``data_generation``
checkout for basic math. Formulas and float32 conventions are byte-identical to
``data_generation/src/utils/math3d.py`` and ``.../smplx_ops/sensors.py`` — see
``tests/test_geom_equivalence`` for the equivalence guard.

Quaternions are (w, x, y, z), sensor-to-world convention.
"""

from __future__ import annotations

import math

import numpy as np

# Joint indices for the corpus motion layout (``data_generation_pipeline_v1``,
# 21 joints). Only the arm chains used for wrist sensor placement are named.
_JOINT_INDEX_V1 = {
    "left_elbow": 6, "left_wrist": 7, "left_hand": 8,
    "right_elbow": 10, "right_wrist": 11, "right_hand": 12,
}

_PLACEMENT_JOINTS = {
    "wrist_right": ("right_elbow", "right_wrist", "right_hand"),
    "wrist_left": ("left_elbow", "left_wrist", "left_hand"),
}


def normalize(vector: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / np.clip(norms, eps, None)


def quaternion_normalize(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion_normalize(quaternion)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    m = matrix.astype(np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        diag = np.argmax(np.diag(m))
        if diag == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif diag == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    return quaternion_normalize(np.array([qw, qx, qy, qz], dtype=np.float32))


def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion_normalize(quaternion)
    return np.array([qw, -qx, -qy, -qz], dtype=np.float32)


def rotate_world_to_local(vector: np.ndarray, sensor_to_world_quat: np.ndarray) -> np.ndarray:
    return quaternion_to_matrix(quaternion_inverse(sensor_to_world_quat)) @ vector


def second_difference(values: np.ndarray, dt: float) -> np.ndarray:
    second = np.zeros_like(values, dtype=np.float32)
    second[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / (dt * dt)
    second[0] = second[1]
    second[-1] = second[-2]
    return second


def rotation_matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    angle = math.acos(max(min((np.trace(matrix) - 1.0) * 0.5, 1.0), -1.0))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    denom = 2.0 * math.sin(angle)
    axis = np.array(
        [
            (matrix[2, 1] - matrix[1, 2]) / denom,
            (matrix[0, 2] - matrix[2, 0]) / denom,
            (matrix[1, 0] - matrix[0, 1]) / denom,
        ],
        dtype=np.float32,
    )
    return axis * angle


def build_frame(x_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    x_axis = normalize(x_axis.astype(np.float32))
    z_axis = normalize(np.cross(x_axis, y_hint.astype(np.float32)))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def compute_sensor_trajectory(
    joints: np.ndarray,
    joint_layout: str,
    placement: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sensor world position (T, 3) and orientation quat_wxyz (T, 4).

    The wrist joint gives position; orientation is the frame built from the
    hand-minus-wrist forward axis and the wrist-minus-elbow up hint. Mirrors
    ``data_generation/src/smplx_ops/sensors.compute_sensor_trajectory``.
    """
    if joint_layout != "data_generation_pipeline_v1":
        raise ValueError(
            f"unsupported joint_layout {joint_layout!r}; expected "
            "'data_generation_pipeline_v1' (extend _JOINT_INDEX_V1 to add layouts)"
        )
    if placement not in _PLACEMENT_JOINTS:
        raise ValueError(f"unsupported placement {placement!r}, expected one of {sorted(_PLACEMENT_JOINTS)}")
    elbow_name, wrist_name, hand_name = _PLACEMENT_JOINTS[placement]
    elbow_idx = _JOINT_INDEX_V1[elbow_name]
    wrist_idx = _JOINT_INDEX_V1[wrist_name]
    hand_idx = _JOINT_INDEX_V1[hand_name]

    positions = joints[:, wrist_idx].copy()
    quaternions = np.zeros((joints.shape[0], 4), dtype=np.float32)
    for frame_idx in range(joints.shape[0]):
        wrist = joints[frame_idx, wrist_idx]
        hand = joints[frame_idx, hand_idx]
        elbow = joints[frame_idx, elbow_idx]
        frame = build_frame(hand - wrist, wrist - elbow)
        quaternions[frame_idx] = quaternion_from_matrix(frame)
    return positions, quaternions
