"""naive generator — the self-contained reference implementation.

Pure finite-difference kinematics: double-difference the sensor world position
into specific force (minus gravity), rotate into the sensor frame for accel;
finite-difference the sensor orientation for gyro. Zero magnetometer. No realism
modules, no randomness — this anchors the bottom of the generator ranking and is
the canonical example of a protocol-conforming generator.

Depends only on numpy + repo code (``geom`` for math, ``globalpose_origin_adapter``
for quaternion continuity); no external ``data_generation`` checkout required.

Protocol: see ``docs/imu_generation_protocol.md`` and ``sim2real.gen_common``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as np

from globalpose_origin_adapter import enforce_quaternion_continuity
from sim2real import geom
from sim2real.gen_common import run_generator

GENERATOR = "naive"
GRAVITY_WORLD = np.array([0.0, -9.81, 0.0], dtype=np.float32)  # pipeline world frame


def synthesize(motion, positions, quat_wxyz, config, seed):
    fps = int(round(motion.fps))
    dt = 1.0 / float(fps)

    specific_force_world = geom.second_difference(positions.astype(np.float32), dt) - GRAVITY_WORLD
    accel = np.zeros_like(specific_force_world, dtype=np.float32)
    gyro = np.zeros_like(specific_force_world, dtype=np.float32)
    for idx in range(positions.shape[0]):
        accel[idx] = geom.rotate_world_to_local(specific_force_world[idx], quat_wxyz[idx])
        if idx == 0:
            continue
        prev_rot = geom.quaternion_to_matrix(quat_wxyz[idx - 1])
        curr_rot = geom.quaternion_to_matrix(quat_wxyz[idx])
        gyro[idx] = geom.rotation_matrix_to_axis_angle(prev_rot.T @ curr_rot) / dt

    quat = enforce_quaternion_continuity(quat_wxyz.astype(np.float64))
    mag = np.zeros((quat.shape[0], 3), dtype=np.float64)
    return {"quat": quat, "acc": accel, "gyro": gyro, "mag": mag, "fps": float(fps), "extra_meta": {}}


if __name__ == "__main__":
    run_generator(GENERATOR, synthesize)
