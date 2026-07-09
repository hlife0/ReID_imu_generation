"""humogen generator — wraps HuMoGen's external synthesize_imu.

Faithful to the maintained HuMoGen_origin behaviour: xyzw quaternions in,
target_hz = motion fps, HuMoGen's default gravity, zero magnetic field, true
trajectory orientation on the quat channels. The HuMoGen synthesis module lives
outside this repo; its path is taken from ``params.synth_module`` in the config
(falling back to the maintained default). Runs in the external generator venv.

Protocol: see ``docs/imu_generation_protocol.md`` and ``sim2real.gen_common``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as np

from globalpose_origin_adapter import enforce_quaternion_continuity
from sim2real.gen_common import run_generator

GENERATOR = "humogen"
DEFAULT_SYNTH_MODULE = "/data/luoyizhang/HuMoGen/src/core/synthesize_imu.py"


def load_humogen_synthesize(module_path: Path):
    spec = importlib.util.spec_from_file_location("humogen_synthesize_imu", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HuMoGen synthesize module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthesize_imu


def synthesize(motion, positions, quat_wxyz, config, seed):
    params = config.get("params", {})
    np.random.seed(seed)
    fps = int(round(motion.fps))
    timestamps = np.arange(motion.num_frames, dtype=np.float32) / float(fps)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    synthesize_imu = load_humogen_synthesize(Path(params.get("synth_module", DEFAULT_SYNTH_MODULE)))
    imu = synthesize_imu(
        timestamps_in=timestamps,
        pos_world_in=positions.astype(np.float32),
        quat_world_in=quat_xyzw.astype(np.float32),
        target_hz=float(fps),
        add_noise=bool(params.get("add_noise", True)),
        accel_noise_std=float(params.get("accel_noise_std", 0.01)),
        gyro_noise_std=float(params.get("gyro_noise_std", 0.005)),
    )

    frames = len(imu["accel_x"])
    quat = enforce_quaternion_continuity(quat_wxyz.astype(np.float64))[:frames]
    acc = np.stack([imu["accel_x"], imu["accel_y"], imu["accel_z"]], axis=1)
    gyro = np.stack([imu["gyro_x"], imu["gyro_y"], imu["gyro_z"]], axis=1)
    mag = np.zeros((frames, 3), dtype=np.float64)
    return {"quat": quat, "acc": acc, "gyro": gyro, "mag": mag, "fps": float(fps),
            "extra_meta": {"params": params}}


if __name__ == "__main__":
    run_generator(GENERATOR, synthesize)
