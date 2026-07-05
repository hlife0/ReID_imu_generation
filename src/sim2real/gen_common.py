"""Shared helpers for the generator adapters (run inside the external venv).

Adapters add REPO_ROOT (index 0) and DATA_GENERATION_ROOT (appended) to
sys.path before importing this module. ``src`` is a namespace package, so
``src.sim2real.*`` resolves from this repo while ``src.smplx_ops.*`` /
``src.imu.*`` / ``src.utils.*`` resolve from /home/hrli/data_generation —
the same trick the existing ``_run_pipeline_impl.py`` scripts rely on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import (
    IMU_CHANNELS_13,
    ImuSequence,
    MotionSequence,
    config_hash,
    source_to_token,
    synth_source,
    write_manifest,
)

_PLACEMENTS = {"R_LowArm": "wrist_right", "L_LowArm": "wrist_left"}


def load_config(config_path: Path) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def load_motion_and_trajectory(motion_path: Path, sensor: str):
    """Load motion.npz and derive the sensor 6DoF trajectory.

    Returns (motion, positions (T,3) float32, quat_wxyz (T,4) float32).
    """
    from src.smplx_ops.sensors import compute_sensor_trajectory  # data_generation

    if sensor not in _PLACEMENTS:
        raise ValueError(f"unsupported sensor {sensor!r}, expected one of {sorted(_PLACEMENTS)}")
    motion = MotionSequence.load(Path(motion_path))
    positions, quat_wxyz = compute_sensor_trajectory(motion.joints, placement=_PLACEMENTS[sensor])
    return motion, positions.astype(np.float32), quat_wxyz.astype(np.float32)


def write_synth_stream(
    out_dir: Path,
    *,
    generator: str,
    config: dict,
    config_path: Path,
    motion_path: Path,
    seed: int,
    sensor: str,
    fps: float,
    quat: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    mag: np.ndarray,
    extra_meta: dict | None = None,
) -> dict:
    """Assemble the canonical 13-channel ImuSequence and write npz + manifest.

    File names carry the stream identity:
    ``synth_<generator>_<cfg8>.npz`` / ``synth_<generator>_<cfg8>.manifest.json``
    (the ``source_to_token`` format, matching window-shard naming).
    """
    frames = min(len(quat), len(acc), len(gyro), len(mag))
    data = np.concatenate(
        [
            np.asarray(quat, dtype=np.float32)[:frames],
            np.asarray(acc, dtype=np.float32)[:frames],
            np.asarray(gyro, dtype=np.float32)[:frames],
            np.asarray(mag, dtype=np.float32)[:frames],
        ],
        axis=1,
    )
    if data.shape[1] != len(IMU_CHANNELS_13):
        raise ValueError(f"expected {len(IMU_CHANNELS_13)} channels, got {data.shape[1]}")

    source = synth_source(generator, config)
    token = source_to_token(source)
    imu = ImuSequence(
        data=data,
        channels=IMU_CHANNELS_13,
        fps=float(fps),
        source=source,
        sensor=sensor,
        meta={"generator": generator, "seed": seed, **(extra_meta or {})},
    )
    out_dir = Path(out_dir)
    npz_path = imu.save(out_dir / f"{token}.npz")
    manifest = write_manifest(
        out_dir / f"{token}.manifest.json",
        stage="generate",
        config=config,
        inputs={"motion": Path(motion_path), "config": Path(config_path)},
        seed=seed,
        extra={"generator": generator, "sensor": sensor, "fps": float(fps), "frames": frames},
    )
    return {
        "npz": str(npz_path),
        "manifest": str(out_dir / f"{token}.manifest.json"),
        "source": source,
        "config_hash": config_hash(config),
        "frames": frames,
    }
