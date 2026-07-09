"""Generator harness + shared helpers for the IMU generation protocol.

The protocol (see ``docs/imu_generation_protocol.md``): a generator is a CLI
that reads one standardized ``MotionSequence`` (motion.npz) plus a JSON config
and writes one standardized 13-channel ``ImuSequence`` (synth npz + manifest).

``run_generator`` implements that contract once — argument parsing, config
load + generator-name check, motion+trajectory load, artifact write, and the
stdout JSON receipt — so each generator file contains only its synthesis core.
A generator conforms by construction if it goes through this harness.

Sensor trajectory and all basic math come from the repo-native ``geom`` module,
so no generator needs the external ``data_generation`` checkout for the shared
path (only generators that wrap external research code, e.g. humogen, still do).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from . import geom
from .contracts import (
    IMU_CHANNELS_13,
    ImuSequence,
    MotionSequence,
    config_hash,
    source_to_token,
    synth_source,
    write_manifest,
)

# Sensor name -> trajectory placement understood by ``geom``.
_PLACEMENTS = {"R_LowArm": "wrist_right", "L_LowArm": "wrist_left"}

# A synthesis core: (motion, positions (T,3), quat_wxyz (T,4), config, seed) ->
# dict with float arrays ``quat``/``acc``/``gyro``/``mag``, a ``fps`` float, and
# an ``extra_meta`` dict recorded on the ImuSequence.
SynthesizeFn = Callable[[MotionSequence, np.ndarray, np.ndarray, dict, int], dict]


def load_config(config_path: Path) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def load_motion_and_trajectory(motion_path: Path, sensor: str):
    """Load motion.npz and derive the sensor 6DoF trajectory (repo-native).

    Returns (motion, positions (T,3) float32, quat_wxyz (T,4) float32).
    """
    if sensor not in _PLACEMENTS:
        raise ValueError(f"unsupported sensor {sensor!r}, expected one of {sorted(_PLACEMENTS)}")
    motion = MotionSequence.load(Path(motion_path))
    positions, quat_wxyz = geom.compute_sensor_trajectory(
        motion.joints, motion.joint_layout, _PLACEMENTS[sensor]
    )
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
    ``synth_<generator>_<cfg8>.npz`` / ``.manifest.json`` (``source_to_token``
    format, matching window-shard naming).
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
    write_manifest(
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


def run_generator(generator: str, synthesize: SynthesizeFn) -> None:
    """Standard generator entry point: parse args, synthesize, write, report.

    ``synthesize`` returns a dict with ``quat``/``acc``/``gyro``/``mag`` arrays,
    a ``fps`` float, and (optionally) an ``extra_meta`` dict. Everything else —
    the CLI contract and the ImuSequence/manifest write — is handled here.
    """
    parser = argparse.ArgumentParser(description=f"{generator} IMU generator")
    parser.add_argument("--motion", required=True)
    parser.add_argument("--sensor", default="R_LowArm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if config.get("generator") != generator:
        raise SystemExit(f"config generator {config.get('generator')!r} != {generator!r}")

    motion, positions, quat_wxyz = load_motion_and_trajectory(Path(args.motion), args.sensor)
    result = synthesize(motion, positions, quat_wxyz, config, args.seed)

    receipt = write_synth_stream(
        Path(args.out),
        generator=generator,
        config=config,
        config_path=Path(args.config),
        motion_path=Path(args.motion),
        seed=args.seed,
        sensor=args.sensor,
        fps=result["fps"],
        quat=result["quat"],
        acc=result["acc"],
        gyro=result["gyro"],
        mag=result["mag"],
        extra_meta=result.get("extra_meta", {}),
    )
    print(json.dumps(receipt))
