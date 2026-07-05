"""naive_kinematics generator adapter — file-level contract (M1, implemented).

Same CLI and output contract as generators/globalpose/generate.py. The
deliberately-simple baseline: pure finite-difference kinematics via
data_generation's ``src.imu.synthesize.synthesize_imu_from_world_trajectory``
with gravity [0, -9.81, 0] (pipeline world frame), no realism modules — its
TSTR score anchors the bottom of the generator ranking. Runs in the external
generator venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
# data_generation's `src` is a regular package and shadows the repo's `src`
# namespace; expose repo code as top-level `sim2real` via REPO_ROOT/src.
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
DATA_GENERATION_ROOT = Path("/home/hrli/data_generation")
if str(DATA_GENERATION_ROOT) not in sys.path:
    sys.path.append(str(DATA_GENERATION_ROOT))

import numpy as np

from globalpose_origin_adapter import enforce_quaternion_continuity
from sim2real.gen_common import load_config, load_motion_and_trajectory, write_synth_stream

GENERATOR = "naive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motion", required=True)
    parser.add_argument("--sensor", default="R_LowArm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if config.get("generator") != GENERATOR:
        raise SystemExit(f"config generator {config.get('generator')!r} != {GENERATOR!r}")

    from src.imu.synthesize import synthesize_imu_from_world_trajectory  # data_generation

    np.random.seed(args.seed)
    motion, positions, quat_wxyz = load_motion_and_trajectory(Path(args.motion), args.sensor)
    fps = int(round(motion.fps))

    imu = synthesize_imu_from_world_trajectory(
        positions=positions,
        quaternions=quat_wxyz,
        fps=fps,
        gravity_world=np.array([0.0, -9.81, 0.0], dtype=np.float32),
    )

    quat = enforce_quaternion_continuity(imu["quat"].astype(np.float64))
    frames = quat.shape[0]
    mag = np.zeros((frames, 3), dtype=np.float64)

    result = write_synth_stream(
        Path(args.out),
        generator=GENERATOR,
        config=config,
        config_path=Path(args.config),
        motion_path=Path(args.motion),
        seed=args.seed,
        sensor=args.sensor,
        fps=float(fps),
        quat=quat,
        acc=imu["accel"],
        gyro=imu["gyro"],
        mag=mag,
        extra_meta={},
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
