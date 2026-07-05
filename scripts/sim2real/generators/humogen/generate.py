"""HuMoGen_origin generator adapter — file-level contract (M1, implemented).

Same CLI and output contract as generators/globalpose/generate.py. Faithful
to the maintained scripts/totalcapture_test/HuMoGen_origin behaviour:
xyzw quaternions, target_hz = motion fps, HuMoGen's default gravity vector,
zero magnetic field, true trajectory orientation as the quat channels.
Runs in the external generator venv.
"""

from __future__ import annotations

import argparse
import importlib.util
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

GENERATOR = "humogen"
DEFAULT_SYNTH_MODULE = "/data/luoyizhang/HuMoGen/src/core/synthesize_imu.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motion", required=True)
    parser.add_argument("--sensor", default="R_LowArm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--synth-module", default=DEFAULT_SYNTH_MODULE,
                        help="path to HuMoGen's synthesize_imu.py")
    return parser.parse_args()


def load_humogen_synthesize(module_path: Path):
    spec = importlib.util.spec_from_file_location("humogen_synthesize_imu", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load HuMoGen synthesize module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthesize_imu


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if config.get("generator") != GENERATOR:
        raise SystemExit(f"config generator {config.get('generator')!r} != {GENERATOR!r}")
    params = config.get("params", {})

    np.random.seed(args.seed)
    motion, positions, quat_wxyz = load_motion_and_trajectory(Path(args.motion), args.sensor)
    fps = int(round(motion.fps))
    timestamps = np.arange(motion.num_frames, dtype=np.float32) / float(fps)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    synthesize_imu = load_humogen_synthesize(Path(args.synth_module))
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
        acc=acc,
        gyro=gyro,
        mag=mag,
        extra_meta={"params": params},
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
