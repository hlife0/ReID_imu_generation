"""GlobalPose_origin generator adapter — file-level contract (M1, implemented).

    generate.py --motion <motion.npz> --sensor R_LowArm \\
                --config <cfg.json> --seed N --out <dir>

Writes ``synth_globalpose_<cfg8>.npz`` + ``synth_globalpose_<cfg8>.manifest.json``
into --out. Ports the synthesis core of
scripts/totalcapture_test/GlobalPose_origin/_run_pipeline_impl.py verbatim,
with the config ``switches`` block controlling each realism stage:

    installation_error_rbs              random sensor mounting rotation (RBS)
    position_random_walk                random-walk + static offset on sensor position
    orientation_random_walk             random-walk + static offset on sensor orientation
    sensor_noise                        gaussian + random-walk noise on acc/gyro/mag
    orientation_from_noisy_integration  orientation estimated by integrating noisy
                                        angular velocity (+ per-frame rotation noise)
                                        instead of using the true orientation

With every switch ON and the same seed this reproduces the maintained
pipeline's output exactly (identical random-draw order). Runs in the external
generator venv (torch required).
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
import torch

from globalpose_origin_adapter import enforce_quaternion_continuity, matrix_to_quaternion_wxyz
from sim2real.gen_common import load_config, load_motion_and_trajectory, write_synth_stream
from src.utils.math3d import quaternion_to_matrix

GENERATOR = "globalpose"

SWITCH_NAMES = (
    "installation_error_rbs",
    "position_random_walk",
    "orientation_random_walk",
    "sensor_noise",
    "orientation_from_noisy_integration",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motion", required=True)
    parser.add_argument("--sensor", default="R_LowArm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_globalpose_imu_simulator():
    simulation_path = REPO_ROOT / "third-party" / "GlobalPose" / "articulate" / "utils" / "imu" / "simulation.py"
    spec = importlib.util.spec_from_file_location("globalpose_imu_simulation", simulation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load IMUSimulator from {simulation_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IMUSimulator


def walking_noise(shape, std: float, device) -> torch.Tensor:
    return torch.cumsum(torch.normal(torch.zeros(shape, device=device), std), dim=0)


def random_rotation_matrices(count: int, device) -> torch.Tensor:
    quat = torch.randn(count, 4, device=device)
    quat = quat / quat.norm(dim=1, keepdim=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack(
        [
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
        ],
        dim=1,
    ).unsqueeze(0)


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / angle
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    ca = torch.cos(angle[..., 0])
    sa = torch.sin(angle[..., 0])
    C = 1 - ca
    return torch.stack(
        [
            torch.stack((ca + x * x * C, x * y * C - z * sa, x * z * C + y * sa), dim=-1),
            torch.stack((y * x * C + z * sa, ca + y * y * C, y * z * C - x * sa), dim=-1),
            torch.stack((z * x * C - y * sa, z * y * C + x * sa, ca + z * z * C), dim=-1),
        ],
        dim=-2,
    )


def synthesize_globalpose(
    positions: np.ndarray,
    rotations: np.ndarray,
    fps: float,
    seed: int,
    switches: dict,
) -> dict:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = torch.tensor(positions, dtype=torch.float32, device=device)
    r = torch.tensor(rotations, dtype=torch.float32, device=device)
    num_frames, num_sensors = p.shape[:2]
    k = float(np.sqrt(np.pi / 8.0))

    if switches["installation_error_rbs"]:
        rbs = random_rotation_matrices(num_sensors, device=device)
    else:
        rbs = torch.eye(3, device=device).expand(1, num_sensors, 3, 3).contiguous()

    if switches["position_random_walk"]:
        dp = walking_noise((num_frames, num_sensors, 3), std=1e-3 * k * np.sqrt(1 / fps), device=device)
        dp = dp + torch.randn(num_sensors, 3, device=device) * (1e-2 * k)
    else:
        dp = torch.zeros(num_frames, num_sensors, 3, device=device)

    if switches["orientation_random_walk"]:
        dw = walking_noise((num_frames, num_sensors, 3), std=1e-2 * k * np.sqrt(1 / fps), device=device)
        dw = dw + torch.randn(num_sensors, 3, device=device) * (1e-1 * k)
    else:
        dw = torch.zeros(num_frames, num_sensors, 3, device=device)
    dr = axis_angle_to_matrix(dw)

    p_imu = p + r.matmul(dp.unsqueeze(-1)).squeeze(-1)
    r_imu = r.matmul(rbs).matmul(dr)

    IMUSimulator = load_globalpose_imu_simulator()
    simulator = IMUSimulator()
    simulator.set_trajectory(p_imu.squeeze(1), r_imu.squeeze(1), fps=fps)
    a_sensor = simulator.get_acceleration(gW=(0.0, -9.8, 0.0)).unsqueeze(1)
    w_sensor = simulator.get_angular_velocity().unsqueeze(1)
    m_sensor = simulator.get_magnetic_field(mW=(1.0, 0.0, 0.0)).unsqueeze(1)

    if switches["sensor_noise"]:
        a_sensor = torch.normal(a_sensor, std=5e-2) + walking_noise(a_sensor.shape, std=1e-4 * np.sqrt(1 / fps), device=device)
        w_sensor = torch.normal(w_sensor, std=5e-3) + walking_noise(w_sensor.shape, std=1e-5 * np.sqrt(1 / fps), device=device)
        m_sensor = torch.normal(m_sensor, std=5e-3) + walking_noise(m_sensor.shape, std=1e-5 * np.sqrt(1 / fps), device=device)

    if switches["orientation_from_noisy_integration"]:
        delta_rot = axis_angle_to_matrix(w_sensor / fps)
        r_est = torch.empty_like(r_imu)
        r_est[0] = r_imu[0]
        for frame_index in range(1, num_frames):
            r_est[frame_index] = r_est[frame_index - 1].matmul(delta_rot[frame_index])
        n_rot = axis_angle_to_matrix(torch.randn(num_frames, num_sensors, 3, device=device) * (0.1 * k))
        r_est = r_est.matmul(n_rot)
    else:
        r_est = r_imu.clone()

    gravity = torch.tensor([0.0, -9.8, 0.0], dtype=torch.float32, device=device)
    a_model = r_est.matmul(a_sensor.unsqueeze(-1)).squeeze(-1) + gravity
    w_model = r_est.matmul(w_sensor.unsqueeze(-1)).squeeze(-1)
    r_body = r_est.matmul(rbs.transpose(-1, -2))
    m_model = r_body.transpose(-1, -2).matmul(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 3, 1)
    ).squeeze(-1)

    quat = np.stack(
        [
            [matrix_to_quaternion_wxyz(r_body[frame_index, sensor_index].cpu().numpy()) for sensor_index in range(num_sensors)]
            for frame_index in range(num_frames)
        ],
        axis=0,
    )
    return {
        "quat": quat,
        "acc": a_model.cpu().numpy().astype(np.float64),
        "gyro": w_model.cpu().numpy().astype(np.float64),
        "mag": m_model.cpu().numpy().astype(np.float64),
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if config.get("generator") != GENERATOR:
        raise SystemExit(f"config generator {config.get('generator')!r} != {GENERATOR!r}")
    switches = {name: bool(config["switches"][name]) for name in SWITCH_NAMES}

    motion, positions, quat_wxyz = load_motion_and_trajectory(Path(args.motion), args.sensor)
    rotations = np.stack([quaternion_to_matrix(q) for q in quat_wxyz], axis=0).astype(np.float32)[:, None, :, :]

    generated = synthesize_globalpose(
        positions=positions[:, None, :].astype(np.float32),
        rotations=rotations,
        fps=motion.fps,
        seed=args.seed,
        switches=switches,
    )
    quat = enforce_quaternion_continuity(generated["quat"][:, 0])

    result = write_synth_stream(
        Path(args.out),
        generator=GENERATOR,
        config=config,
        config_path=Path(args.config),
        motion_path=Path(args.motion),
        seed=args.seed,
        sensor=args.sensor,
        fps=motion.fps,
        quat=quat,
        acc=generated["acc"][:, 0],
        gyro=generated["gyro"][:, 0],
        mag=generated["mag"][:, 0],
        extra_meta={"switches": switches},
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
