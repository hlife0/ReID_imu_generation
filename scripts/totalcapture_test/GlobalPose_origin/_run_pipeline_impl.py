from __future__ import annotations

import argparse
import csv
import json
import sys
import importlib.util
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_GENERATION_ROOT = Path("/home/hrli/data_generation")
if str(DATA_GENERATION_ROOT) not in sys.path:
    sys.path.append(str(DATA_GENERATION_ROOT))

def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_repo_adapter = _load_module("repo_globalpose_origin_adapter", REPO_ROOT / "src" / "globalpose_origin_adapter.py")
_repo_evaluation = _load_module("repo_evaluation_imu_csv", REPO_ROOT / "src" / "evaluation" / "imu_csv.py")

enforce_quaternion_continuity = _repo_adapter.enforce_quaternion_continuity
load_imu_csv = _repo_evaluation.load_imu_csv
from src.smplx_ops.sensors import compute_sensor_trajectory
from src.smplx_ops.smplx_runner import SMPLXRunner
from src.utils.math3d import quaternion_to_matrix


FPS = 60.0
TARGET_SENSOR = "R_LowArm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GlobalPose-origin synthetic IMU pipeline on the local TotalCapture reference sample.")
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        help="Repository-local processed data root containing the standard triplet.",
    )
    parser.add_argument("--sequence-name", default="S1_freestyle3")
    parser.add_argument("--sensor-name", default=TARGET_SENSOR)
    parser.add_argument(
        "--output-root",
        default="outputs/totalcapture_test/GlobalPose_origin",
        help="Directory where per-sensor CSVs and metadata will be written.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = (
        (REPO_ROOT / args.processed_root).resolve()
        if not Path(args.processed_root).is_absolute()
        else Path(args.processed_root)
    )
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    processed_triplet_dir = processed_root / "totalcapture_test" / args.sequence_name
    raw_imu_csv = processed_triplet_dir / f"{args.sequence_name.lower()}_{args.sensor_name}.csv"
    if not raw_imu_csv.exists():
        raise FileNotFoundError(f"Processed IMU CSV not found: {raw_imu_csv}")
    raw_loaded = load_imu_csv(raw_imu_csv)
    raw_data = {
        "quat": np.stack([raw_loaded[f"quat{i}"] for i in range(4)], axis=1)[:, None, :],
        "acc": np.stack([raw_loaded["acc_x"], raw_loaded["acc_y"], raw_loaded["acc_z"]], axis=1)[:, None, :],
        "gyro": np.stack([raw_loaded["gyro_x"], raw_loaded["gyro_y"], raw_loaded["gyro_z"]], axis=1)[:, None, :],
        "mag": np.stack([raw_loaded["mag_x"], raw_loaded["mag_y"], raw_loaded["mag_z"]], axis=1)[:, None, :],
    }
    generated_data = load_generated_smplx_stream(
        smplx_npz=processed_triplet_dir / f"{args.sequence_name.lower()}_smplx.npz",
        sequence_name=args.sequence_name,
        sensor_name=args.sensor_name,
    )

    align_frame_count = min(
        raw_data["quat"].shape[0],
        generated_data["quat"].shape[0],
    )
    raw_data = trim_stream(raw_data, align_frame_count)
    generated_data = trim_stream(generated_data, align_frame_count)

    csv_dir = output_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_files = {}
    sensor_name = args.sensor_name
    for sensor_index, sensor_name in enumerate([sensor_name]):
        raw_csv = csv_dir / f"{sensor_name}_raw.csv"
        generated_csv = csv_dir / f"{sensor_name}_generated.csv"
        write_sensor_csv(raw_csv, raw_data, sensor_index)
        write_sensor_csv(generated_csv, generated_data, sensor_index)
        csv_files[sensor_name] = {
            "raw_csv": str(raw_csv),
            "generated_csv": str(generated_csv),
        }

    manifest = {
        "raw_sample_dir": str(processed_root.parent / "raw" / "totalcapture" / args.sequence_name),
        "processed_triplet_dir": str(processed_triplet_dir),
        "processed_imu_csv": str(raw_imu_csv),
        "processed_smplx_npz": str(processed_triplet_dir / f"{args.sequence_name.lower()}_smplx.npz"),
        "sequence_name": args.sequence_name,
        "sensor_name": args.sensor_name,
        "seed": args.seed,
        "fps": FPS,
        "frame_count": align_frame_count,
        "sensor_order": [args.sensor_name],
        "csv_files": csv_files,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def trim_stream(data: dict[str, np.ndarray], frame_count: int) -> dict[str, np.ndarray]:
    return {key: value[-frame_count:] for key, value in data.items()}


def load_generated_smplx_stream(
    smplx_npz: Path,
    sequence_name: str,
    sensor_name: str,
) -> dict[str, np.ndarray]:
    data = np.load(smplx_npz, allow_pickle=True)
    gender = str(data["gender"]).lower()
    trans = np.asarray(data["trans"], dtype=np.float32)
    root_orient = np.asarray(data["root_orient"], dtype=np.float32)
    body_pose = np.asarray(data["pose_body"], dtype=np.float32)
    left_hand_pose = np.asarray(data["pose_hand"][:, :45], dtype=np.float32)
    right_hand_pose = np.asarray(data["pose_hand"][:, 45:], dtype=np.float32)
    jaw_pose = np.asarray(data["pose_jaw"], dtype=np.float32)
    leye_pose = np.asarray(data["pose_eye"][:, :3], dtype=np.float32)
    reye_pose = np.asarray(data["pose_eye"][:, 3:], dtype=np.float32)
    betas = np.asarray(data["betas"], dtype=np.float32)
    num_frames = int(trans.shape[0])
    if betas.ndim == 1:
        betas = np.repeat(betas[None, :], num_frames, axis=0)

    model_root = Path("/home/hrli/data_generation/data/interx/raw/smplx_models")
    model = smplx.create(
        str(model_root),
        model_type="smplx",
        gender=gender,
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=int(data["num_betas"]),
        batch_size=num_frames,
    )
    with torch.no_grad():
        output = model(
            betas=torch.tensor(betas, dtype=torch.float32),
            global_orient=torch.tensor(root_orient, dtype=torch.float32),
            body_pose=torch.tensor(body_pose, dtype=torch.float32),
            left_hand_pose=torch.tensor(left_hand_pose, dtype=torch.float32),
            right_hand_pose=torch.tensor(right_hand_pose, dtype=torch.float32),
            jaw_pose=torch.tensor(jaw_pose, dtype=torch.float32),
            leye_pose=torch.tensor(leye_pose, dtype=torch.float32),
            reye_pose=torch.tensor(reye_pose, dtype=torch.float32),
            transl=torch.tensor(trans, dtype=torch.float32),
            return_verts=False,
        )

    joints_full = output.joints.detach().cpu().numpy().astype(np.float32)
    joints_full = SMPLXRunner.convert_smplx_to_pipeline_world(joints_full)
    joints = SMPLXRunner._project_real_joints_to_pipeline_layout(joints_full, list(SMPLX_JOINT_NAMES))
    placement = "wrist_right" if sensor_name == "R_LowArm" else "wrist_left"
    positions, quaternions = compute_sensor_trajectory(joints, placement=placement)
    rotations = np.stack([quaternion_to_matrix(q) for q in quaternions], axis=0).astype(np.float32)[:, None, :, :]

    generated = synthesize_with_globalpose_official_core(
        positions=positions[:, None, :].astype(np.float32),
        rotations=rotations,
        fps=float(data["mocap_frame_rate"]),
        seed=0,
    )
    quat = enforce_quaternion_continuity(generated["quat"][:, 0])
    return {
        "quat": quat[:, None, :],
        "acc": generated["acc"],
        "gyro": generated["gyro"],
        "mag": generated["mag"],
    }


def synthesize_with_globalpose_official_core(
    positions: np.ndarray,
    rotations: np.ndarray,
    fps: float,
    seed: int,
) -> dict[str, np.ndarray]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = torch.tensor(positions, dtype=torch.float32, device=device)
    r = torch.tensor(rotations, dtype=torch.float32, device=device)
    num_frames, num_sensors = p.shape[:2]
    k = float(np.sqrt(np.pi / 8.0))

    rbs = random_rotation_matrices(num_sensors, device=device)
    dp = walking_noise((num_frames, num_sensors, 3), std=1e-3 * k * np.sqrt(1 / fps), device=device)
    dp = dp + torch.randn(num_sensors, 3, device=device) * (1e-2 * k)
    dw = walking_noise((num_frames, num_sensors, 3), std=1e-2 * k * np.sqrt(1 / fps), device=device)
    dw = dw + torch.randn(num_sensors, 3, device=device) * (1e-1 * k)
    dr = axis_angle_to_matrix(dw)

    p_imu = p + r.matmul(dp.unsqueeze(-1)).squeeze(-1)
    r_imu = r.matmul(rbs).matmul(dr)

    IMUSimulator = load_globalpose_imu_simulator()
    simulator = IMUSimulator()
    simulator.set_trajectory(p_imu.squeeze(1), r_imu.squeeze(1), fps=fps)
    a_sensor = simulator.get_acceleration(gW=(0.0, -9.8, 0.0)).unsqueeze(1)
    w_sensor = simulator.get_angular_velocity().unsqueeze(1)
    m_sensor = simulator.get_magnetic_field(mW=(1.0, 0.0, 0.0)).unsqueeze(1)

    a_sensor = torch.normal(a_sensor, std=5e-2) + walking_noise(a_sensor.shape, std=1e-4 * np.sqrt(1 / fps), device=device)
    w_sensor = torch.normal(w_sensor, std=5e-3) + walking_noise(w_sensor.shape, std=1e-5 * np.sqrt(1 / fps), device=device)
    m_sensor = torch.normal(m_sensor, std=5e-3) + walking_noise(m_sensor.shape, std=1e-5 * np.sqrt(1 / fps), device=device)

    delta_rot = axis_angle_to_matrix(w_sensor / fps)
    r_est = torch.empty_like(r_imu)
    r_est[0] = r_imu[0]
    for frame_index in range(1, num_frames):
        r_est[frame_index] = r_est[frame_index - 1].matmul(delta_rot[frame_index])

    n_rot = axis_angle_to_matrix(torch.randn(num_frames, num_sensors, 3, device=device) * (0.1 * k))
    r_est = r_est.matmul(n_rot)

    gravity = torch.tensor([0.0, -9.8, 0.0], dtype=torch.float32, device=device)
    a_model = r_est.matmul(a_sensor.unsqueeze(-1)).squeeze(-1) + gravity
    w_model = r_est.matmul(w_sensor.unsqueeze(-1)).squeeze(-1)
    r_body = r_est.matmul(rbs.transpose(-1, -2))
    m_model = r_body.transpose(-1, -2).matmul(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 3, 1)).squeeze(-1)

    quat = np.stack(
        [[matrix_to_quaternion(r_body[frame_index, sensor_index].cpu().numpy()) for sensor_index in range(num_sensors)] for frame_index in range(num_frames)],
        axis=0,
    )
    return {
        "quat": quat,
        "acc": a_model.cpu().numpy().astype(np.float64),
        "gyro": w_model.cpu().numpy().astype(np.float64),
        "mag": m_model.cpu().numpy().astype(np.float64),
    }


def load_globalpose_imu_simulator():
    simulation_path = REPO_ROOT / "third-party" / "GlobalPose" / "articulate" / "utils" / "imu" / "simulation.py"
    spec = importlib.util.spec_from_file_location("globalpose_imu_simulation", simulation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load IMUSimulator from {simulation_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IMUSimulator


def walking_noise(shape: tuple[int, ...], std: float, device: torch.device) -> torch.Tensor:
    return torch.cumsum(torch.normal(torch.zeros(shape, device=device), std), dim=0)


def random_rotation_matrices(count: int, device: torch.device) -> torch.Tensor:
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


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), np.finfo(np.float64).eps)


def write_sensor_csv(path: Path, data: dict[str, np.ndarray], sensor_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_idx",
                "quat0",
                "quat1",
                "quat2",
                "quat3",
                "acc_x",
                "acc_y",
                "acc_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "mag_x",
                "mag_y",
                "mag_z",
            ]
        )
        for frame_index in range(data["quat"].shape[0]):
            quat = data["quat"][frame_index, sensor_index]
            acc = data["acc"][frame_index, sensor_index]
            gyro = data["gyro"][frame_index, sensor_index]
            mag = data["mag"][frame_index, sensor_index]
            writer.writerow(
                [
                    frame_index + 1,
                    f"{quat[0]:.6f}",
                    f"{quat[1]:.6f}",
                    f"{quat[2]:.6f}",
                    f"{quat[3]:.6f}",
                    f"{acc[0]:.6f}",
                    f"{acc[1]:.6f}",
                    f"{acc[2]:.6f}",
                    f"{gyro[0]:.6f}",
                    f"{gyro[1]:.6f}",
                    f"{gyro[2]:.6f}",
                    f"{mag[0]:.6f}",
                    f"{mag[1]:.6f}",
                    f"{mag[2]:.6f}",
                ]
            )


if __name__ == "__main__":
    main()
