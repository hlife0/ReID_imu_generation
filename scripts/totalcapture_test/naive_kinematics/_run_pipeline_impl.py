from __future__ import annotations

import argparse
import csv
import json
import sys
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


def _load_repo_module(module_name: str, module_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_repo_adapter = _load_repo_module("repo_globalpose_origin_adapter", REPO_ROOT / "src" / "globalpose_origin_adapter.py")
_repo_evaluation = _load_repo_module("repo_evaluation_imu_csv", REPO_ROOT / "src" / "evaluation" / "imu_csv.py")

enforce_quaternion_continuity = _repo_adapter.enforce_quaternion_continuity
load_imu_csv = _repo_evaluation.load_imu_csv

from src.imu.synthesize import synthesize_imu_from_world_trajectory
from src.smplx_ops.sensors import compute_sensor_trajectory
from src.smplx_ops.smplx_runner import SMPLXRunner


TARGET_SENSOR = "R_LowArm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the naive_kinematics synthetic IMU pipeline on the current processed triplet.")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--sequence-name", default="S1_freestyle3")
    parser.add_argument("--sensor-name", default=TARGET_SENSOR)
    parser.add_argument("--output-root", default="outputs/totalcapture_test/naive_kinematics")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = (REPO_ROOT / args.processed_root).resolve() if not Path(args.processed_root).is_absolute() else Path(args.processed_root)
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    processed_triplet_dir = processed_root / "totalcapture_test" / args.sequence_name
    raw_imu_csv = processed_triplet_dir / f"{args.sequence_name.lower()}_{args.sensor_name}.csv"
    smplx_npz = processed_triplet_dir / f"{args.sequence_name.lower()}_smplx.npz"
    if not raw_imu_csv.exists():
        raise FileNotFoundError(f"Processed IMU CSV not found: {raw_imu_csv}")
    if not smplx_npz.exists():
        raise FileNotFoundError(f"Processed SMPL-X npz not found: {smplx_npz}")

    raw_loaded = load_imu_csv(raw_imu_csv)
    raw_data = {
        "quat": np.stack([raw_loaded[f"quat{i}"] for i in range(4)], axis=1)[:, None, :],
        "acc": np.stack([raw_loaded["acc_x"], raw_loaded["acc_y"], raw_loaded["acc_z"]], axis=1)[:, None, :],
        "gyro": np.stack([raw_loaded["gyro_x"], raw_loaded["gyro_y"], raw_loaded["gyro_z"]], axis=1)[:, None, :],
        "mag": np.stack([raw_loaded["mag_x"], raw_loaded["mag_y"], raw_loaded["mag_z"]], axis=1)[:, None, :],
    }
    generated_data = load_generated_smplx_stream(smplx_npz=smplx_npz, sensor_name=args.sensor_name)

    align_frame_count = min(raw_data["quat"].shape[0], generated_data["quat"].shape[0])
    raw_data = trim_stream(raw_data, align_frame_count)
    generated_data = trim_stream(generated_data, align_frame_count)

    csv_dir = output_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = csv_dir / f"{args.sensor_name}_raw.csv"
    generated_csv = csv_dir / f"{args.sensor_name}_generated.csv"
    write_sensor_csv(raw_csv, raw_data, 0)
    write_sensor_csv(generated_csv, generated_data, 0)

    manifest = {
        "method": "naive_kinematics",
        "raw_sample_dir": str(processed_root.parent / "raw" / "totalcapture" / args.sequence_name),
        "processed_triplet_dir": str(processed_triplet_dir),
        "processed_video_mp4": str(processed_triplet_dir / f"TC_{args.sequence_name}_cam1.mp4"),
        "processed_imu_csv": str(raw_imu_csv),
        "processed_smplx_npz": str(smplx_npz),
        "sequence_name": args.sequence_name,
        "sensor_name": args.sensor_name,
        "seed": args.seed,
        "fps": int(round(float(np.load(smplx_npz, allow_pickle=True)["mocap_frame_rate"]))),
        "frame_count": align_frame_count,
        "sensor_order": [args.sensor_name],
        "csv_files": {
            args.sensor_name: {
                "raw_csv": str(raw_csv),
                "generated_csv": str(generated_csv),
            }
        },
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def load_generated_smplx_stream(smplx_npz: Path, sensor_name: str) -> dict[str, np.ndarray]:
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
    fps = int(round(float(data["mocap_frame_rate"])))

    imu = synthesize_imu_from_world_trajectory(
        positions=positions,
        quaternions=quaternions,
        fps=fps,
        gravity_world=np.array([0.0, -9.81, 0.0], dtype=np.float32),
    )

    quat = enforce_quaternion_continuity(imu["quat"].astype(np.float64))
    zeros_mag = np.zeros_like(imu["accel"], dtype=np.float64)
    return {
        "quat": quat[:, None, :],
        "acc": imu["accel"].astype(np.float64)[:, None, :],
        "gyro": imu["gyro"].astype(np.float64)[:, None, :],
        "mag": zeros_mag[:, None, :],
    }


def trim_stream(data: dict[str, np.ndarray], frame_count: int) -> dict[str, np.ndarray]:
    return {key: value[-frame_count:] for key, value in data.items()}


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
