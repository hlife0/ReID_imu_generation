from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES

DATA_GENERATION_ROOT = Path("/home/hrli/data_generation")
GLOBALPOSE_ROOT = Path("/home/hrli/imu_generation/third-party/GlobalPose")
if str(DATA_GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_GENERATION_ROOT))
from src.smplx_ops.sensors import compute_sensor_trajectory
from src.smplx_ops.smplx_runner import SMPLXRunner
from src.utils.math3d import quaternion_to_matrix


def _load_globalpose_imu_simulator():
    simulation_path = GLOBALPOSE_ROOT / "articulate" / "utils" / "imu" / "simulation.py"
    spec = importlib.util.spec_from_file_location("globalpose_imu_simulation", simulation_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load IMUSimulator from {simulation_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IMUSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize a right-arm IMU CSV using existing local SMPL-X and GlobalPose tools.")
    parser.add_argument("--input-smplx", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input_smplx, allow_pickle=True)

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

    model = smplx.create(
        str(Path(args.model_root)),
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
    positions, quaternions = compute_sensor_trajectory(joints, placement="wrist_right")
    rotations = np.stack([quaternion_to_matrix(q) for q in quaternions], axis=0).astype(np.float32)

    IMUSimulator = _load_globalpose_imu_simulator()
    simulator = IMUSimulator()
    simulator.set_trajectory(
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(rotations, dtype=torch.float32),
        fps=float(data["mocap_frame_rate"]),
    )
    acc = simulator.get_acceleration(gW=(0.0, -9.8, 0.0)).cpu().numpy().astype(np.float32)
    gyro = simulator.get_angular_velocity().cpu().numpy().astype(np.float32)
    mag = simulator.get_magnetic_field(mW=(1.0, 0.0, 0.0)).cpu().numpy().astype(np.float32)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        for idx in range(acc.shape[0]):
            writer.writerow(
                [
                    idx + 1,
                    f"{quaternions[idx, 0]:.6f}",
                    f"{quaternions[idx, 1]:.6f}",
                    f"{quaternions[idx, 2]:.6f}",
                    f"{quaternions[idx, 3]:.6f}",
                    f"{acc[idx, 0]:.6f}",
                    f"{acc[idx, 1]:.6f}",
                    f"{acc[idx, 2]:.6f}",
                    f"{gyro[idx, 0]:.6f}",
                    f"{gyro[idx, 1]:.6f}",
                    f"{gyro[idx, 2]:.6f}",
                    f"{mag[idx, 0]:.6f}",
                    f"{mag[idx, 1]:.6f}",
                    f"{mag[idx, 2]:.6f}",
                ]
            )


if __name__ == "__main__":
    main()
