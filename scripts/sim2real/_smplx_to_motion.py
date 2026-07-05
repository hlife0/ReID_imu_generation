"""SMPL-X stageii npz -> canonical motion.npz (MotionSequence) — runs in the
external generator venv (needs torch + smplx + data_generation modules).

One SMPL-X forward per sequence, shared by every generator adapter — the
expensive step happens once, adapters only re-derive the cheap sensor
trajectory from the stored joints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# NOTE: data_generation's `src` is a REGULAR package (has __init__.py) and
# therefore shadows this repo's `src` namespace package entirely. We expose
# this repo's code as the top-level `sim2real` package instead by putting
# REPO_ROOT/src on sys.path; `src.*` then always means data_generation.
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
DATA_GENERATION_ROOT = Path("/home/hrli/data_generation")
if str(DATA_GENERATION_ROOT) not in sys.path:
    sys.path.append(str(DATA_GENERATION_ROOT))

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES

from sim2real.contracts import MotionSequence, write_manifest
from src.smplx_ops.smplx_runner import SMPLXRunner

JOINT_LAYOUT = "data_generation_pipeline_v1"
DEFAULT_MODEL_ROOT = "/home/hrli/data_generation/data/interx/raw/smplx_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smplx", required=True, help="AMASS SMPL-X stageii npz")
    parser.add_argument("--out", required=True, help="output motion.npz path")
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smplx_path = Path(args.smplx)
    out_path = Path(args.out)

    data = np.load(smplx_path, allow_pickle=True)
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
    fps = float(data["mocap_frame_rate"])
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

    motion = MotionSequence(
        joints=joints,
        fps=fps,
        joint_layout=JOINT_LAYOUT,
        meta={
            "smplx_source": str(smplx_path),
            "gender": gender,
            "num_betas": int(data["num_betas"]),
        },
    )
    motion.save(out_path)
    write_manifest(
        out_path.parent / "motion.manifest.json",
        stage="smplx_to_motion",
        inputs={"smplx": smplx_path},
        extra={"frames": num_frames, "fps": fps, "gender": gender, "joint_layout": JOINT_LAYOUT},
    )
    print(json.dumps({"out": str(out_path), "frames": num_frames, "fps": fps}))


if __name__ == "__main__":
    main()
