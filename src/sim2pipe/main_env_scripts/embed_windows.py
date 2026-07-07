"""Extract per-window IMU/video embeddings with the MAIN PROJECT's model.

Executed INSIDE the main project's python environment (autism_test) with the
main repo as cwd/PYTHONPATH — launched by scripts/sim2pipe/04_eval_fullgallery.py
via bridge.run_main_python. Mirrors the model/dataset setup of the main repo's
src/engine/eval.py exactly (frozen MotionBERT + DeSPITE + trained alignment
head, WindowAlignmentDataset, shuffle=False), but instead of computing the
in-batch top1 it DUMPS all embeddings to an npz, so the caller can apply the
sim2real judgment (full-gallery IMU->motion retrieval) outside.

Row order in the output equals the CSV row order (shuffle=False).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.alignment_dataset import WindowAlignmentDataset
from src.engine.common import build_alignment_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--motionbert_root", type=str, default="/home/fzliang/origin/MotionBERT")
    parser.add_argument("--motionbert_config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml")
    parser.add_argument("--motionbert_ckpt", type=str, default="")
    parser.add_argument("--skip_motionbert_ckpt", action="store_true")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--imu_stats_json", type=str, required=True)
    parser.add_argument("--imu_sensor", type=str, default="R_LowArm")
    parser.add_argument("--repeat_single_sensor", type=int, default=4)
    parser.add_argument("--imu_lowpass_cutoff_hz", type=float, default=None)
    parser.add_argument("--imu_lowpass_fs_hz", type=float, default=30.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_npz", type=str, required=True)
    # build_alignment_model reads these attributes; keep eval.py defaults.
    parser.add_argument("--imu_encoder_type", type=str, default="lstm")
    parser.add_argument("--physics_d_model", type=int, default=128)
    parser.add_argument("--physics_n_heads", type=int, default=4)
    parser.add_argument("--physics_num_layers", type=int, default=3)
    parser.add_argument("--physics_fs_hz", type=float, default=30.0)
    parser.add_argument("--physics_n_fft", type=int, default=64)
    parser.add_argument("--physics_dropout", type=float, default=0.1)
    parser.add_argument("--use_global_motion", action="store_true")
    parser.add_argument("--global_motion_input_dim", type=int, default=2)
    parser.add_argument("--global_motion_hidden_dim", type=int, default=64)
    parser.add_argument("--global_motion_num_layers", type=int, default=2)
    parser.add_argument("--global_motion_dropout", type=float, default=0.1)
    parser.add_argument("--global_motion_input_type", type=str, default="diff_raw")
    parser.add_argument("--global_motion_fusion_type", type=str, default="concat")
    parser.add_argument("--global_motion_fusion_proj", action="store_true")
    parser.add_argument("--global_motion_root_source", type=str, default="auto")
    parser.add_argument("--global_motion_train_only", action="store_true")
    parser.add_argument("--global_motion_aux_weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    stats = json.loads(Path(args.imu_stats_json).read_text())
    imu_mean = np.asarray(stats["imu_mean"], dtype=np.float32)
    imu_std = np.asarray(stats["imu_std"], dtype=np.float32)

    ds = WindowAlignmentDataset(
        args.test_csv,
        root_dir=args.data_root,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_sensor=args.imu_sensor.strip() or None,
        repeat_single_sensor=args.repeat_single_sensor,
        imu_lowpass_cutoff_hz=args.imu_lowpass_cutoff_hz,
        imu_lowpass_fs_hz=args.imu_lowpass_fs_hz,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model, _ = build_alignment_model(args, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    imu_chunks, video_chunks = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(imu=batch["imu"].to(device), skeleton=batch["skeleton"].to(device))
            imu_chunks.append(out["imu"].float().cpu().numpy())
            video_chunks.append(out["video"].float().cpu().numpy())

    imu_emb = np.concatenate(imu_chunks, axis=0)
    video_emb = np.concatenate(video_chunks, axis=0)
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, imu=imu_emb, video=video_emb,
                        checkpoint=str(args.checkpoint), test_csv=str(args.test_csv))
    print(json.dumps({"out": str(out_path), "n": int(imu_emb.shape[0]),
                      "dim": int(imu_emb.shape[1])}))


if __name__ == "__main__":
    main()
