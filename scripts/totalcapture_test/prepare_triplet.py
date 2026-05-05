from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.globalpose_origin_adapter import (
    prepare_totalcapture_processed_triplet,
    stage_totalcapture_raw_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the standard TotalCapture triplet under data/processed.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--sequence-name", default="S1_freestyle3")
    parser.add_argument("--sensor-name", default="R_LowArm")
    parser.add_argument("--camera-name", default="cam1")
    parser.add_argument("--raw-imu-source-root", default="/data/lxhong")
    parser.add_argument("--totalcapture-source-root", default="/data/lxhong/totalcapture")
    parser.add_argument(
        "--smplx-source",
        default="data/legacy/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_smplx.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = (REPO_ROOT / args.raw_root).resolve() if not Path(args.raw_root).is_absolute() else Path(args.raw_root)
    processed_root = (REPO_ROOT / args.processed_root).resolve() if not Path(args.processed_root).is_absolute() else Path(args.processed_root)
    smplx_source = (REPO_ROOT / args.smplx_source).resolve() if not Path(args.smplx_source).is_absolute() else Path(args.smplx_source)
    raw_sample = stage_totalcapture_raw_sample(
        totalcapture_source_root=Path(args.totalcapture_source_root),
        imu_source_root=Path(args.raw_imu_source_root),
        raw_root=raw_root,
        sequence_name=args.sequence_name,
    )
    triplet = prepare_totalcapture_processed_triplet(
        raw_sample_dir=raw_sample["sequence_dir"],
        processed_root=processed_root,
        sequence_name=args.sequence_name,
        sensor_name=args.sensor_name,
        smplx_source=smplx_source,
        camera_name=args.camera_name,
    )
    print(json.dumps({"raw_sample": raw_sample, "triplet": triplet}, indent=2))


if __name__ == "__main__":
    main()
