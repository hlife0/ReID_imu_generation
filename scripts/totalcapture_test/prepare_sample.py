from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.totalcapture_test import stage_totalcapture_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage one TotalCapture test sample as video + single-sensor IMU + SMPL-X.")
    parser.add_argument("--sequence-id", default="S1_freestyle3")
    parser.add_argument("--video-source-root", default="/data/lxhong/totalcapture")
    parser.add_argument("--imu-source-root", default="/data/lxhong")
    parser.add_argument("--stageii-totalcapture-root", default="/data/luoyizhang/HuMoGen/data/AMASS/TotalCapture")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sensor-name", default="R_LowArm")
    parser.add_argument("--camera-name", default="cam1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = stage_totalcapture_test(
        video_source_root=Path(args.video_source_root),
        imu_source_root=Path(args.imu_source_root),
        stageii_totalcapture_root=Path(args.stageii_totalcapture_root),
        data_root=Path(args.data_root),
        sequence_id=args.sequence_id,
        sensor_name=args.sensor_name,
        camera_name=args.camera_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
