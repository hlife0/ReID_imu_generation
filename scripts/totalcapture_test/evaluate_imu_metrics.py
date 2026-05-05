from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_imu_csv_pair, write_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real vs synthetic IMU metrics for the TotalCapture test sample.")
    parser.add_argument("--real-csv", default="data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm.csv")
    parser.add_argument("--synthetic-csv", default="data/interim/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm_synthetic.csv")
    parser.add_argument("--output-json", default="outputs/totalcapture_test/S1_freestyle3/imu_metrics.json")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--peak-min-distance-seconds", type=float, default=0.25)
    parser.add_argument("--peak-prominence-fraction", type=float, default=0.10)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--window-overlap", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_imu_csv_pair(
        real_csv=args.real_csv,
        synthetic_csv=args.synthetic_csv,
        fps=args.fps,
        peak_min_distance_seconds=args.peak_min_distance_seconds,
        peak_prominence_fraction=args.peak_prominence_fraction,
        window_seconds=args.window_seconds,
        window_overlap=args.window_overlap,
    )
    write_metrics_json(metrics, args.output_json)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
