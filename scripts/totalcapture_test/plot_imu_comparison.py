from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real and synthetic TotalCapture test IMU on the same figure.")
    parser.add_argument("--real-csv", default="data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm.csv")
    parser.add_argument("--synthetic-csv", default="data/interim/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm_synthetic.csv")
    parser.add_argument("--output-png", default="outputs/totalcapture_test/S1_freestyle3/r_lowarm_real_vs_synthetic.png")
    parser.add_argument("--plot-python", default="/home/hrli/data_generation/.venv/bin/python")
    parser.add_argument("--title", default="S1_freestyle3 R_LowArm: real vs synthetic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helper = Path(__file__).resolve().parent / "_plot_imu_comparison.py"
    completed = subprocess.run(
        [
            args.plot_python,
            str(helper),
            "--real-csv",
            args.real_csv,
            "--synthetic-csv",
            args.synthetic_csv,
            "--output-png",
            args.output_png,
            "--title",
            args.title,
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or completed.stdout or "plot helper failed")
    print(args.output_png)


if __name__ == "__main__":
    main()
