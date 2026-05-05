from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot legacy TotalCapture real vs synthetic IMU on the same figure.")
    parser.add_argument("--real-csv", default="data/legacy/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm.csv")
    parser.add_argument("--synthetic-csv", default="data/legacy/interim/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm_synthetic.csv")
    parser.add_argument("--output-png", default="outputs/legacy/totalcapture_test/S1_freestyle3/r_lowarm_real_vs_synthetic.png")
    parser.add_argument("--plot-python", default=sys.executable)
    parser.add_argument("--title", default="Legacy S1_freestyle3 R_LowArm: real vs synthetic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = REPO_ROOT / "scripts" / "totalcapture_test" / "plot_imu_comparison.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--real-csv",
            args.real_csv,
            "--synthetic-csv",
            args.synthetic_csv,
            "--output-png",
            args.output_png,
            "--plot-python",
            args.plot_python,
            "--title",
            args.title,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.stderr or completed.stdout or "legacy plot helper failed")
    print(args.output_png)


if __name__ == "__main__":
    main()
