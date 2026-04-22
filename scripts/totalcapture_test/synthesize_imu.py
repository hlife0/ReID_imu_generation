from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize a right-forearm IMU sequence from a stored SMPL-X sequence.")
    parser.add_argument("--input-smplx", default="data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_smplx.npz")
    parser.add_argument("--output-csv", default="data/interim/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm_synthetic.csv")
    parser.add_argument("--smplx-python", default="/home/hrli/data_generation/.venv/bin/python")
    parser.add_argument("--model-root", default="/home/hrli/data_generation/data/interx/raw/smplx_models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_smplx = Path(args.input_smplx)
    output_csv = Path(args.output_csv)
    helper_script = Path(__file__).resolve().parent / "_synthesize_imu_existing.py"
    completed = subprocess.run(
        [
            args.smplx_python,
            str(helper_script),
            "--input-smplx",
            str(input_smplx),
            "--output-csv",
            str(output_csv),
            "--model-root",
            args.model_root,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "existing-tool IMU synthesis helper failed")

    print(
        json.dumps(
            {
                "input_smplx": str(input_smplx),
                "output_csv": str(output_csv),
                "smplx_python": args.smplx_python,
                "model_root": args.model_root,
                "backend": "data_generation.SPLMXRunner + data_generation.compute_sensor_trajectory + GlobalPose.IMUSimulator",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
