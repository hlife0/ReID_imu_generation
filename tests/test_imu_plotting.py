from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ImuPlottingTest(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[str]) -> None:
        path.write_text(
            "\n".join(
                [
                    "frame_idx,quat0,quat1,quat2,quat3,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z",
                    *rows,
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_cli_generates_overlay_png(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "totalcapture_test" / "plot_imu_comparison.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            real_csv = tmp / "real.csv"
            synthetic_csv = tmp / "synthetic.csv"
            output_png = tmp / "comparison.png"

            self._write_csv(
                real_csv,
                [
                    "1,1,0,0,0,0,0,1,0.1,0.2,0.3,0.4,0.5,0.6",
                    "2,1,0,0,0,0,1,0,0.7,0.8,0.9,1.0,1.1,1.2",
                    "3,1,0,0,0,1,0,0,1.3,1.4,1.5,1.6,1.7,1.8",
                ],
            )
            self._write_csv(
                synthetic_csv,
                [
                    "1,0.9,0.1,0,0,0,0,0.8,0.2,0.3,0.4,0.5,0.6,0.7",
                    "2,0.9,0.1,0,0,0,0.8,0,0.8,0.9,1.0,1.1,1.2,1.3",
                    "3,0.9,0.1,0,0,0.8,0,0,1.4,1.5,1.6,1.7,1.8,1.9",
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--real-csv",
                    str(real_csv),
                    "--synthetic-csv",
                    str(synthetic_csv),
                    "--output-png",
                    str(output_png),
                    "--plot-python",
                    "/home/hrli/data_generation/.venv/bin/python",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue(output_png.is_file())
            self.assertGreater(output_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
