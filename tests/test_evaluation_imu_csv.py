from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class EvaluationImuCsvTest(unittest.TestCase):
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

    def test_evaluation_wrapper_accepts_real_and_synthetic_csv_paths(self) -> None:
        from src.evaluation import evaluate_imu_csv_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            real_csv = tmp / "real.csv"
            synthetic_csv = tmp / "synthetic.csv"
            self._write_csv(
                real_csv,
                [
                    "1,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "2,1,0,0,0,0,0,2,0,2,0,0,0,0",
                    "3,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "4,1,0,0,0,0,0,3,0,3,0,0,0,0",
                ],
            )
            self._write_csv(
                synthetic_csv,
                [
                    "1,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "2,1,0,0,0,0,0,2,0,2,0,0,0,0",
                    "3,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "4,1,0,0,0,0,0,3,0,3,0,0,0,0",
                ],
            )

            metrics = evaluate_imu_csv_pair(real_csv=real_csv, synthetic_csv=synthetic_csv, fps=4.0)

        self.assertEqual(metrics["metadata"]["real_csv"], str(real_csv))
        self.assertEqual(metrics["metadata"]["synthetic_csv"], str(synthetic_csv))
        self.assertEqual(metrics["metadata"]["frame_count"], 4)
        self.assertAlmostEqual(metrics["motion_intensity"]["acc_magnitude_rmse"], 0.0)
        self.assertAlmostEqual(metrics["motion_intensity"]["gyro_magnitude_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
