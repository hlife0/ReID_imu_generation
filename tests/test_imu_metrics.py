from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class ImuMetricsTest(unittest.TestCase):
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

    def test_evaluates_table_metrics_from_real_and_synthetic_csv(self) -> None:
        from src.imu_metrics import evaluate_imu_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            real_csv = tmp / "real.csv"
            synthetic_csv = tmp / "synthetic.csv"
            self._write_csv(
                real_csv,
                [
                    "1,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "2,1,0,0,0,0,0,3,0,2,0,0,0,0",
                    "3,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "4,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "5,1,0,0,0,0,4,0,0,0,4,0,0,0",
                    "6,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "7,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "8,1,0,0,0,5,0,0,5,0,0,0,0,0",
                ],
            )
            self._write_csv(
                synthetic_csv,
                [
                    "1,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "2,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "3,1,0,0,0,0,0,3,0,2,0,0,0,0",
                    "4,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "5,1,0,0,0,0,4,0,0,0,4,0,0,0",
                    "6,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "7,1,0,0,0,0,0,1,0,0,0,0,0,0",
                    "8,1,0,0,0,5,0,0,5,0,0,0,0,0",
                ],
            )

            metrics = evaluate_imu_pair(
                real_csv,
                synthetic_csv,
                fps=4.0,
                peak_min_distance_seconds=0.25,
                peak_prominence_fraction=0.15,
                window_seconds=1.0,
                window_overlap=0.5,
            )

        self.assertEqual(metrics["metadata"]["frame_count"], 8)
        comparison = metrics["real_vs_synthetic"]
        self.assertAlmostEqual(comparison["acc_magnitude"]["real"]["mean"], 2.125)
        self.assertAlmostEqual(comparison["acc_magnitude"]["synthetic"]["mean"], 2.125)
        self.assertAlmostEqual(comparison["acc_magnitude"]["delta"]["mean"], 0.0)
        self.assertAlmostEqual(comparison["acc_magnitude"]["relative_delta"]["mean"], 0.0)
        self.assertAlmostEqual(comparison["gyro_magnitude"]["real"]["mean"], 1.375)
        self.assertAlmostEqual(comparison["gyro_magnitude"]["synthetic"]["mean"], 1.375)
        self.assertAlmostEqual(comparison["gyro_magnitude"]["delta"]["mean"], 0.0)
        self.assertAlmostEqual(comparison["gyro_magnitude"]["relative_delta"]["mean"], 0.0)
        self.assertAlmostEqual(metrics["motion_intensity"]["acc_magnitude_rmse"], 1.0)
        self.assertAlmostEqual(metrics["motion_intensity"]["gyro_magnitude_rmse"], 1.0)
        self.assertLess(metrics["temporal_consistency"]["acc_magnitude_correlation"], 1.0)
        self.assertLess(metrics["temporal_consistency"]["gyro_magnitude_correlation"], 1.0)

        acc_peak = metrics["event_consistency"]["acc_peak_timing_error"]
        gyro_peak = metrics["event_consistency"]["gyro_peak_timing_error"]
        self.assertEqual(acc_peak["real_peak_count"], 3)
        self.assertEqual(acc_peak["synthetic_peak_count"], 3)
        self.assertEqual(acc_peak["matched_peak_count"], 3)
        self.assertAlmostEqual(acc_peak["mean_abs_error_frames"], 1.0 / 3.0)
        self.assertAlmostEqual(acc_peak["mean_abs_error_seconds"], (1.0 / 3.0) / 4.0)
        self.assertEqual(gyro_peak["matched_peak_count"], 3)
        self.assertAlmostEqual(gyro_peak["mean_abs_error_frames"], 1.0 / 3.0)

        self.assertIn("acc_magnitude_psd_distance", metrics["frequency_structure"])
        self.assertIn("gyro_magnitude_psd_distance", metrics["frequency_structure"])
        self.assertGreaterEqual(metrics["frequency_structure"]["acc_magnitude_psd_distance"], 0.0)
        self.assertGreaterEqual(metrics["frequency_structure"]["gyro_magnitude_psd_distance"], 0.0)

        window_stats = metrics["window_statistics"]
        self.assertEqual(window_stats["window_size_frames"], 4)
        self.assertEqual(window_stats["step_size_frames"], 2)
        self.assertEqual(window_stats["window_count"], 3)
        for signal_name in ["acc_magnitude", "gyro_magnitude"]:
            self.assertIn(signal_name, window_stats["feature_distance"])
            self.assertIn("overall_rmse", window_stats["feature_distance"][signal_name])
            for feature in ["mean", "std", "max", "energy"]:
                self.assertIn(feature, window_stats["feature_distance"][signal_name])

    def test_zero_distance_for_identical_signals(self) -> None:
        from src.imu_metrics import evaluate_imu_pair

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "imu.csv"
            self._write_csv(
                csv_path,
                [
                    "1,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "2,1,0,0,0,0,0,2,0,2,0,0,0,0",
                    "3,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "4,1,0,0,0,0,0,3,0,3,0,0,0,0",
                    "5,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "6,1,0,0,0,0,0,4,0,4,0,0,0,0",
                    "7,1,0,0,0,0,0,1,0,1,0,0,0,0",
                    "8,1,0,0,0,0,0,2,0,2,0,0,0,0",
                ],
            )

            metrics = evaluate_imu_pair(
                csv_path,
                csv_path,
                fps=4.0,
                peak_min_distance_seconds=0.25,
                window_seconds=1.0,
                window_overlap=0.5,
            )

        self.assertAlmostEqual(metrics["motion_intensity"]["acc_magnitude_rmse"], 0.0)
        self.assertAlmostEqual(metrics["motion_intensity"]["gyro_magnitude_rmse"], 0.0)
        self.assertAlmostEqual(metrics["temporal_consistency"]["acc_magnitude_correlation"], 1.0)
        self.assertAlmostEqual(metrics["temporal_consistency"]["gyro_magnitude_correlation"], 1.0)
        self.assertAlmostEqual(metrics["frequency_structure"]["acc_magnitude_psd_distance"], 0.0)
        self.assertAlmostEqual(metrics["frequency_structure"]["gyro_magnitude_psd_distance"], 0.0)
        self.assertAlmostEqual(
            metrics["window_statistics"]["feature_distance"]["acc_magnitude"]["overall_rmse"],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["window_statistics"]["feature_distance"]["gyro_magnitude"]["overall_rmse"],
            0.0,
        )

    def test_peak_timing_compares_each_real_peak_to_nearest_synthetic_peak(self) -> None:
        from src.imu_metrics import peak_timing_error

        real = np.asarray([0.0, 5.0, 0.0, 0.0, 0.0, 4.0, 0.0])
        synthetic = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0])

        metrics = peak_timing_error(
            real,
            synthetic,
            fps=1.0,
            min_distance_seconds=1.0,
            prominence_fraction=0.1,
        )

        self.assertEqual(metrics["real_peak_count"], 2)
        self.assertEqual(metrics["synthetic_peak_count"], 1)
        self.assertEqual(metrics["matched_peak_count"], 2)
        self.assertAlmostEqual(metrics["mean_abs_error_frames"], 2.0)
        self.assertAlmostEqual(metrics["median_abs_error_frames"], 2.0)
        self.assertAlmostEqual(metrics["max_abs_error_frames"], 4.0)


if __name__ == "__main__":
    unittest.main()
