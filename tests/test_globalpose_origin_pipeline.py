from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class GlobalPoseOriginPipelineTest(unittest.TestCase):
    def test_current_generation_impl_does_not_reference_legacy(self) -> None:
        impl_path = Path(__file__).resolve().parents[1] / "scripts" / "totalcapture_test" / "GlobalPose_origin" / "_run_pipeline_impl.py"
        text = impl_path.read_text(encoding="utf-8")
        self.assertNotIn("scripts/legacy", text)
        self.assertNotIn("_synthesize_imu_existing.py", text)

    def test_builds_data_raw_artifact_path_for_sensor(self) -> None:
        from scripts.totalcapture_test.GlobalPose_origin.run_pipeline import build_sensor_artifact_paths

        paths = build_sensor_artifact_paths(
            repo_root=Path("/repo"),
            data_root=Path("/repo/data"),
            output_root=Path("/repo/outputs/totalcapture_test/GlobalPose_origin"),
            sequence_name="S1_freestyle3",
            sensor_name="R_LowArm",
        )

        self.assertEqual(
            paths["raw_sample_dir"],
            Path("/repo/data/raw/totalcapture/S1_freestyle3"),
        )
        self.assertEqual(
            paths["processed_triplet_dir"],
            Path("/repo/data/processed/totalcapture_test/S1_freestyle3"),
        )
        self.assertEqual(
            paths["processed_video_mp4"],
            Path("/repo/data/processed/totalcapture_test/S1_freestyle3/TC_S1_freestyle3_cam1.mp4"),
        )
        self.assertEqual(
            paths["processed_imu_csv"],
            Path("/repo/data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm.csv"),
        )
        self.assertEqual(
            paths["processed_smplx_npz"],
            Path("/repo/data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_smplx.npz"),
        )
        self.assertEqual(
            paths["generated_output_csv"],
            Path("/repo/outputs/totalcapture_test/GlobalPose_origin/csv/R_LowArm_generated.csv"),
        )
        self.assertEqual(
            paths["plot_png"],
            Path("/repo/outputs/totalcapture_test/GlobalPose_origin/plots/R_LowArm_raw_vs_generated.png"),
        )

    def test_report_supports_single_r_lowarm_sensor(self) -> None:
        from scripts.totalcapture_test.GlobalPose_origin.run_pipeline import build_report

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_root = tmp / "outputs"
            output_root.mkdir(parents=True)
            (output_root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "raw_sample_dir": "/repo/data/raw/totalcapture/S1_freestyle3",
                        "processed_triplet_dir": "/repo/data/processed/totalcapture_test/S1_freestyle3",
                        "processed_video_mp4": "/repo/data/processed/totalcapture_test/S1_freestyle3/TC_S1_freestyle3_cam1.mp4",
                        "processed_imu_csv": "/repo/data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm.csv",
                        "processed_smplx_npz": "/repo/data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_smplx.npz",
                        "sensor_order": ["R_LowArm"],
                        "csv_files": {
                            "R_LowArm": {
                                "raw_csv": "/repo/outputs/csv/R_LowArm_raw.csv",
                                "generated_csv": "/repo/outputs/csv/R_LowArm_generated.csv",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            sensor_metrics = {
                "R_LowArm": {
                    "raw_vs_generated": {
                        "real_vs_synthetic": {
                            "acc_magnitude": {
                                "real": {"mean": 1.0, "std": 0.1, "min": 0.5, "median": 1.0, "max": 1.5, "rms": 1.01, "mean_square_energy": 1.02},
                                "synthetic": {"mean": 1.1, "std": 0.2, "min": 0.4, "median": 1.1, "max": 1.6, "rms": 1.12, "mean_square_energy": 1.25},
                                "delta": {"mean": 0.1, "std": 0.1, "min": -0.1, "median": 0.1, "max": 0.1, "rms": 0.11, "mean_square_energy": 0.23},
                                "relative_delta": {"mean": 0.1, "std": 1.0, "min": -0.2, "median": 0.1, "max": 0.07, "rms": 0.1, "mean_square_energy": 0.2},
                            },
                            "gyro_magnitude": {
                                "real": {"mean": 2.0, "std": 0.3, "min": 1.0, "median": 2.0, "max": 3.0, "rms": 2.1, "mean_square_energy": 4.4},
                                "synthetic": {"mean": 2.3, "std": 0.4, "min": 1.1, "median": 2.2, "max": 3.4, "rms": 2.4, "mean_square_energy": 5.6},
                                "delta": {"mean": 0.3, "std": 0.1, "min": 0.1, "median": 0.2, "max": 0.4, "rms": 0.3, "mean_square_energy": 1.2},
                                "relative_delta": {"mean": -0.15, "std": 0.2, "min": 0.1, "median": 0.1, "max": 0.2, "rms": 0.2, "mean_square_energy": 0.4},
                            },
                        },
                        "temporal_consistency": {
                            "acc_magnitude_correlation": 0.7,
                            "gyro_magnitude_correlation": 0.6,
                        },
                        "motion_intensity": {
                            "acc_magnitude_rmse": 2.2,
                            "gyro_magnitude_rmse": 1.3,
                        },
                        "event_consistency": {
                            "acc_peak_timing_error": {
                                "real_peak_count": 10,
                                "synthetic_peak_count": 12,
                                "mean_abs_error_frames": 3.0,
                                "mean_abs_error_seconds": 0.05,
                                "median_abs_error_frames": 2.0,
                                "max_abs_error_frames": 8.0,
                            },
                            "gyro_peak_timing_error": {
                                "real_peak_count": 8,
                                "synthetic_peak_count": 9,
                                "mean_abs_error_frames": 5.0,
                                "mean_abs_error_seconds": 0.08,
                                "median_abs_error_frames": 4.0,
                                "max_abs_error_frames": 10.0,
                            },
                        },
                        "frequency_structure": {
                            "acc_magnitude_psd_distance": 0.4,
                            "gyro_magnitude_psd_distance": 0.9,
                        },
                        "window_statistics": {
                            "window_seconds": 1.0,
                            "window_count": 5,
                            "feature_distance": {
                                "acc_magnitude": {"mean": 0.1, "std": 0.2, "max": 0.3, "energy": 0.4, "overall_rmse": 0.25},
                                "gyro_magnitude": {"mean": 0.5, "std": 0.6, "max": 0.7, "energy": 0.8, "overall_rmse": 0.65},
                            },
                        },
                    },
                }
            }

            report = build_report(
                output_root=output_root,
                seed=0,
                sensor_metrics=sensor_metrics,
            )

        self.assertIn("| R_LowArm |", report)
        self.assertIn("SMPL-X", report)
        self.assertIn("raw vs generated", report)
        self.assertIn("原始样例目录", report)
        self.assertIn("标准三元组目录", report)
        self.assertIn("原始与生成统计", report)
        self.assertIn("事件一致性", report)
        self.assertIn("窗口统计", report)
        self.assertNotIn("原始 IMU 来源", report)
        self.assertNotIn("| L_LowArm |", report)
        self.assertNotIn("| Head |", report)

    def test_pipeline_helper_evaluates_and_writes_metrics_via_src_evaluation(self) -> None:
        from scripts.totalcapture_test.GlobalPose_origin.run_pipeline import evaluate_sensor_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_csv = tmp / "raw.csv"
            candidate_csv = tmp / "candidate.csv"
            output_json = tmp / "metrics.json"
            csv_text = (
                "frame_idx,quat0,quat1,quat2,quat3,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z\n"
                "1,1,0,0,0,0,0,1,0,1,0,0,0,0\n"
                "2,1,0,0,0,0,0,2,0,2,0,0,0,0\n"
            )
            raw_csv.write_text(csv_text, encoding="utf-8")
            candidate_csv.write_text(csv_text, encoding="utf-8")

            metrics = evaluate_sensor_metrics(reference_csv=raw_csv, candidate_csv=candidate_csv, output_json=output_json, fps=60.0)
            self.assertTrue(output_json.is_file())
            self.assertEqual(metrics["metadata"]["real_csv"], str(raw_csv))
            self.assertAlmostEqual(metrics["motion_intensity"]["acc_magnitude_rmse"], 0.0)
            written = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(written["metadata"]["synthetic_csv"], str(candidate_csv))


if __name__ == "__main__":
    unittest.main()
