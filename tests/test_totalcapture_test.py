from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class StageTotalCaptureTestTripletTest(unittest.TestCase):
    def _write_fake_xsens(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "3\t2",
                    "1",
                    "Head\t1\t0\t0\t0\t1\t2\t3",
                    "R_LowArm\t0\t1\t0\t0\t4\t5\t6",
                    "L_LowArm\t0\t0\t1\t0\t7\t8\t9",
                    "2",
                    "Head\t1\t0\t0\t0\t10\t11\t12",
                    "R_LowArm\t0\t1\t0\t0\t13\t14\t15",
                    "L_LowArm\t0\t0\t1\t0\t16\t17\t18",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _write_fake_stageii(self, path: Path) -> None:
        np.savez(
            path,
            gender=np.array("neutral"),
            surface_model_type=np.array("smplx"),
            mocap_frame_rate=np.array(60.0),
            mocap_time_length=np.array(2.0 / 60.0),
            trans=np.arange(6, dtype=np.float32).reshape(2, 3),
            betas=np.arange(16, dtype=np.float32),
            num_betas=np.array(16),
            root_orient=np.arange(6, dtype=np.float32).reshape(2, 3),
            pose_body=np.arange(126, dtype=np.float32).reshape(2, 63),
            pose_hand=np.arange(180, dtype=np.float32).reshape(2, 90),
            pose_jaw=np.arange(6, dtype=np.float32).reshape(2, 3),
            pose_eye=np.arange(12, dtype=np.float32).reshape(2, 6),
            poses=np.arange(330, dtype=np.float32).reshape(2, 165),
        )

    def test_stages_exact_three_files_for_totalcapture_test_sequence(self) -> None:
        from src.totalcapture_test import stage_totalcapture_test

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "raw_totalcapture"
            stageii_root = tmp / "amass_totalcapture"
            data_root = tmp / "repo_data"

            (raw_root / "freestyle3").mkdir(parents=True)
            (raw_root / "s1").mkdir(parents=True)
            (stageii_root / "s1").mkdir(parents=True)

            (raw_root / "freestyle3" / "TC_S1_freestyle3_cam1.mp4").write_bytes(b"fake video")
            self._write_fake_xsens(raw_root / "s1" / "s1_freestyle3_Xsens.sensors")
            self._write_fake_stageii(stageii_root / "s1" / "freestyle3_stageii.npz")

            result = stage_totalcapture_test(
                raw_totalcapture_root=raw_root,
                stageii_totalcapture_root=stageii_root,
                data_root=data_root,
                sequence_id="S1_freestyle3",
                sensor_name="R_LowArm",
                camera_name="cam1",
            )

            output_dir = data_root / "processed" / "totalcapture_test" / "S1_freestyle3"
            files = sorted(p.name for p in output_dir.iterdir())
            self.assertEqual(
                files,
                [
                    "TC_S1_freestyle3_cam1.mp4",
                    "s1_freestyle3_R_LowArm.csv",
                    "s1_freestyle3_smplx.npz",
                ],
            )
            self.assertEqual(result["sequence_id"], "S1_freestyle3")
            self.assertEqual(result["output_dir"], str(output_dir))

            imu_csv = (output_dir / "s1_freestyle3_R_LowArm.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                imu_csv[0],
                "frame_idx,quat0,quat1,quat2,quat3,acc_x,acc_y,acc_z",
            )
            self.assertEqual(
                imu_csv[1],
                "1,0.000000,1.000000,0.000000,0.000000,4.000000,5.000000,6.000000",
            )
            self.assertEqual(
                imu_csv[2],
                "2,0.000000,1.000000,0.000000,0.000000,13.000000,14.000000,15.000000",
            )
            self.assertEqual(len(imu_csv), 3)

            smplx = np.load(output_dir / "s1_freestyle3_smplx.npz", allow_pickle=True)
            self.assertEqual(str(smplx["surface_model_type"]), "smplx")
            self.assertEqual(smplx["poses"].shape, (2, 165))
            self.assertEqual(smplx["pose_hand"].shape, (2, 90))
            self.assertEqual(smplx["pose_body"].shape, (2, 63))

    def test_cli_script_runs_from_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "totalcapture_test" / "prepare_sample.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            raw_root = tmp / "raw_totalcapture"
            stageii_root = tmp / "amass_totalcapture"
            data_root = tmp / "repo_data"

            (raw_root / "freestyle3").mkdir(parents=True)
            (raw_root / "s1").mkdir(parents=True)
            (stageii_root / "s1").mkdir(parents=True)

            (raw_root / "freestyle3" / "TC_S1_freestyle3_cam1.mp4").write_bytes(b"fake video")
            self._write_fake_xsens(raw_root / "s1" / "s1_freestyle3_Xsens.sensors")
            self._write_fake_stageii(stageii_root / "s1" / "freestyle3_stageii.npz")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--raw-totalcapture-root",
                    str(raw_root),
                    "--stageii-totalcapture-root",
                    str(stageii_root),
                    "--data-root",
                    str(data_root),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertIn("S1_freestyle3", completed.stdout)


if __name__ == "__main__":
    unittest.main()
