from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


class GlobalPoseOriginAdapterTest(unittest.TestCase):
    def test_parses_totalcapture_sensor_stream_and_gt_trajectory(self) -> None:
        from src.globalpose_origin_adapter import (
            GLOBALPOSE_SENSOR_ORDER,
            build_globalpose_sensor_trajectories,
            parse_totalcapture_gt_orientations,
            parse_totalcapture_gt_positions,
            parse_totalcapture_sensor_stream,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sensors_path = tmp / "sample_Xsens.sensors"
            positions_path = tmp / "gt_skel_gbl_pos.txt"
            orientations_path = tmp / "gt_skel_gbl_ori.txt"

            sensors_path.write_text(
                "\n".join(
                    [
                        "3\t2",
                        "1",
                        "Head\t1\t0\t0\t0\t1\t2\t3\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6",
                        "R_LowArm\t0\t1\t0\t0\t4\t5\t6\t0.7\t0.8\t0.9\t1.0\t1.1\t1.2",
                        "Pelvis\t0\t0\t1\t0\t7\t8\t9\t1.3\t1.4\t1.5\t1.6\t1.7\t1.8",
                        "2",
                        "Head\t1\t0\t0\t0\t10\t11\t12\t1.9\t2.0\t2.1\t2.2\t2.3\t2.4",
                        "R_LowArm\t0\t1\t0\t0\t13\t14\t15\t2.5\t2.6\t2.7\t2.8\t2.9\t3.0",
                        "Pelvis\t0\t0\t1\t0\t16\t17\t18\t3.1\t3.2\t3.3\t3.4\t3.5\t3.6",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            positions_path.write_text(
                "\n".join(
                    [
                        "Hips\tHead\tRightForeArm\tLeftForeArm\tRightLeg\tLeftLeg\t",
                        "1 2 3\t4 5 6\t7 8 9\t10 11 12\t13 14 15\t16 17 18\t",
                        "2 3 4\t5 6 7\t8 9 10\t11 12 13\t14 15 16\t17 18 19\t",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            orientations_path.write_text(
                "\n".join(
                    [
                        "Hips\tHead\tRightForeArm\tLeftForeArm\tRightLeg\tLeftLeg\t",
                        "1 0 0 0\t0 1 0 0\t0 0 1 0\t0 0 0 1\t0.5 0.5 0.5 0.5\t0.5 -0.5 0.5 -0.5\t",
                        "1 0 0 0\t0 1 0 0\t0 0 1 0\t0 0 0 1\t0.5 0.5 0.5 0.5\t0.5 -0.5 0.5 -0.5\t",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            sensor_stream = parse_totalcapture_sensor_stream(sensors_path, sensor_names=["Head", "R_LowArm", "Pelvis"])
            gt_positions = parse_totalcapture_gt_positions(positions_path)
            gt_orientations = parse_totalcapture_gt_orientations(orientations_path)
            trajectories = build_globalpose_sensor_trajectories(gt_positions, gt_orientations)

        self.assertEqual(sensor_stream["frame_count"], 2)
        self.assertEqual(sensor_stream["sensor_names"], ["Head", "R_LowArm", "Pelvis"])
        self.assertEqual(sensor_stream["quat"].shape, (2, 3, 4))
        self.assertEqual(sensor_stream["acc"].shape, (2, 3, 3))
        self.assertTrue(np.allclose(sensor_stream["acc"][0, 1], [4.0, 5.0, 6.0]))
        self.assertTrue(np.allclose(sensor_stream["mag"][1, 2], [3.4, 3.5, 3.6]))

        self.assertTrue(np.allclose(gt_positions["Hips"][0], [-0.0254, 0.0508, -0.0762], atol=1e-6))
        self.assertTrue(np.allclose(gt_positions["RightForeArm"][1], [-0.2032, 0.2286, -0.254], atol=1e-6))
        self.assertEqual(gt_orientations["Head"].shape, (2, 4))

        self.assertEqual(list(trajectories.keys()), list(GLOBALPOSE_SENSOR_ORDER))
        self.assertEqual(trajectories["R_LowArm"]["positions"].shape, (2, 3))
        self.assertEqual(trajectories["Pelvis"]["quaternions"].shape, (2, 4))

    def test_enforces_quaternion_sign_continuity(self) -> None:
        from src.globalpose_origin_adapter import enforce_quaternion_continuity

        quaternions = np.asarray(
            [
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )

        fixed = enforce_quaternion_continuity(quaternions)

        self.assertTrue(np.allclose(fixed[0], [0.5, 0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(fixed[1], [0.5, 0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(fixed[2], [0.5, 0.5, 0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
