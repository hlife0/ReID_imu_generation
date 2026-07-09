"""IMU generation protocol conformance — the naive reference must produce a
valid 13-channel ImuSequence from a standardized MotionSequence input.

See ``docs/imu_generation_protocol.md``.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sim2real.contracts import IMU_CHANNELS_13, ImuSequence, MotionSequence, parse_source
from sim2real.gen_common import load_motion_and_trajectory, write_synth_stream


def _load_naive_synthesize():
    path = REPO_ROOT / "scripts/sim2real/generators/naive/generate.py"
    spec = importlib.util.spec_from_file_location("naive_generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthesize


def _toy_motion(num_frames: int = 40) -> MotionSequence:
    rng = np.random.default_rng(0)
    joints = np.cumsum(rng.normal(scale=0.01, size=(num_frames, 21, 3)), axis=0).astype(np.float32)
    return MotionSequence(joints=joints, fps=60.0, joint_layout="data_generation_pipeline_v1")


class GenerationProtocolTest(unittest.TestCase):
    def test_naive_reference_conforms(self) -> None:
        synthesize = _load_naive_synthesize()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            motion_path = _toy_motion().save(tmp / "motion.npz")
            motion, positions, quat_wxyz = load_motion_and_trajectory(motion_path, "R_LowArm")

            out = synthesize(motion, positions, quat_wxyz, {"generator": "naive"}, 0)
            for key in ("quat", "acc", "gyro", "mag", "fps"):
                self.assertIn(key, out)

            receipt = write_synth_stream(
                tmp, generator="naive", config={"generator": "naive"},
                config_path=tmp / "cfg.json", motion_path=motion_path, seed=0,
                sensor="R_LowArm", fps=out["fps"], quat=out["quat"], acc=out["acc"],
                gyro=out["gyro"], mag=out["mag"], extra_meta=out.get("extra_meta", {}),
            )

            imu = ImuSequence.load(receipt["npz"])
            self.assertEqual(imu.channels, IMU_CHANNELS_13)
            self.assertEqual(imu.data.shape[1], 13)
            self.assertEqual(imu.fps, 60.0)
            self.assertEqual(imu.sensor, "R_LowArm")
            self.assertEqual(parse_source(imu.source)["generator"], "naive")
            self.assertTrue(np.isfinite(imu.data).all())

    def test_naive_is_deterministic(self) -> None:
        synthesize = _load_naive_synthesize()
        motion = _toy_motion()
        positions, quat = __import__("sim2real.geom", fromlist=["compute_sensor_trajectory"]).compute_sensor_trajectory(
            motion.joints, motion.joint_layout, "wrist_right"
        )
        a = synthesize(motion, positions.astype(np.float32), quat.astype(np.float32), {}, 0)
        b = synthesize(motion, positions.astype(np.float32), quat.astype(np.float32), {}, 7)
        self.assertTrue(np.array_equal(a["acc"], b["acc"]))  # seed-independent by construction


if __name__ == "__main__":
    unittest.main()
