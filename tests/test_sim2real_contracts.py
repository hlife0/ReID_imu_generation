from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ConfigIdentityTest(unittest.TestCase):
    def test_config_hash_is_key_order_invariant(self) -> None:
        from src.sim2real.contracts import config_hash

        a = {"generator": "globalpose", "switches": {"sensor_noise": True}, "sensor": "R_LowArm"}
        b = {"sensor": "R_LowArm", "generator": "globalpose", "switches": {"sensor_noise": True}}
        self.assertEqual(config_hash(a), config_hash(b))
        self.assertEqual(len(config_hash(a)), 8)
        self.assertNotEqual(config_hash(a), config_hash({**a, "sensor": "L_LowArm"}))

    def test_synth_source_roundtrip_and_token(self) -> None:
        from src.sim2real.contracts import parse_source, source_to_token, synth_source

        cfg = {"generator": "globalpose", "switches": {}}
        source = synth_source("globalpose", cfg)
        parsed = parse_source(source)
        self.assertEqual(parsed["kind"], "synth")
        self.assertEqual(parsed["generator"], "globalpose")
        self.assertEqual(len(parsed["config_hash"]), 8)
        self.assertEqual(source_to_token(source), f"synth_globalpose_{parsed['config_hash']}")
        self.assertEqual(source_to_token("real"), "real")
        with self.assertRaises(ValueError):
            parse_source("synthetic/globalpose/a3f2c1d0")


class SequenceRoundtripTest(unittest.TestCase):
    def test_motion_sequence_roundtrip(self) -> None:
        from src.sim2real.contracts import MotionSequence

        rng = np.random.default_rng(0)
        joints = rng.standard_normal((50, 17, 3)).astype(np.float32)
        motion = MotionSequence(
            joints=joints,
            fps=60.0,
            joint_layout="pipeline_world_v1",
            meta={"sequence": "S1_freestyle3", "subject": "S1"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = motion.save(Path(tmpdir) / "motion.npz")
            loaded = MotionSequence.load(path)
        np.testing.assert_array_equal(loaded.joints, joints)
        self.assertEqual(loaded.fps, 60.0)
        self.assertEqual(loaded.joint_layout, "pipeline_world_v1")
        self.assertEqual(loaded.meta["sequence"], "S1_freestyle3")
        self.assertAlmostEqual(loaded.duration_s, 50 / 60.0)

    def test_motion_sequence_rejects_bad_shapes_and_nan(self) -> None:
        from src.sim2real.contracts import MotionSequence

        with self.assertRaises(ValueError):
            MotionSequence(joints=np.zeros((10, 17, 2)), fps=60.0, joint_layout="x")
        bad = np.zeros((10, 17, 3), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            MotionSequence(joints=bad, fps=60.0, joint_layout="x")
        with self.assertRaises(ValueError):
            MotionSequence(joints=np.zeros((10, 17, 3)), fps=0.0, joint_layout="x")

    def test_imu_sequence_roundtrip(self) -> None:
        from src.sim2real.contracts import ImuSequence, synth_source

        rng = np.random.default_rng(1)
        data = rng.standard_normal((120, 6)).astype(np.float32)
        channels = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
        source = synth_source("globalpose", {"switches": {}})
        imu = ImuSequence(
            data=data,
            channels=channels,
            fps=60.0,
            source=source,
            sensor="R_LowArm",
            meta={"sequence": "S1_freestyle3"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = imu.save(Path(tmpdir) / "imu" / "synth__globalpose__abc.npz")
            loaded = ImuSequence.load(path)
        np.testing.assert_array_equal(loaded.data, data)
        self.assertEqual(loaded.channels, channels)
        self.assertEqual(loaded.source, source)
        self.assertEqual(loaded.sensor, "R_LowArm")

    def test_imu_sequence_rejects_channel_mismatch_and_bad_source(self) -> None:
        from src.sim2real.contracts import ImuSequence

        data = np.zeros((10, 6), dtype=np.float32)
        with self.assertRaises(ValueError):
            ImuSequence(data=data, channels=("a", "b"), fps=60.0, source="real", sensor="R_LowArm")
        channels = ("a", "b", "c", "d", "e", "f")
        with self.assertRaises(ValueError):
            ImuSequence(data=data, channels=channels, fps=60.0, source="fake", sensor="R_LowArm")
        with self.assertRaises(ValueError):
            ImuSequence(
                data=data, channels=("a",) * 6, fps=60.0, source="real", sensor="R_LowArm"
            )


class ManifestTest(unittest.TestCase):
    def test_manifest_write_and_load(self) -> None:
        from src.sim2real.contracts import config_hash, load_manifest, write_manifest

        cfg = {"generator": "globalpose", "switches": {"sensor_noise": True}}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_file = tmp / "motion.npz"
            input_file.write_bytes(b"not-a-real-npz")

            written = write_manifest(
                tmp / "artifact",
                stage="generate",
                config=cfg,
                inputs={"motion": input_file, "gone": tmp / "missing.npz"},
                seed=42,
                extra={"sequence": "S1_freestyle3"},
            )
            loaded = load_manifest(tmp / "artifact")

        self.assertEqual(loaded, json.loads(json.dumps(written)))
        self.assertEqual(loaded["stage"], "generate")
        self.assertEqual(loaded["seed"], 42)
        self.assertEqual(loaded["config_hash"], config_hash(cfg))
        self.assertEqual(loaded["inputs"]["motion"]["bytes"], len(b"not-a-real-npz"))
        self.assertIn("sha1", loaded["inputs"]["motion"])
        self.assertTrue(loaded["inputs"]["gone"]["missing"])
        self.assertEqual(loaded["extra"]["sequence"], "S1_freestyle3")


if __name__ == "__main__":
    unittest.main()
