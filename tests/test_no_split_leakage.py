from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SPLIT = REPO_ROOT / "configs" / "sim2real" / "splits" / "totalcapture_subject_v1.json"


class FrozenSplitTest(unittest.TestCase):
    def test_frozen_split_is_valid_and_subject_disjoint(self) -> None:
        from src.sim2real.splits import load_split

        spec = load_split(FROZEN_SPLIT)
        self.assertEqual(spec.name, "totalcapture_subject_v1")
        self.assertEqual(spec.dataset, "totalcapture")
        self.assertEqual(spec.train_subjects, ("S1", "S2", "S3"))
        self.assertEqual(spec.val_subjects, ("S4",))
        self.assertEqual(spec.test_subjects, ("S5",))
        # disjointness is enforced by load_split; reaching here means it held
        self.assertEqual(len(spec.all_subjects()), 5)

    def test_split_of_sequence(self) -> None:
        from src.sim2real.splits import load_split

        spec = load_split(FROZEN_SPLIT)
        self.assertEqual(spec.split_of_sequence("S1_freestyle3"), "train")
        self.assertEqual(spec.split_of_sequence("s4_walking1"), "val")
        self.assertEqual(spec.split_of_sequence("S5_acting2"), "test")
        with self.assertRaises(KeyError):
            spec.split_of_sequence("S9_freestyle1")


class SubjectParsingTest(unittest.TestCase):
    def test_subject_of_sequence(self) -> None:
        from src.sim2real.splits import subject_of_sequence

        self.assertEqual(subject_of_sequence("S1_freestyle3"), "S1")
        self.assertEqual(subject_of_sequence("s2_acting1"), "S2")
        self.assertEqual(subject_of_sequence("S10"), "S10")
        with self.assertRaises(ValueError):
            subject_of_sequence("freestyle3_S1")

    def test_shard_split(self) -> None:
        from src.sim2real.splits import shard_split

        self.assertEqual(shard_split("train__synth_globalpose_a3f2c1d0"), "train")
        self.assertEqual(shard_split("test__real.npz"), "test")
        with self.assertRaises(ValueError):
            shard_split("training__real")


class LeakageCheckTest(unittest.TestCase):
    def _spec(self):
        from src.sim2real.splits import load_split

        return load_split(FROZEN_SPLIT)

    def test_clean_shards_pass(self) -> None:
        from src.sim2real.splits import find_leakage

        shards = {
            "train__real": {"S1", "S2", "S3"},
            "train__synth_globalpose_a3f2c1d0": {"S1", "S2", "S3"},
            "val__real": {"S4"},
            "test__real": {"S5"},
        }
        self.assertEqual(find_leakage(self._spec(), shards), [])

    def test_planted_test_subject_in_train_shard_is_caught(self) -> None:
        from src.sim2real.splits import find_leakage

        shards = {
            "train__synth_globalpose_a3f2c1d0": {"S1", "S2", "S5"},
            "test__real": {"S5"},
        }
        violations = find_leakage(self._spec(), shards)
        self.assertEqual(len(violations), 1)
        self.assertIn("LEAKAGE", violations[0])
        self.assertIn("S5", violations[0])

    def test_unknown_subject_and_bad_shard_name_are_flagged(self) -> None:
        from src.sim2real.splits import find_leakage

        shards = {
            "train__real": {"S9"},
            "holdout__real": {"S5"},
        }
        violations = find_leakage(self._spec(), shards)
        self.assertEqual(len(violations), 2)
        self.assertTrue(any("S9" in v for v in violations))
        self.assertTrue(any("holdout__real" in v for v in violations))


if __name__ == "__main__":
    unittest.main()
