"""Unit tests for src/sim2real/alignment.py (lag estimation + aligned slices)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.alignment import aligned_slices, estimate_lag, lag_from_meta


class AlignedSlicesTest(unittest.TestCase):
    def test_zero_lag_equals_head_align(self) -> None:
        sl_i, sl_m = aligned_slices(48, 50, 0)
        self.assertEqual((sl_i, sl_m), (slice(0, 48), slice(0, 48)))

    def test_positive_lag_shifts_motion(self) -> None:
        # real[i] <-> motion[i+44]; real shorter: motion tail is used
        sl_i, sl_m = aligned_slices(3561, 3605, 44)
        self.assertEqual(sl_i, slice(0, 3561))
        self.assertEqual(sl_m, slice(44, 3605))

    def test_negative_lag_shifts_imu(self) -> None:
        sl_i, sl_m = aligned_slices(100, 90, -5)
        self.assertEqual(sl_i, slice(5, 95))
        self.assertEqual(sl_m, slice(0, 90))

    def test_equal_slice_lengths_always(self) -> None:
        for len_i, len_m, lag in [(10, 10, 3), (10, 20, -7), (33, 21, 0), (5, 5, -4)]:
            sl_i, sl_m = aligned_slices(len_i, len_m, lag)
            self.assertEqual(sl_i.stop - sl_i.start, sl_m.stop - sl_m.start)
            self.assertGreaterEqual(sl_i.start, 0)
            self.assertLessEqual(sl_i.stop, len_i)
            self.assertLessEqual(sl_m.stop, len_m)

    def test_empty_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            aligned_slices(10, 10, 10)
        with self.assertRaises(ValueError):
            aligned_slices(10, 10, -10)


class EstimateLagTest(unittest.TestCase):
    def _signal(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # smooth band-limited signal so shifted correlation has a sharp peak
        x = rng.standard_normal(n)
        kernel = np.hanning(9)
        return np.convolve(x, kernel / kernel.sum(), mode="same")

    def test_recovers_known_positive_lag(self) -> None:
        motion = self._signal(2000)
        imu = motion[30:1900]  # imu[i] == motion[i+30]
        lag, corr = estimate_lag(imu, motion, max_lag=60)
        self.assertEqual(lag, 30)
        self.assertGreater(corr, 0.99)

    def test_recovers_known_negative_lag(self) -> None:
        imu_full = self._signal(2000, seed=1)
        imu = imu_full  # motion[i] == imu[i+12]  =>  imu[i] <-> motion[i-12]
        motion = imu_full[12:1900]
        lag, corr = estimate_lag(imu, motion, max_lag=60)
        self.assertEqual(lag, -12)
        self.assertGreater(corr, 0.99)

    def test_zero_lag_with_noise(self) -> None:
        motion = self._signal(1500, seed=2)
        rng = np.random.default_rng(3)
        imu = motion[:1400] + 0.05 * rng.standard_normal(1400)
        lag, corr = estimate_lag(imu, motion, max_lag=60)
        self.assertEqual(lag, 0)
        self.assertGreater(corr, 0.9)

    def test_constant_signal_raises(self) -> None:
        with self.assertRaises(ValueError):
            estimate_lag(np.ones(100), np.ones(100), max_lag=10)


class LagFromMetaTest(unittest.TestCase):
    def test_dict_alignment(self) -> None:
        meta = {"alignment": {"method": "naive_bridge_lagscan_v1", "imu_motion_lag": 44}}
        self.assertEqual(lag_from_meta(meta), 44)

    def test_legacy_string_and_missing(self) -> None:
        self.assertIsNone(lag_from_meta({"alignment": "tail"}))
        self.assertIsNone(lag_from_meta({}))


if __name__ == "__main__":
    unittest.main()
