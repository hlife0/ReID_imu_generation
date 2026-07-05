from __future__ import annotations

import unittest

import numpy as np


class WindowingDeterminismTest(unittest.TestCase):
    def test_window_starts(self) -> None:
        from src.sim2real.windows import window_starts

        self.assertEqual(window_starts(100, 24, 16), [0, 16, 32, 48, 64])
        self.assertEqual(window_starts(24, 24, 16), [0])
        self.assertEqual(window_starts(23, 24, 16), [])

    def test_extract_windows_deterministic_and_correct(self) -> None:
        from src.sim2real.windows import extract_windows, window_starts

        data = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)
        starts = window_starts(50, 10, 8)
        w1 = extract_windows(data, starts, 10)
        w2 = extract_windows(data, starts, 10)
        np.testing.assert_array_equal(w1, w2)
        self.assertEqual(w1.shape, (len(starts), 10, 3))
        np.testing.assert_array_equal(w1[1], data[8:18])

    def test_center_motion_windows_removes_offset_keeps_dynamics(self) -> None:
        from src.sim2real.windows import center_motion_windows

        rng = np.random.default_rng(0)
        base = rng.standard_normal((5, 24, 17, 3)).astype(np.float32)
        shifted = base + np.array([100.0, -50.0, 7.0], dtype=np.float32)
        c_base = center_motion_windows(base.copy())
        c_shifted = center_motion_windows(shifted.copy())
        # absolute location removed: same windows after centering
        np.testing.assert_allclose(c_base, c_shifted, atol=1e-3)
        # within-window dynamics preserved: frame-to-frame deltas unchanged
        np.testing.assert_allclose(np.diff(c_base, axis=1), np.diff(base, axis=1), atol=1e-5)

    def test_channel_stats(self) -> None:
        from src.sim2real.windows import channel_stats

        windows = np.zeros((4, 10, 2), dtype=np.float32)
        windows[..., 0] = 3.0  # constant channel -> std clamped to 1
        stats = channel_stats(windows)
        self.assertAlmostEqual(stats["mean"][0], 3.0)
        self.assertAlmostEqual(stats["std"][0], 1.0)
        self.assertAlmostEqual(stats["mean"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
