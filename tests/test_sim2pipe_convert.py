"""Unit tests for sim2pipe conversions (mirrors of the main project's contracts)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2pipe.convert import (
    ACC_CHANNELS,
    QUAT_CHANNELS,
    imu_to_48,
    normalize_skeleton_pipe,
    quat_wxyz_to_rotmat,
)

CORPUS_CHANNELS = (
    "quat0", "quat1", "quat2", "quat3",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
)


class TestQuatToRotmat:
    def test_identity(self):
        r = quat_wxyz_to_rotmat(np.array([[1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(r[0], np.eye(3), atol=1e-6)

    def test_180deg_about_z(self):
        # wxyz = (0, 0, 0, 1): x -> -x, y -> -y, z -> z
        r = quat_wxyz_to_rotmat(np.array([[0.0, 0.0, 0.0, 1.0]]))[0]
        np.testing.assert_allclose(r, np.diag([-1.0, -1.0, 1.0]), atol=1e-6)

    def test_90deg_about_z_rotates_x_to_y(self):
        h = np.sqrt(0.5)
        r = quat_wxyz_to_rotmat(np.array([[h, 0.0, 0.0, h]]))[0]
        np.testing.assert_allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)

    def test_unit_quats_give_proper_rotations(self):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(32, 4)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        r = quat_wxyz_to_rotmat(q)
        eye = np.broadcast_to(np.eye(3, dtype=np.float32), r.shape)
        np.testing.assert_allclose(r @ np.swapaxes(r, 1, 2), eye, atol=1e-5)
        np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-5)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            quat_wxyz_to_rotmat(np.zeros((5, 3)))


class TestImuTo48:
    def _make(self, T=7, channels=CORPUS_CHANNELS, seed=1):
        rng = np.random.default_rng(seed)
        data = rng.normal(size=(T, len(channels))).astype(np.float32)
        quat_cols = [list(channels).index(c) for c in QUAT_CHANNELS]
        data[:, quat_cols] /= np.linalg.norm(data[:, quat_cols], axis=1, keepdims=True)
        return data

    def test_layout_rotation_first_then_acc(self):
        data = self._make()
        out = imu_to_48(data, CORPUS_CHANNELS)
        assert out.shape == (data.shape[0], 48)
        assert out.dtype == np.float32

        quat = data[:, :4]
        acc = data[:, 4:7]
        rot9 = quat_wxyz_to_rotmat(quat).reshape(-1, 9)
        for i in range(4):
            np.testing.assert_allclose(out[:, i * 9 : (i + 1) * 9], rot9, atol=1e-6)
            np.testing.assert_allclose(out[:, 36 + i * 3 : 36 + (i + 1) * 3], acc, atol=1e-6)

    def test_column_order_independent(self):
        data = self._make()
        perm = np.random.default_rng(2).permutation(len(CORPUS_CHANNELS))
        shuffled_channels = tuple(CORPUS_CHANNELS[i] for i in perm)
        shuffled_data = data[:, perm]
        np.testing.assert_allclose(
            imu_to_48(data, CORPUS_CHANNELS),
            imu_to_48(shuffled_data, shuffled_channels),
            atol=1e-6,
        )

    def test_missing_channel_raises(self):
        channels = tuple(c for c in CORPUS_CHANNELS if c != "acc_y")
        with pytest.raises(KeyError, match="acc_y"):
            imu_to_48(np.zeros((3, len(channels)), dtype=np.float32), channels)


class TestNormalizeSkeletonPipe:
    def test_root_relative_and_unit_scale(self):
        rng = np.random.default_rng(3)
        skel = rng.normal(size=(11, 17, 3)).astype(np.float32) * 100.0
        out = normalize_skeleton_pipe(skel)
        np.testing.assert_allclose(out[:, 0, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(out[:, 8, :], axis=-1), 1.0, atol=1e-5)

    def test_translation_and_scale_invariant(self):
        rng = np.random.default_rng(4)
        skel = rng.normal(size=(5, 17, 3)).astype(np.float32)
        shifted = skel * 3.5 + np.array([10.0, -2.0, 7.0], dtype=np.float32)
        np.testing.assert_allclose(
            normalize_skeleton_pipe(skel), normalize_skeleton_pipe(shifted), atol=1e-4
        )

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            normalize_skeleton_pipe(np.zeros((5, 21, 3)))


def test_channel_constants_cover_corpus():
    assert set(QUAT_CHANNELS) | set(ACC_CHANNELS) <= set(CORPUS_CHANNELS)
