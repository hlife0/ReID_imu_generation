"""Contract tests: export produces the main project's unified npz schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.contracts import ImuSequence, MotionSequence
from src.sim2pipe.export import export_corpus, export_sequence, split_corpus_dirname, stream_token
from src.sim2pipe.ledger import append_row, cell_key, done_keys

CHANNELS = (
    "quat0", "quat1", "quat2", "quat3",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
)


def _make_corpus_seq(root: Path, name: str, t_motion: int = 50, t_imu: int = 48) -> Path:
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    seq_dir = root / name
    (seq_dir / "imu").mkdir(parents=True)

    joints21 = rng.normal(size=(t_motion, 21, 3)).astype(np.float32)
    MotionSequence(
        joints=joints21, fps=60.0, joint_layout="data_generation_pipeline_v1"
    ).save(seq_dir / "motion.npz")
    joints17 = rng.normal(size=(t_motion, 17, 3)).astype(np.float32)
    MotionSequence(
        joints=joints17, fps=60.0, joint_layout="estimated_h36m17_v1"
    ).save(seq_dir / "motion_estimated.npz")

    data = rng.normal(size=(t_imu, len(CHANNELS))).astype(np.float32)
    data[:, :4] /= np.linalg.norm(data[:, :4], axis=1, keepdims=True)
    for fname, source in [("real.npz", "real"), ("synth_naive_8f7d9e76.npz", "synth/naive/8f7d9e76")]:
        ImuSequence(
            data=data, channels=CHANNELS, fps=60.0, source=source, sensor="R_LowArm"
        ).save(seq_dir / "imu" / fname)
    return seq_dir


UNIFIED_KEYS = {
    "video_path", "dataset", "sequence_id", "frame_ids",
    "imu", "imu_ids", "gt_person_ids", "gt_bboxes", "gt_visibility",
    "gt_skeleton", "gt_skeleton_meters",
}


class TestExportSequence:
    @pytest.fixture()
    def corpus(self, tmp_path):
        root = tmp_path / "corpus"
        _make_corpus_seq(root, "S1_acting1")
        return root

    @pytest.mark.parametrize("motion_source", ["motion", "motion_estimated"])
    def test_unified_schema(self, corpus, tmp_path, motion_source):
        rec = export_sequence(corpus / "S1_acting1", "real.npz", motion_source, tmp_path / "out")
        assert rec.sequence_id == "totalcapture_S1_acting1_cam1"
        assert rec.n_frames == 48  # min(motion 50, imu 48)

        data = dict(np.load(rec.npz_path, allow_pickle=True))
        assert set(data.keys()) == UNIFIED_KEYS
        T = rec.n_frames
        assert data["imu"].shape == (T, 1, 48) and data["imu"].dtype == np.float32
        assert data["gt_skeleton"].shape == (T, 1, 17, 3)
        assert data["gt_skeleton_meters"].shape == (T, 1, 17, 3)
        assert data["gt_bboxes"].shape == (T, 1, 4)
        assert data["gt_visibility"].shape == (T, 1) and data["gt_visibility"].dtype == bool
        assert data["frame_ids"].shape == (T,) and data["frame_ids"].dtype == np.int64
        assert str(data["sequence_id"].item()) == rec.sequence_id
        # gt_skeleton is normalized: root at origin, |thorax - root| == 1
        skel = data["gt_skeleton"][:, 0]
        np.testing.assert_allclose(skel[:, 0, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(skel[:, 8, :], axis=-1), 1.0, atol=1e-4)

        meta = json.loads(rec.npz_path.with_suffix(".json").read_text())
        assert meta["sequence_id"] == rec.sequence_id
        assert meta["n_frames"] == T
        assert meta["sim2pipe"]["motion_source"] == motion_source

    def test_sequence_id_parsing(self):
        assert split_corpus_dirname("S1_acting1") == ("S1", "acting1")
        assert split_corpus_dirname("S3_walking2_take1") == ("S3", "walking2_take1")
        with pytest.raises(ValueError):
            split_corpus_dirname("acting1")


class TestExportCorpus:
    def test_export_and_manifest(self, tmp_path):
        root = tmp_path / "corpus"
        _make_corpus_seq(root, "S1_acting1")
        _make_corpus_seq(root, "S5_walking2")
        # a sequence without the synth stream is skipped, not fatal
        odd = _make_corpus_seq(root, "S4_rom3")
        (odd / "imu" / "synth_naive_8f7d9e76.npz").unlink()

        out_dir, exported = export_corpus(
            root, "synth_naive_8f7d9e76.npz", "motion_estimated", tmp_path / "export"
        )
        assert out_dir == tmp_path / "export" / "motion_estimated" / "synth_naive_8f7d9e76"
        assert [e.sequence_id for e in exported] == [
            "totalcapture_S1_acting1_cam1",
            "totalcapture_S5_walking2_cam1",
        ]
        manifest = json.loads((out_dir / "export_manifest.json").read_text())
        assert manifest["skipped_sequences"] == ["S4_rom3"]
        assert manifest["n_sequences"] == 2

    def test_missing_stream_everywhere_raises(self, tmp_path):
        root = tmp_path / "corpus"
        _make_corpus_seq(root, "S1_acting1")
        with pytest.raises(FileNotFoundError):
            export_corpus(root, "synth_ghost_00000000.npz", "motion", tmp_path / "export")

    def test_stream_token(self):
        assert stream_token("real.npz") == "real"
        assert stream_token("synth_naive_8f7d9e76.npz") == "synth_naive_8f7d9e76"


class TestLedger:
    def test_append_done_resume(self, tmp_path):
        ledger = tmp_path / "results.jsonl"
        row = {
            "protocol": "tstr", "imu_stream": "synth_naive_8f7d9e76",
            "motion_source": "motion_estimated", "seed": 0, "val_top1": 0.1,
        }
        append_row(ledger, row)
        append_row(ledger, {**row, "seed": 42})
        assert cell_key(row) in done_keys(ledger)
        assert len(done_keys(ledger)) == 2

    def test_rejects_incomplete_row(self, tmp_path):
        with pytest.raises(KeyError):
            append_row(tmp_path / "results.jsonl", {"protocol": "tstr"})
