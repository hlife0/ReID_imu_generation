"""Export sim2real corpus streams into the main project's unified npz schema.

One export root per (imu stream, motion source) pair; the main repo's
``preprocess.imu_source: synthetic`` path simply copies every ``*.npz``
(+ sidecar ``.json``) from ``synthetic_imu_root`` into its sequences dir,
so each exported file must be a complete unified-schema sequence:

    video_path, dataset, sequence_id, frame_ids,
    imu [T, 1, 48], imu_ids [1],
    gt_person_ids [1], gt_bboxes [T, 1, 4], gt_visibility [T, 1],
    gt_skeleton [T, 1, 17, 3] (normalized), gt_skeleton_meters [T, 1, 17, 3]

Sequence ids follow the main repo's ``totalcapture_{subject}_{session}_{camera}``
convention so its subject-level split config applies unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.sim2real.contracts import ImuSequence, MotionSequence
from src.sim2real.gen_estimated import retarget_21_to_h36m17
from src.sim2pipe.convert import imu_to_48, normalize_skeleton_pipe

MOTION_SOURCES = ("motion", "motion_estimated")
_SEQ_DIR_RE = re.compile(r"^(S\d+)_(.+)$")


@dataclass(frozen=True)
class ExportedSequence:
    sequence_id: str
    npz_path: Path
    n_frames: int


def split_corpus_dirname(name: str) -> tuple[str, str]:
    """'S1_acting1' -> ('S1', 'acting1')."""
    m = _SEQ_DIR_RE.match(name)
    if not m:
        raise ValueError(f"corpus dir name {name!r} does not match S<i>_<session>")
    return m.group(1), m.group(2)


def _load_skeleton17(seq_dir: Path, motion_source: str) -> tuple[np.ndarray, float]:
    if motion_source not in MOTION_SOURCES:
        raise ValueError(f"motion_source must be one of {MOTION_SOURCES}, got {motion_source!r}")
    motion = MotionSequence.load(seq_dir / f"{motion_source}.npz")
    joints = motion.joints
    if joints.shape[1] == 21:
        joints = retarget_21_to_h36m17(joints)
    elif joints.shape[1] != 17:
        raise ValueError(f"{seq_dir.name}/{motion_source}: expected 17 or 21 joints, got {joints.shape[1]}")
    return joints.astype(np.float32), motion.fps


def export_sequence(
    seq_dir: Path,
    imu_filename: str,
    motion_source: str,
    out_dir: Path,
    camera: str = "cam1",
) -> ExportedSequence:
    """Export one corpus sequence x one IMU stream to a unified-schema npz."""
    seq_dir = Path(seq_dir)
    subject, session = split_corpus_dirname(seq_dir.name)

    imu = ImuSequence.load(seq_dir / "imu" / imu_filename)
    skel17, motion_fps = _load_skeleton17(seq_dir, motion_source)
    if abs(imu.fps - motion_fps) > 1e-6:
        raise ValueError(
            f"{seq_dir.name}: imu fps {imu.fps} != motion fps {motion_fps}; "
            "the main repo consumes both 1:1 with no resampling"
        )

    imu48 = imu_to_48(imu.data, imu.channels)
    tlen = min(imu48.shape[0], skel17.shape[0])
    if tlen < 1:
        raise ValueError(f"{seq_dir.name}: empty overlap between imu and motion")
    imu48 = imu48[:tlen]
    skel17 = skel17[:tlen]

    sequence_id = f"totalcapture_{subject}_{session}_{camera}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sequence_id}.npz"

    np.savez_compressed(
        npz_path,
        video_path=np.array("", dtype=object),
        dataset=np.array("totalcapture", dtype=object),
        sequence_id=np.array(sequence_id, dtype=object),
        frame_ids=np.arange(tlen, dtype=np.int64),
        imu=imu48[:, np.newaxis, :].astype(np.float32),
        imu_ids=np.array([0], dtype=np.int64),
        gt_person_ids=np.array([0], dtype=np.int64),
        gt_bboxes=np.zeros((tlen, 1, 4), dtype=np.float32),
        gt_visibility=np.ones((tlen, 1), dtype=bool),
        gt_skeleton=normalize_skeleton_pipe(skel17)[:, np.newaxis, :, :],
        gt_skeleton_meters=skel17[:, np.newaxis, :, :].astype(np.float32),
    )
    meta = {
        "video_path": "",
        "dataset": "totalcapture",
        "sequence_id": sequence_id,
        "n_frames": int(tlen),
        "n_imu": 1,
        "n_gt": 1,
        "n_pred": 0,
        "has_gt": True,
        "imu_ids": [0],
        "gt_person_ids": [0],
        "extract_person_ids": [],
        "sim2pipe": {
            "imu_source": imu.source,
            "imu_stream_file": imu_filename,
            "motion_source": motion_source,
            "sensor": imu.sensor,
            "fps": imu.fps,
        },
    }
    npz_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return ExportedSequence(sequence_id=sequence_id, npz_path=npz_path, n_frames=tlen)


def stream_token(imu_filename: str) -> str:
    """'synth_naive_8f7d9e76.npz' -> 'synth_naive_8f7d9e76'; 'real.npz' -> 'real'."""
    return Path(imu_filename).stem


def export_corpus(
    corpus_root: Path,
    imu_filename: str,
    motion_source: str,
    export_root: Path,
    camera: str = "cam1",
) -> tuple[Path, list[ExportedSequence]]:
    """Export every corpus sequence that carries ``imu_filename``.

    Returns the export dir (= the main repo's ``synthetic_imu_root``) and
    the per-sequence records. Sequences missing this stream are skipped
    and reported by the caller via the manifest.
    """
    corpus_root = Path(corpus_root)
    out_dir = Path(export_root) / motion_source / stream_token(imu_filename)
    exported: list[ExportedSequence] = []
    skipped: list[str] = []
    for seq_dir in sorted(p for p in corpus_root.iterdir() if p.is_dir()):
        if not (seq_dir / "imu" / imu_filename).exists():
            skipped.append(seq_dir.name)
            continue
        exported.append(export_sequence(seq_dir, imu_filename, motion_source, out_dir, camera))
    if not exported:
        raise FileNotFoundError(f"no sequence under {corpus_root} carries imu/{imu_filename}")

    manifest = {
        "corpus_root": str(corpus_root),
        "imu_stream_file": imu_filename,
        "motion_source": motion_source,
        "camera": camera,
        "n_sequences": len(exported),
        "skipped_sequences": skipped,
        "sequences": [
            {"sequence_id": e.sequence_id, "n_frames": e.n_frames} for e in exported
        ],
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir, exported
