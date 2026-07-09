"""Parallel-corpus construction.

Layer-A artifacts under ``data/interim/sim2real/corpus/totalcapture/<SEQ>/``:

    motion.npz + motion.manifest.json     canonical MotionSequence (one SMPL-X
                                          forward per sequence, external venv)
    imu/real.npz                          real R_LowArm stream (13 channels)
    imu/synth_<gen>_<cfg8>.npz            one per generator x config (adapters)
    imu/synth_<gen>_<cfg8>.manifest.json
    meta.json                             subject/session/split, sources, frames

Sequence inventory = AMASS SMPL-X availability INTERSECTED with the lxhong
raw-IMU availability. Real streams are read through a per-subject extraction
cache so each ``<s>_Gyro_Mag.tar.gz`` is scanned exactly once.

This module is numpy-only (runs in the repo environment); everything needing
torch/smplx goes through the file-level generator contract via subprocess in
``scripts/sim2real/01_build_corpus.py``.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from src.globalpose_origin_adapter import load_totalcapture_raw_sensor_stream

from .contracts import IMU_CHANNELS_13, REAL_SOURCE, ImuSequence

DEFAULT_AMASS_ROOT = Path("/data/luoyizhang/HuMoGen/data/AMASS/TotalCapture")
DEFAULT_LXHONG_ROOT = Path("/data/lxhong")
REAL_IMU_FPS = 60.0

AUX_SUFFIX = "_Xsens_AuxFields.sensors"


def list_smplx_sequences(amass_root: Path = DEFAULT_AMASS_ROOT) -> dict:
    """Map sequence name ('S1_acting1') -> SMPL-X stageii npz path."""
    sequences = {}
    for subject_dir in sorted(Path(amass_root).glob("s[0-9]*")):
        subject = subject_dir.name.upper()
        for npz in sorted(subject_dir.glob("*_stageii.npz")):
            session = npz.name[: -len("_stageii.npz")]
            sequences[f"{subject}_{session}"] = npz
    return sequences


def _aux_file_to_session(file_name: str) -> str:
    """'Acting1_Xsens_AuxFields.sensors' -> 'acting1'."""
    return file_name[: -len(AUX_SUFFIX)].lower()


def _canonical_aux_name(session: str) -> str:
    """Canonical cache filename matching what the loader probes.

    Source tarballs are inconsistent across subjects (s1 ships
    'Acting1_...', s2/s3 ship 'acting1_...'); the loader expects the
    capitalized form for acting sessions, so normalize on extraction.
    """
    aux = session.capitalize() if session.startswith("acting") else session
    return f"{aux}{AUX_SUFFIX}"


def ensure_aux_cache(lxhong_root: Path, cache_root: Path, subject_lower: str) -> set:
    """Extract all AuxFields members for one subject in a single tar pass.

    Returns the set of session names whose real richer-IMU stream is
    available. The cache layout (``<cache>/<subject>/<Aux>_Xsens_AuxFields.sensors``)
    matches what ``load_totalcapture_raw_sensor_stream`` probes directly.
    """
    subject_dir = Path(cache_root) / subject_lower
    marker = subject_dir / ".complete"
    if marker.exists():
        return {_aux_file_to_session(p.name) for p in subject_dir.glob(f"*{AUX_SUFFIX}")}

    archive = Path(lxhong_root) / f"{subject_lower}_Gyro_Mag.tar.gz"
    if not archive.exists():
        return set()
    subject_dir.mkdir(parents=True, exist_ok=True)
    sessions = set()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = Path(member.name).name
            if not name.endswith(AUX_SUFFIX):
                continue
            session = _aux_file_to_session(name)
            dst = subject_dir / _canonical_aux_name(session)
            if not dst.exists():
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                dst.write_bytes(extracted.read())
            sessions.add(session)
    marker.write_text("")
    return sessions


def extract_real_imu(aux_cache_root: Path, sequence_name: str, sensor: str) -> ImuSequence:
    """Real 13-channel stream for one sequence, from the per-subject cache."""
    parsed = load_totalcapture_raw_sensor_stream(
        imu_source_root=Path(aux_cache_root),
        sequence_name=sequence_name,
        sensor_name=sensor,
    )
    data = np.concatenate(
        [parsed["quat"][:, 0], parsed["acc"][:, 0], parsed["gyro"][:, 0], parsed["mag"][:, 0]],
        axis=1,
    ).astype(np.float32)
    return ImuSequence(
        data=data,
        channels=IMU_CHANNELS_13,
        fps=REAL_IMU_FPS,
        source=REAL_SOURCE,
        sensor=sensor,
        meta={"sequence": sequence_name, "origin": "lxhong Xsens_AuxFields"},
    )
