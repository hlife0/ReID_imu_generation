"""Data contracts for the sim2real evaluation subsystem.

Everything that crosses a stage boundary (corpus -> windows -> probe ->
report) is written and read through this module, so shape, naming, and
provenance rules are enforced in exactly one place.

Two rules shape the design:

1. File-level contracts. The generator pipelines and the evaluation side run
   in different Python environments (HuMoGen_origin / naive_kinematics execute
   inside an external venv via subprocess), so artifacts are exchanged as
   ``.npz`` arrays plus JSON manifests — never as in-process Python objects.
2. Provenance by construction. Every artifact directory carries a
   ``manifest.json`` (see ``write_manifest``) recording the resolved config,
   its ``config_hash``, input file hashes, seed, and git commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1

REAL_SOURCE = "real"
MANIFEST_FILENAME = "manifest.json"

# Canonical 13-channel IMU layout shared by real and synthetic streams
# (mirrors the existing CSV convention of the maintained pipelines).
IMU_CHANNELS_13 = (
    "quat0", "quat1", "quat2", "quat3",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Configuration identity
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for hashing and npz headers."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable 8-hex-char identity of a configuration mapping."""
    return hashlib.sha1(canonical_json(dict(cfg)).encode("utf-8")).hexdigest()[:8]


def synth_source(generator: str, cfg: Mapping[str, Any] | str) -> str:
    """Canonical source id for a synthetic stream: ``synth/<generator>/<hash>``."""
    if not generator:
        raise ValueError("generator name must be non-empty")
    digest = cfg if isinstance(cfg, str) else config_hash(cfg)
    return f"synth/{generator}/{digest}"


def parse_source(source: str) -> dict:
    """Split a source id into its parts, validating the format.

    Returns ``{"kind": "real"}`` or
    ``{"kind": "synth", "generator": ..., "config_hash": ...}``.
    """
    if source == REAL_SOURCE:
        return {"kind": "real"}
    parts = source.split("/")
    if len(parts) == 3 and parts[0] == "synth" and parts[1] and parts[2]:
        return {"kind": "synth", "generator": parts[1], "config_hash": parts[2]}
    raise ValueError(
        f"invalid source id {source!r}: expected 'real' or 'synth/<generator>/<config_hash>'"
    )


def source_to_token(source: str) -> str:
    """Filesystem-safe token for a source id, used in shard file names.

    ``real`` -> ``real``; ``synth/globalpose/a3f2c1d0`` -> ``synth_globalpose_a3f2c1d0``.
    """
    parsed = parse_source(source)
    if parsed["kind"] == "real":
        return REAL_SOURCE
    return f"synth_{parsed['generator']}_{parsed['config_hash']}"


# ---------------------------------------------------------------------------
# Sequence-level artifacts (corpus layer A)
# ---------------------------------------------------------------------------

def _save_npz(path: Path, header: dict, arrays: dict) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"sequence artifacts must be .npz files, got {path.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, header=np.array(canonical_json(header)), **arrays)
    return path


def _load_npz(path: Path, expected_kind: str) -> tuple[dict, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        header = json.loads(str(data["header"][()]))
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version {header.get('schema_version')!r} != {SCHEMA_VERSION}"
            )
        if header.get("kind") != expected_kind:
            raise ValueError(f"{path}: kind {header.get('kind')!r} != {expected_kind!r}")
        arrays = {name: data[name] for name in data.files if name != "header"}
    return header, arrays


@dataclass
class MotionSequence:
    """Canonical motion stream: joint positions over time.

    ``joints`` has shape (T, J, 3), float32. ``joint_layout`` names the joint
    convention (e.g. ``"pipeline_world_v1"``) so downstream code can refuse to
    mix incompatible layouts.
    """

    joints: np.ndarray
    fps: float
    joint_layout: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float32)
        if self.joints.ndim != 3 or self.joints.shape[2] != 3:
            raise ValueError(f"joints must have shape (T, J, 3), got {self.joints.shape}")
        if self.joints.shape[0] < 1 or self.joints.shape[1] < 1:
            raise ValueError(f"joints must be non-empty, got shape {self.joints.shape}")
        if not np.isfinite(self.joints).all():
            raise ValueError("joints contain NaN or Inf")
        if not self.fps or self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        self.fps = float(self.fps)
        if not self.joint_layout:
            raise ValueError("joint_layout must be a non-empty string")

    @property
    def num_frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    def save(self, path: Path) -> Path:
        header = {
            "schema_version": SCHEMA_VERSION,
            "kind": "motion",
            "fps": self.fps,
            "joint_layout": self.joint_layout,
            "meta": self.meta,
        }
        return _save_npz(Path(path), header, {"joints": self.joints})

    @classmethod
    def load(cls, path: Path) -> "MotionSequence":
        header, arrays = _load_npz(Path(path), "motion")
        return cls(
            joints=arrays["joints"],
            fps=header["fps"],
            joint_layout=header["joint_layout"],
            meta=header.get("meta", {}),
        )


@dataclass
class ImuSequence:
    """A real or synthetic IMU stream for one sensor.

    ``data`` has shape (T, C), float32, with ``channels`` naming each column
    (e.g. ``("acc_x", ..., "gyro_z")``). ``source`` is ``"real"`` or
    ``synth_source(generator, cfg)``.
    """

    data: np.ndarray
    channels: tuple
    fps: float
    source: str
    sensor: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError(f"data must have shape (T, C), got {self.data.shape}")
        if self.data.shape[0] < 1 or self.data.shape[1] < 1:
            raise ValueError(f"data must be non-empty, got shape {self.data.shape}")
        self.channels = tuple(str(c) for c in self.channels)
        if len(self.channels) != self.data.shape[1]:
            raise ValueError(
                f"{len(self.channels)} channel names for {self.data.shape[1]} columns"
            )
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"duplicate channel names: {self.channels}")
        if not self.fps or self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps!r}")
        self.fps = float(self.fps)
        parse_source(self.source)  # validates format
        if not self.sensor:
            raise ValueError("sensor must be a non-empty string (e.g. 'R_LowArm')")

    @property
    def num_frames(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    def save(self, path: Path) -> Path:
        header = {
            "schema_version": SCHEMA_VERSION,
            "kind": "imu",
            "fps": self.fps,
            "channels": list(self.channels),
            "source": self.source,
            "sensor": self.sensor,
            "meta": self.meta,
        }
        return _save_npz(Path(path), header, {"data": self.data})

    @classmethod
    def load(cls, path: Path) -> "ImuSequence":
        header, arrays = _load_npz(Path(path), "imu")
        return cls(
            data=arrays["data"],
            channels=tuple(header["channels"]),
            fps=header["fps"],
            source=header["source"],
            sensor=header["sensor"],
            meta=header.get("meta", {}),
        )


# ---------------------------------------------------------------------------
# Provenance manifests
# ---------------------------------------------------------------------------

def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(cwd: Path | None = None) -> str | None:
    """Best-effort current commit; None when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _manifest_path(target: Path) -> Path:
    target = Path(target)
    return target if target.suffix == ".json" else target / MANIFEST_FILENAME


def write_manifest(
    target: Path,
    *,
    stage: str,
    config: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Path] | None = None,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict:
    """Write ``manifest.json`` describing how an artifact was produced.

    ``target`` may be an artifact directory (manifest is placed inside it) or
    a full ``*.json`` path. ``inputs`` maps logical names to file paths; each
    existing input is recorded with size and sha1 so any run can be traced
    back to exact bytes.
    """
    recorded_inputs = {}
    for name, path in (inputs or {}).items():
        path = Path(path)
        if path.is_file():
            recorded_inputs[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha1": file_sha1(path),
            }
        else:
            recorded_inputs[name] = {"path": str(path), "missing": True}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "seed": seed,
        "config": dict(config) if config is not None else None,
        "config_hash": config_hash(config) if config is not None else None,
        "inputs": recorded_inputs,
        "extra": dict(extra) if extra is not None else {},
    }

    path = _manifest_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_manifest(target: Path) -> dict:
    return json.loads(_manifest_path(target).read_text(encoding="utf-8"))
