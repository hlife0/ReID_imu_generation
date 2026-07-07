"""Window materialization (corpus layer A -> benchmark layer B) — M2 (implemented).

Materializes ``data/interim/sim2real/windows/<benchmark_id>/``:

    spec.json            resolved benchmark spec + shard inventory + per-source
                         train stats + leakage-check result
    <split>__<token>.npz shards: imu (N,W,C) raw windows, motion (N,W,J,3)
                         per-window-centered, sequence/subject/start metadata

Rules enforced here:

- val/test shards contain ONLY the real source; synthetic windows exist for
  train-split sequences only.
- Gate-failed synthetic streams are excluded per (sequence, stream).
- IMU windows are stored RAW; per-source train-split channel stats are
  recorded in spec.json and normalization is applied at protocol load time
  (train-source stats applied unchanged to val/test — the honest TSTR rule).
- Motion windows are centered per window (single offset over time x joints):
  removes absolute location (identity shortcut) while keeping within-window
  translation dynamics.
- After writing, ``splits.find_leakage`` runs as a hard assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .alignment import aligned_slices, lag_from_meta
from .contracts import (
    IMU_CHANNELS_13,
    REAL_SOURCE,
    SCHEMA_VERSION,
    ImuSequence,
    MotionSequence,
    canonical_json,
    file_sha1,
)
from .splits import find_leakage, load_split


def window_starts(num_frames: int, window_len: int, stride: int) -> list:
    if num_frames < window_len:
        return []
    return list(range(0, num_frames - window_len + 1, stride))


def extract_windows(data: np.ndarray, starts: list, window_len: int) -> np.ndarray:
    """Stack windows from a (T, ...) array -> (N, window_len, ...)."""
    if not starts:
        return np.empty((0, window_len) + data.shape[1:], dtype=data.dtype)
    return np.stack([data[s : s + window_len] for s in starts], axis=0)


def center_motion_windows(motion_windows: np.ndarray) -> np.ndarray:
    """Subtract each window's mean over (time, joints) — one offset per window."""
    if motion_windows.shape[0] == 0:
        return motion_windows
    offset = motion_windows.mean(axis=(1, 2), keepdims=True)
    return (motion_windows - offset).astype(np.float32)


def channel_stats(imu_windows: np.ndarray) -> dict:
    """Per-channel mean/std over all windows and frames of one train source."""
    flat = imu_windows.reshape(-1, imu_windows.shape[-1])
    std = flat.std(axis=0)
    std[std == 0.0] = 1.0
    return {"mean": flat.mean(axis=0).tolist(), "std": std.tolist()}


def _save_shard(path: Path, header: dict, imu, motion, sequences, subjects, starts) -> None:
    np.savez_compressed(
        path,
        header=np.array(canonical_json(header)),
        imu=np.asarray(imu, dtype=np.float32),
        motion=np.asarray(motion, dtype=np.float32),
        sequence=np.asarray(sequences),
        subject=np.asarray(subjects),
        start=np.asarray(starts, dtype=np.int32),
    )


def load_shard(path: Path) -> tuple:
    """Return (header dict, arrays dict) for one shard."""
    with np.load(Path(path), allow_pickle=False) as data:
        header = json.loads(str(data["header"][()]))
        arrays = {name: data[name] for name in data.files if name != "header"}
    return header, arrays


def materialize_benchmark(benchmark_spec_path: Path, corpus_root: Path, windows_root: Path) -> Path:
    spec_path = Path(benchmark_spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    window_len, stride = int(spec["window_len"]), int(spec["stride"])
    target_fps = float(spec["target_fps"])
    channels = list(spec["channels"])
    channel_idx = [IMU_CHANNELS_13.index(c) for c in channels]
    # Which per-sequence motion file to pair IMU against. Defaults to the
    # ground-truth "motion" stream so pre-estskel benchmarks are unchanged;
    # the estskel benchmark sets this to "motion_estimated" (H36M-17).
    motion_source = str(spec.get("motion_source", "motion"))

    split_path = spec_path.parent.parent.parent.parent / spec["split_file"]
    split = load_split(split_path)

    dataset_root = Path(corpus_root) / spec["dataset"]
    out_dir = Path(windows_root) / spec["benchmark_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = {}   # (split, token) -> dict of lists
    excluded = []

    for meta_path in sorted(dataset_root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seq, subject, seq_split = meta["sequence"], meta["subject"], meta["split"]
        seq_dir = meta_path.parent
        motion = MotionSequence.load(seq_dir / f"{motion_source}.npz")
        gate = (meta.get("gate") or {}).get("streams", {})

        stream_paths = [seq_dir / "imu" / "real.npz"]
        if seq_split == "train":
            stream_paths += sorted((seq_dir / "imu").glob("synth_*.npz"))

        for stream_path in stream_paths:
            imu = ImuSequence.load(stream_path)
            token = "real" if imu.source == REAL_SOURCE else stream_path.stem
            if imu.source != REAL_SOURCE:
                verdict = gate.get(stream_path.name)
                if verdict is None or not verdict["passed"]:
                    excluded.append({"sequence": seq, "stream": stream_path.name,
                                     "reason": "gate failed" if verdict else "not gated"})
                    continue
            if abs(imu.fps - target_fps) > 1e-6 or abs(motion.fps - target_fps) > 1e-6:
                excluded.append({"sequence": seq, "stream": stream_path.name,
                                 "reason": f"fps != target ({imu.fps}/{motion.fps} vs {target_fps})"})
                continue

            # Per-sequence temporal alignment (real[i] <-> motion[i+lag]);
            # synthetic streams are generated on the motion timebase (lag 0).
            if imu.source == REAL_SOURCE:
                lag = lag_from_meta(meta)
                if lag is None:
                    raise RuntimeError(
                        f"{seq}: meta.json has no per-sequence alignment; run "
                        "scripts/sim2real/01c_estimate_alignment.py before building windows"
                    )
            else:
                lag = 0
            sl_imu, sl_motion = aligned_slices(imu.num_frames, motion.num_frames, lag)
            n = sl_imu.stop - sl_imu.start
            starts = window_starts(n, window_len, stride)
            imu_w = extract_windows(imu.data[sl_imu][:, channel_idx], starts, window_len)
            motion_w = center_motion_windows(
                extract_windows(motion.joints[sl_motion], starts, window_len)
            )
            bucket = buckets.setdefault((seq_split, token), {
                "imu": [], "motion": [], "sequence": [], "subject": [], "start": [],
            })
            bucket["imu"].append(imu_w)
            bucket["motion"].append(motion_w)
            bucket["sequence"] += [seq] * len(starts)
            bucket["subject"] += [subject] * len(starts)
            bucket["start"] += starts

    shards, stats, shard_subjects = {}, {}, {}
    for (seq_split, token), bucket in sorted(buckets.items()):
        shard_name = f"{seq_split}__{token}"
        imu_w = np.concatenate(bucket["imu"], axis=0)
        motion_w = np.concatenate(bucket["motion"], axis=0)
        header = {
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": spec["benchmark_id"],
            "split": seq_split,
            "source_token": token,
            "window_len": window_len,
            "stride": stride,
            "channels": channels,
            "num_windows": int(imu_w.shape[0]),
            "num_joints": int(motion_w.shape[2]),
        }
        _save_shard(out_dir / f"{shard_name}.npz", header,
                    imu_w, motion_w, bucket["sequence"], bucket["subject"], bucket["start"])
        shards[shard_name] = {"windows": int(imu_w.shape[0]),
                              "sequences": len(set(bucket["sequence"]))}
        shard_subjects[shard_name] = set(bucket["subject"])
        if seq_split == "train":
            stats[token] = channel_stats(imu_w)

    violations = find_leakage(split, shard_subjects)
    if violations:
        raise RuntimeError("split leakage detected:\n" + "\n".join(violations))

    resolved = {
        **spec,
        "resolved": {
            "split_file_sha1": file_sha1(split_path),
            "shards": shards,
            "train_channel_stats": stats,
            "excluded_streams": excluded,
            "leakage_check": "clean",
        },
    }
    (out_dir / "spec.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_dir
