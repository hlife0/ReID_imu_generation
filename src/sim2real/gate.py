"""L0 gate: signal-level sanity checks on the parallel corpus — M1 (implemented).

The gate's single job is to catch broken pairing/alignment (wrong sequence,
wrong sensor, wildly different lengths, fps mismatch) BEFORE any downstream
run — so a large sim-to-real gap is never misdiagnosed alignment breakage.

Gate metrics are MAGNITUDE correlations (rotation-invariant), because axis-
level correlations legitimately differ across frame conventions (measured on
the known-good S1_freestyle3: acc-axis pearson is even negative while the
pair is clearly aligned).

Fail rule: a stream fails only when acc AND gyro magnitude correlations are
BOTH below the floor (or fps/length mismatch). Broken pairing kills both
channels at once; a single weak channel is a GENERATOR-quality property
(e.g. HuMoGen-origin's Z-up gravity in a Y-up world corrupts acc but not
gyro) — exactly what the L2 benchmark must be allowed to punish, so those
streams stay in. Weak channels are recorded as ``weak_channels``.

Thresholds live in the benchmark spec ``gate`` block (calibrated on
S1_freestyle3, see ``_gate_calibration`` there). Streams that fail are
excluded from window materialization; reasons land in the sequence meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .alignment import aligned_slices, lag_from_meta
from .contracts import ImuSequence

ACC_SLICE = slice(4, 7)
GYRO_SLICE = slice(7, 10)

REQUIRED_THRESHOLDS = ("min_acc_mag_pearson", "min_gyro_mag_pearson", "max_tail_trim_ratio")


def _magnitude(data: np.ndarray, channels: slice) -> np.ndarray:
    return np.linalg.norm(data[:, channels], axis=1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def check_thresholds(thresholds: dict) -> None:
    missing = [k for k in REQUIRED_THRESHOLDS if thresholds.get(k) is None]
    if missing:
        raise ValueError(
            f"gate thresholds not calibrated (null/missing: {missing}); "
            "calibrate on a known-good sequence first — see docs/sim2real_design.md §8"
        )


def gate_stream(real: ImuSequence, synth: ImuSequence, thresholds: dict,
                lag: int | None = None) -> dict:
    """PASS/FAIL one synthetic stream against the real stream.

    ``lag`` is the per-sequence ``imu_motion_lag`` (real[i] <-> motion[i+lag];
    synthetic streams live on the motion timebase, so the same lag applies).
    ``None`` falls back to the legacy tail alignment (pre-01c corpora).
    """
    reasons = []
    if abs(real.fps - synth.fps) > 1e-6:
        reasons.append(f"fps mismatch: real={real.fps} synth={synth.fps}")

    if lag is None:
        frames = min(real.num_frames, synth.num_frames)
        sl_r = sl_s = slice(-frames, None)
    else:
        sl_r, sl_s = aligned_slices(real.num_frames, synth.num_frames, lag)
        frames = sl_r.stop - sl_r.start
    trim_ratio = 1.0 - frames / max(real.num_frames, synth.num_frames)
    if trim_ratio > thresholds["max_tail_trim_ratio"]:
        reasons.append(f"tail trim ratio {trim_ratio:.3f} > {thresholds['max_tail_trim_ratio']}")

    r, g = real.data[sl_r], synth.data[sl_s]
    metrics = {
        "frames_aligned": int(frames),
        "tail_trim_ratio": round(float(trim_ratio), 4),
        "lag_applied": lag,
        "acc_mag_pearson": round(_pearson(_magnitude(r, ACC_SLICE), _magnitude(g, ACC_SLICE)), 4),
        "gyro_mag_pearson": round(_pearson(_magnitude(r, GYRO_SLICE), _magnitude(g, GYRO_SLICE)), 4),
    }
    if metrics["acc_mag_pearson"] < thresholds["min_acc_mag_pearson"] and \
            metrics["gyro_mag_pearson"] < thresholds["min_gyro_mag_pearson"]:
        reasons.append(
            f"pairing broken: acc_mag_pearson {metrics['acc_mag_pearson']} AND "
            f"gyro_mag_pearson {metrics['gyro_mag_pearson']} both below floor"
        )

    weak = []
    if metrics["acc_mag_pearson"] < thresholds["min_acc_mag_pearson"]:
        weak.append("acc")
    if metrics["gyro_mag_pearson"] < thresholds["min_gyro_mag_pearson"]:
        weak.append("gyro")

    return {"passed": not reasons, "reasons": reasons, "weak_channels": weak, "metrics": metrics}


def gate_sequence(sequence_dir: Path, thresholds: dict) -> dict:
    """Gate every synthetic stream of one corpus sequence; update its meta.json.

    Uses the per-sequence ``imu_motion_lag`` from meta.json when present
    (written by 01c_estimate_alignment); legacy tail alignment otherwise.
    """
    sequence_dir = Path(sequence_dir)
    imu_dir = sequence_dir / "imu"
    real = ImuSequence.load(imu_dir / "real.npz")

    meta_path = sequence_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    lag = lag_from_meta(meta) if meta else None

    results = {}
    for synth_path in sorted(imu_dir.glob("synth_*.npz")):
        synth = ImuSequence.load(synth_path)
        results[synth_path.name] = gate_stream(real, synth, thresholds, lag=lag)

    if meta is not None:
        meta["gate"] = {"thresholds": thresholds, "streams": results}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


def gate_corpus(corpus_root: Path, thresholds: dict) -> dict:
    """Gate every sequence under <corpus_root>/<dataset>/<sequence>/."""
    check_thresholds(thresholds)
    corpus_root = Path(corpus_root)
    sequences = {}
    for meta_path in sorted(corpus_root.glob("*/*/meta.json")):
        sequence_dir = meta_path.parent
        sequences[sequence_dir.name] = gate_sequence(sequence_dir, thresholds)

    total = sum(len(streams) for streams in sequences.values())
    failed = sum(1 for streams in sequences.values() for r in streams.values() if not r["passed"])
    return {
        "thresholds": thresholds,
        "sequences": sequences,
        "summary": {"sequences": len(sequences), "streams": total, "failed_streams": failed},
    }


def render_report_md(report: dict) -> str:
    lines = [
        "# sim2real L0 Gate Report",
        "",
        f"- sequences: {report['summary']['sequences']}",
        f"- streams: {report['summary']['streams']} (failed: {report['summary']['failed_streams']})",
        f"- thresholds: `{json.dumps(report['thresholds'])}`",
        "",
        "| sequence | stream | acc_mag_p | gyro_mag_p | trim | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for seq, streams in sorted(report["sequences"].items()):
        for name, result in sorted(streams.items()):
            m = result["metrics"]
            verdict = "PASS" if result["passed"] else "FAIL: " + "; ".join(result["reasons"])
            lines.append(
                f"| {seq} | {name} | {m['acc_mag_pearson']} | {m['gyro_mag_pearson']} "
                f"| {m['tail_trim_ratio']} | {verdict} |"
            )
    return "\n".join(lines) + "\n"
