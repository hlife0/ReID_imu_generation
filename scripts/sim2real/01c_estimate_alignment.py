"""01c_estimate_alignment — per-sequence real<->motion lag estimation.

For every corpus sequence, estimate ``imu_motion_lag`` (real[i] <-> motion[i+k])
by sliding-Pearson scan of acc magnitudes, using the deterministic naive synth
stream as the motion-timebase bridge (it is a pure function of motion, so its
lag against motion is 0 by construction). Upgrades each ``meta.json``'s
``alignment`` field from the legacy string ``"tail"`` to a dict (see
``src/sim2real/alignment.alignment_record``) and writes a corpus-wide lag
table under outputs/.

Numpy-only; runs under the repo python.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.alignment import (
    DEFAULT_MAX_LAG,
    MIN_BRIDGE_CORR,
    alignment_record,
    estimate_lag,
)
from src.sim2real.contracts import ImuSequence

ACC = slice(4, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", default="data/interim/sim2real/corpus")
    parser.add_argument("--dataset", default="totalcapture")
    parser.add_argument("--bridge-glob", default="synth_naive_*.npz",
                        help="motion-timebase bridge stream (deterministic generator)")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--out", default="outputs/sim2real/alignment")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    dataset_root = _abs(args.corpus_root) / args.dataset
    seq_dirs = sorted(p.parent for p in dataset_root.glob("*/meta.json"))
    if not seq_dirs:
        raise SystemExit(f"no sequences under {dataset_root}")

    rows, warnings = [], []
    for seq_dir in seq_dirs:
        bridges = sorted((seq_dir / "imu").glob(args.bridge_glob))
        if len(bridges) != 1:
            raise SystemExit(
                f"{seq_dir.name}: expected exactly one bridge stream matching "
                f"{args.bridge_glob!r}, found {[b.name for b in bridges]}"
            )
        real = ImuSequence.load(seq_dir / "imu" / "real.npz")
        bridge = ImuSequence.load(bridges[0])

        real_mag = np.linalg.norm(real.data[:, ACC], axis=1)
        bridge_mag = np.linalg.norm(bridge.data[:, ACC], axis=1)
        lag, corr = estimate_lag(real_mag, bridge_mag, max_lag=args.max_lag)

        record = alignment_record(lag, corr, bridges[0].name, max_lag=args.max_lag)
        meta_path = seq_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        legacy = meta.get("alignment")
        meta["alignment"] = record
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

        row = {
            "sequence": seq_dir.name,
            "real_frames": real.num_frames,
            "motion_frames": bridge.num_frames,
            "imu_motion_lag": lag,
            "acc_mag_pearson": round(corr, 4),
            "legacy_alignment": legacy if isinstance(legacy, str) else "dict",
        }
        rows.append(row)
        if corr < MIN_BRIDGE_CORR:
            warnings.append(f"{seq_dir.name}: bridge corr {corr:.3f} < {MIN_BRIDGE_CORR}")
        print(f"[align] {seq_dir.name:16s} lag={lag:+3d} corr={corr:.4f}")

    out_dir = _abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lag_table.json").write_text(
        json.dumps({"rows": rows, "warnings": warnings}, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# sim2real per-sequence IMU<->motion alignment (naive_bridge_lagscan_v1)",
        "",
        "`imu_motion_lag = k` means `real[i] <-> motion[i+k]`. Estimated by acc-magnitude",
        "sliding Pearson against the deterministic naive synth stream (motion timebase).",
        "",
        "| sequence | real | motion | lag | corr@lag |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['sequence']} | {r['real_frames']} | {r['motion_frames']} "
                     f"| {r['imu_motion_lag']:+d} | {r['acc_mag_pearson']} |")
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    (out_dir / "lag_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lags = [r["imu_motion_lag"] for r in rows]
    print(f"[align] {len(rows)} sequences; lag range [{min(lags)}, {max(lags)}]; "
          f"{len(warnings)} warnings -> {out_dir}")


if __name__ == "__main__":
    main()
