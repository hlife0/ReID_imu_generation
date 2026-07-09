"""04_run_l1 — distribution-level metrics between real and each synth source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.dist_metrics import c2st_auc, frechet_distance, mmd_rbf, standardize_by_reference
from src.sim2real.embed import embed_windows
from src.sim2real.windows import load_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-id", default="tc_rlowarm_w24_v1")
    parser.add_argument("--windows-root", default="data/interim/sim2real/windows")
    parser.add_argument("--encoder", default="stats_v1")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/sim2real")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    windows_dir = _abs(args.windows_root) / args.benchmark_id
    real_header, real_arrays = load_shard(windows_dir / f"{args.split}__real.npz")
    real_emb = embed_windows(real_arrays["imu"], args.encoder)

    rows = []
    for shard_path in sorted(windows_dir.glob(f"{args.split}__synth_*.npz")):
        header, arrays = load_shard(shard_path)
        synth_emb = embed_windows(arrays["imu"], args.encoder)
        ref, other = standardize_by_reference(real_emb, synth_emb)
        rows.append({
            "source": header["source_token"],
            "windows": header["num_windows"],
            "frechet": round(frechet_distance(ref, other), 4),
            "mmd2": round(mmd_rbf(ref, other, seed=args.seed), 6),
            "c2st_auc": round(c2st_auc(ref, other, seed=args.seed), 4),
        })

    rows.sort(key=lambda r: r["c2st_auc"])
    report = {
        "benchmark_id": args.benchmark_id,
        "split": args.split,
        "encoder": args.encoder,
        "seed": args.seed,
        "real_windows": real_header["num_windows"],
        "rows": rows,
        "notes": "lower frechet/mmd2 = closer to real; c2st_auc 0.5 = indistinguishable, 1.0 = trivially separable",
    }

    out_dir = _abs(args.out) / args.benchmark_id / "l1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "l1_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# L1 distribution metrics — {args.benchmark_id} ({args.split} split, encoder={args.encoder})",
        "",
        "| source | windows | Fréchet ↓ | MMD² ↓ | C2ST AUC →0.5 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['source']} | {r['windows']} | {r['frechet']} | {r['mmd2']} | {r['c2st_auc']} |")
    (out_dir / "l1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for r in rows:
        print(f"[l1] {r['source']:38s} frechet={r['frechet']:10.3f} mmd2={r['mmd2']:.5f} c2st_auc={r['c2st_auc']:.4f}")
    print(f"[l1] report -> {out_dir / 'l1_report.md'}")


if __name__ == "__main__":
    main()
