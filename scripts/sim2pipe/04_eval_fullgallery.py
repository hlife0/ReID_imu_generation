"""04_eval_fullgallery — re-judge sim2pipe checkpoints with the sim2real protocol.

Judgment-logic alignment (2026-07-07 decision): the main project's native eval
(in-batch top1, batch=64 consecutive overlapping windows, bidirectional) is a
protocol-dependent yardstick — chance depends on batch size, negatives are
near-duplicates from one video. This driver replaces the JUDGMENT with the
sim2real one, while keeping the main project's model as the encoder:

  * gallery  = ALL test windows (same 1005-window grid as sim2real's
               test__real shard — verified identical);
  * metric   = one-directional IMU -> motion R@1 / R@5, chance = 1/N;
  * breakdown = per-sequence, exactly like src/sim2real/probe/retrieve.py.

No retraining: for each already-trained cell work dir it extracts embeddings
via src/sim2pipe/main_env_scripts/embed_windows.py (subprocess in the main
project's env) and computes the metrics here in numpy. Appends one row per
cell to <outputs-root>/results.jsonl and writes report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2pipe.bridge import PipePaths, run_main_python

EMBED_SCRIPT = REPO_ROOT / "src" / "sim2pipe" / "main_env_scripts" / "embed_windows.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--work-roots", nargs="+",
                   default=["data/interim/sim2pipe/probe_lagfix",
                            "data/interim/sim2pipe/probe_true_floor_lagfix"])
    p.add_argument("--paths", default="configs/sim2pipe/paths.yaml")
    p.add_argument("--outputs-root", default="outputs/sim2pipe_fullgal")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--only", nargs="*", help="restrict to work-dir tags")
    return p.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def fullgallery_metrics(imu_emb: np.ndarray, video_emb: np.ndarray,
                        sequences: list) -> dict:
    """sim2real judgment: IMU->motion full-gallery retrieval (cosine)."""
    a = imu_emb / np.linalg.norm(imu_emb, axis=1, keepdims=True)
    b = video_emb / np.linalg.norm(video_emb, axis=1, keepdims=True)
    sim = a @ b.T
    order = np.argsort(-sim, axis=1)
    target = np.arange(len(a))
    hit1 = order[:, 0] == target
    hit5 = (order[:, :5] == target[:, None]).any(axis=1)

    seq_arr = np.asarray(sequences)
    per_sequence = {}
    for seq in sorted(set(sequences)):
        mask = seq_arr == seq
        per_sequence[seq] = {"r1": round(float(hit1[mask].mean()), 4),
                             "windows": int(mask.sum())}
    return {
        "r_at_1": round(float(hit1.mean()), 4),
        "r_at_5": round(float(hit5.mean()), 4),
        "gallery_size": int(len(a)),
        "chance_r_at_1": round(1.0 / len(a), 6),
        "per_sequence": per_sequence,
    }


def read_sequences(test_csv: Path) -> list:
    with open(test_csv, newline="") as handle:
        return [f"{row['subject']}_{row['session']}" for row in csv.DictReader(handle)]


def parse_tag(tag: str) -> dict:
    # <protocol>__<stream>__<motion_source>__s<seed>[__destroy][__suffix]
    parts = tag.split("__")
    info = {"protocol": parts[0], "imu_stream": parts[1], "motion_source": parts[2],
            "seed": int(parts[3].lstrip("s")), "destroy_pairing": "destroy" in parts[4:]}
    return info


def main() -> None:
    args = parse_args()
    paths = PipePaths.load(_abs(args.paths))
    out_root = _abs(args.outputs_root)
    ledger = out_root / "results.jsonl"
    done = set()
    if ledger.exists():
        done = {json.loads(l)["tag"] for l in ledger.read_text().splitlines() if l.strip()}

    work_dirs = []
    for root in args.work_roots:
        work_dirs += sorted(p for p in _abs(root).iterdir() if (p / "slice").is_dir())
    if args.only:
        work_dirs = [w for w in work_dirs if w.name in args.only]

    rows = []
    for work in work_dirs:
        tag = work.name
        if tag in done:
            print(f"skip (done): {tag}")
            continue
        train_dir = work / "train" / tag
        best_pt = train_dir / "best.pt"
        stats_json = train_dir / "imu_stats.json"
        test_csv = work / "slice" / "windows_test.csv"
        if not (best_pt.exists() and stats_json.exists() and test_csv.exists()):
            print(f"MISSING artifacts for {tag}, skipping")
            continue

        emb_npz = work / "test_embeddings.npz"
        argv = [str(EMBED_SCRIPT),
                "--test_csv", str(test_csv), "--data_root", str(work / "slice"),
                "--checkpoint", str(best_pt), "--imu_stats_json", str(stats_json),
                "--motionbert_ckpt", str(paths.motionbert_ckpt),
                "--imu_sensor", "R_LowArm", "--repeat_single_sensor", "4",
                "--device", args.device, "--out_npz", str(emb_npz)]
        mb_root = paths.extra.get("motionbert_root")
        if mb_root:
            argv += ["--motionbert_root", str(mb_root)]
        r = run_main_python(paths, argv, log_path=work / "embed.log")
        if r.returncode != 0:
            raise RuntimeError(f"embedding extraction failed for {tag}; see {work/'embed.log'}")

        with np.load(emb_npz, allow_pickle=True) as data:
            metrics = fullgallery_metrics(data["imu"], data["video"], read_sequences(test_csv))

        row = {"tag": tag, **parse_tag(tag), **metrics, "work": str(work)}
        rows.append(row)
        out_root.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{tag}: R@1={metrics['r_at_1']} R@5={metrics['r_at_5']} "
              f"(chance {metrics['chance_r_at_1']})")

    # report over the full ledger
    all_rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    groups = {}
    for r in all_rows:
        key = (r["protocol"], r["imu_stream"], r["destroy_pairing"])
        groups.setdefault(key, []).append(r["r_at_1"])
    lines = [
        "# sim2pipe full-gallery judgment (sim2real-aligned protocol)",
        "",
        "Main-project encoder, sim2real judgment: IMU->motion retrieval over the",
        f"FULL test gallery ({all_rows[0]['gallery_size']} windows, chance "
        f"{all_rows[0]['chance_r_at_1']}). Same window grid and split as "
        "tc_rlowarm_w24_estskel_v1.",
        "",
        "| protocol | stream | destroy | R@1 mean±std (n) |",
        "|---|---|---|---|",
    ]
    for key in sorted(groups):
        v = np.array(groups[key], dtype=float)
        std = v.std(ddof=1) if len(v) > 1 else 0.0
        lines.append(f"| {key[0]} | {key[1]} | {'yes' if key[2] else ''} "
                     f"| {v.mean():.4f}±{std:.4f} ({len(v)}) |")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {out_root / 'report.md'}")


if __name__ == "__main__":
    main()
