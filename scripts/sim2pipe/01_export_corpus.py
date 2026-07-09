"""01_export_corpus — export sim2real corpus streams to main-project npz format.

One export dir per (imu stream, motion source); each dir is a ready
``preprocess.synthetic_imu_root`` for the main repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2pipe.export import MOTION_SOURCES, export_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", default="data/interim/sim2real/corpus/totalcapture")
    parser.add_argument("--export-root", default="data/interim/sim2pipe/export")
    parser.add_argument(
        "--streams",
        nargs="+",
        required=True,
        help="corpus imu/ filenames, e.g. real.npz synth_naive_8f7d9e76.npz",
    )
    parser.add_argument("--motion-source", choices=MOTION_SOURCES, default="motion_estimated")
    parser.add_argument("--camera", default="cam1")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    corpus_root = _abs(args.corpus_root)
    export_root = _abs(args.export_root)

    for stream in args.streams:
        out_dir, exported = export_corpus(
            corpus_root, stream, args.motion_source, export_root, camera=args.camera
        )
        total_frames = sum(e.n_frames for e in exported)
        print(f"[{stream}] {len(exported)} sequences, {total_frames} frames -> {out_dir}")


if __name__ == "__main__":
    main()
