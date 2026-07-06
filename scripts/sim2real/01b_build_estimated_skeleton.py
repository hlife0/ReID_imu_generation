"""01b_build_estimated_skeleton — corpus layer A, estskel variant.

Reads each sequence's ground-truth ``motion.npz`` (21-joint pipeline skeleton)
and writes ``motion_estimated.npz`` (17-joint H36M estimated skeleton) beside
it, plus a manifest. No SMPL-X, no torch — a pure-numpy transform over the
existing corpus, so it runs under the repo python and takes seconds/sequence.

The estskel benchmark (configs/sim2real/benchmarks/tc_rlowarm_w24_estskel_v1.json)
sets ``motion_source: motion_estimated`` and 03_build_windows then pairs the
same IMU streams against this degraded skeleton.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.contracts import MotionSequence, config_hash, write_manifest
from src.sim2real.gen_estimated import ESTIMATED_LAYOUT, degrade_skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", default="data/interim/sim2real/corpus")
    parser.add_argument("--dataset", default="totalcapture")
    parser.add_argument("--config", default="configs/sim2real/generators/estimated_default.json")
    parser.add_argument("--out-name", default="motion_estimated",
                        help="output stem written into each sequence dir")
    parser.add_argument("--only", default=None, help="restrict to one sequence dir name (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    cfg = json.loads(_abs(args.config).read_text(encoding="utf-8"))
    gen_cfg = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "generator"}
    cfg_hash = config_hash(gen_cfg)

    dataset_root = _abs(args.corpus_root) / args.dataset
    seq_dirs = sorted(p.parent for p in dataset_root.glob("*/motion.npz"))
    if args.only:
        seq_dirs = [d for d in seq_dirs if d.name == args.only]
    if not seq_dirs:
        raise SystemExit(f"no sequences with motion.npz under {dataset_root} (only={args.only!r})")

    written = 0
    for seq_dir in seq_dirs:
        out_path = seq_dir / f"{args.out_name}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"[estskel] skip (exists) {seq_dir.name}")
            continue

        gt = MotionSequence.load(seq_dir / "motion.npz")
        est_joints = degrade_skeleton(gt.joints, gen_cfg)
        est = MotionSequence(
            joints=est_joints,
            fps=gt.fps,
            joint_layout=ESTIMATED_LAYOUT,
            meta={
                "derived_from": "motion.npz",
                "source_layout": gt.joint_layout,
                "generator": "estimated_skeleton",
                "config_hash": cfg_hash,
                "num_joints": int(est_joints.shape[1]),
            },
        )
        est.save(out_path)
        write_manifest(
            seq_dir / f"{args.out_name}.manifest.json",
            stage="corpus_estimated_skeleton",
            config=gen_cfg,
            inputs={"motion": seq_dir / "motion.npz"},
            seed=int(gen_cfg.get("seed", 0)),
            extra={"joint_layout": ESTIMATED_LAYOUT, "num_joints": int(est_joints.shape[1])},
        )
        written += 1
        print(f"[estskel] {seq_dir.name}: {gt.joints.shape} -> {est_joints.shape}")

    print(f"[estskel] wrote {written} estimated-skeleton streams "
          f"(layout={ESTIMATED_LAYOUT}, config_hash={cfg_hash})")


if __name__ == "__main__":
    main()
