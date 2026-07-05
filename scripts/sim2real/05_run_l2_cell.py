"""05_run_l2_cell — run ONE matrix cell (M3/M4, implemented).

MUST be executed with the external venv python (torch + GPU), e.g.:
    /home/hrli/data_generation/.venv/bin/python scripts/sim2real/05_run_l2_cell.py \\
        --protocol trtr --train-sources real --seed 0
Normally invoked by 06_run_matrix; direct invocation is for debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from sim2real.protocols import run_cell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-id", default="tc_rlowarm_w24_v1")
    parser.add_argument("--windows-root", default="data/interim/sim2real/windows")
    parser.add_argument("--protocol", required=True,
                        choices=["trtr", "tstr", "mix", "pretrain_finetune"])
    parser.add_argument("--train-sources", nargs="+", required=True)
    parser.add_argument("--pretrain-sources", nargs="*", default=[])
    parser.add_argument("--real-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-pairs", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="outputs/sim2real")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    out_root = _abs(args.out) / args.benchmark_id
    row = run_cell(
        windows_dir=_abs(args.windows_root) / args.benchmark_id,
        results_jsonl=out_root / "results.jsonl",
        cells_dir=out_root / "cells",
        protocol=args.protocol,
        train_tokens=args.train_sources,
        seed=args.seed,
        pretrain_tokens=args.pretrain_sources or None,
        real_fraction=args.real_fraction,
        shuffle_pairs=args.shuffle_pairs,
        device=args.device,
    )
    printable = {k: v for k, v in row.items() if k not in ("per_sequence_r1", "norm_stats")}
    print(json.dumps(printable, ensure_ascii=False))


if __name__ == "__main__":
    main()
