"""07_report — aggregate the results ledger into the final report (M4, implemented)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.report import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark-id", default="tc_rlowarm_w24_v1")
    parser.add_argument("--results", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = REPO_ROOT / "outputs" / "sim2real" / args.benchmark_id
    results = Path(args.results) if args.results else base / "results.jsonl"
    out = Path(args.out) if args.out else base / "report.md"
    path = write_report(results, out, args.benchmark_id)
    print(path.read_text(encoding="utf-8"))
    print(f"[report] -> {path}")


if __name__ == "__main__":
    main()
