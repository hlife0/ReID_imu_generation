"""03_build_windows — materialize benchmark window shards from the corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.windows import materialize_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", default="configs/sim2real/benchmarks/tc_rlowarm_w24_v1.json")
    parser.add_argument("--corpus-root", default="data/interim/sim2real/corpus")
    parser.add_argument("--windows-root", default="data/interim/sim2real/windows")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    out_dir = materialize_benchmark(_abs(args.benchmark), _abs(args.corpus_root), _abs(args.windows_root))
    spec = json.loads((out_dir / "spec.json").read_text(encoding="utf-8"))
    print(f"[windows] -> {out_dir}")
    for shard, info in sorted(spec["resolved"]["shards"].items()):
        print(f"[windows] {shard}: {info['windows']} windows from {info['sequences']} sequences")
    excluded = spec["resolved"]["excluded_streams"]
    if excluded:
        print(f"[windows] excluded {len(excluded)} streams (gate/fps): see spec.json")
    print(f"[windows] leakage check: {spec['resolved']['leakage_check']}")


if __name__ == "__main__":
    main()
