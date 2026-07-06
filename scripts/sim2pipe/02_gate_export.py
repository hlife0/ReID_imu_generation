"""02_gate_export — P0 gate for sim2pipe (design §3).

N0 scope (implemented): --check-deps reports every missing prerequisite
(main repo, its python env, external ckpts, corpus) and exits non-zero if
any is missing.

N1 scope (not yet implemented): format round-trip through the main repo's
preprocess/slice, the real-stream anchor training, and the shuffled-pairs
control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2pipe.bridge import PipePaths, check_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/sim2pipe/paths.yaml")
    parser.add_argument("--corpus-root", default="data/interim/sim2real/corpus/totalcapture")
    parser.add_argument("--check-deps", action="store_true", help="dependency check only (N0)")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    paths_file = _abs(args.paths)
    if not paths_file.exists():
        example = paths_file.with_suffix(".yaml.example")
        print(f"MISSING paths config: {paths_file}")
        print(f"  -> copy {example.name} to {paths_file.name} and adjust for this machine")
        sys.exit(1)

    paths = PipePaths.load(paths_file)
    problems = check_dependencies(paths, corpus_root=_abs(args.corpus_root))
    if problems:
        print(f"GATE FAIL — {len(problems)} missing prerequisite(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("Dependency check PASS: main repo, python env, ckpts, corpus all reachable.")

    if not args.check_deps:
        print("NOTE: round-trip / real-stream anchor / shuffle control are N1 scope, "
              "not implemented yet; ran dependency check only.")


if __name__ == "__main__":
    main()
