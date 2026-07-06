"""03_run_pipe_probe — drive the P1 matrix through the main project's pipeline (N2).

CLI contract frozen at N0; implementation lands in N2 after the P0 gate
(02_gate_export) passes. Reads configs/sim2pipe/matrix_pipe_v1.yaml, skips
cells already present in outputs/sim2pipe/<benchmark_id>/results.jsonl.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", default="configs/sim2pipe/matrix_pipe_v1.yaml")
    parser.add_argument("--paths", default="configs/sim2pipe/paths.yaml")
    parser.add_argument("--export-root", default="data/interim/sim2pipe/export")
    parser.add_argument("--outputs-root", default="outputs/sim2pipe")
    parser.add_argument("--only", nargs="*", help="restrict to cells whose stream token matches")
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise NotImplementedError(
        "N2 milestone: pipe-probe matrix driver is not implemented yet "
        "(prereq: P0 gate via scripts/sim2pipe/02_gate_export.py)."
    )


if __name__ == "__main__":
    main()
