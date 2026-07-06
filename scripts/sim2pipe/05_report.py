"""05_report — aggregate the sim2pipe ledger and compare against sim2real (N3).

CLI contract frozen at N0; implementation lands in N3. Reads only the
results.jsonl ledger; produces report.md with the pipe-probe ranking table
and the probe(sim2real) vs pipe(sim2pipe) rank-consistency verdict.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs-root", default="outputs/sim2pipe")
    parser.add_argument("--benchmark-id", default="pipe_probe_tc_v1")
    parser.add_argument(
        "--sim2real-results",
        default="outputs/sim2real/tc_rlowarm_w24_estskel_v1/results.jsonl",
        help="probe ledger to compare rankings against",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise NotImplementedError("N3 milestone: report aggregation is not implemented yet.")


if __name__ == "__main__":
    main()
