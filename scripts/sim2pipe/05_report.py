"""05_report — aggregate the sim2pipe ledger, compare to sim2real, write report.md (N3).

Reads only results.jsonl; produces:
  * a pipe-probe ranking table (val/test top1, mean±std over seeds);
  * the probe(sim2real) vs pipe(sim2pipe) rank-consistency verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2pipe.ledger import load_rows

_PROTO_ORDER = {"trtr": 0, "tstr": 1, "mix": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outputs-root", default="outputs/sim2pipe")
    p.add_argument("--benchmark-id", default="pipe_probe_tc_v1")
    p.add_argument(
        "--sim2real-results",
        default="outputs/sim2real/tc_rlowarm_w24_estskel_v1/results.jsonl",
    )
    return p.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _ms(xs: list[float]) -> str:
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{st.mean(xs):.4f}±{st.pstdev(xs):.4f}"


def _mean(xs: list[float]) -> float:
    return st.mean(xs) if xs else float("nan")


def aggregate_pipe(rows: list[dict]) -> dict:
    agg: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("shuffle_control"):
            continue
        key = (r["protocol"], r["imu_stream"])
        agg[key]["val"].append(r["val_top1"])
        agg[key]["test"].append(r["test_top1"])
    return agg


def aggregate_probe(path: Path) -> dict:
    """sim2real ledger: source lives in the ``train`` list; metric is r_at_1."""
    agg: dict = defaultdict(list)
    if not path.exists():
        return agg
    for r in load_rows(path):
        if r.get("shuffle_pairs"):
            continue
        src = "+".join(r.get("train", []))
        agg[(r["protocol"], src)].append(r.get("r_at_1", float("nan")))
    return agg


def main() -> None:
    args = parse_args()
    bench_dir = _abs(args.outputs_root) / args.benchmark_id
    rows = load_rows(bench_dir / "results.jsonl")
    if not rows:
        raise SystemExit(f"no results at {bench_dir/'results.jsonl'}")

    pipe = aggregate_pipe(rows)
    probe = aggregate_probe(_abs(args.sim2real_results))
    random_line = rows[0].get("random_line", float("nan"))

    lines: list[str] = []
    lines.append(f"# sim2pipe report — {args.benchmark_id}\n")
    lines.append(
        "Synthetic IMU judged by the **main project's own model** "
        "(IMUVideoMatcher: frozen MotionBERT + frozen DeSPITE, alignment head "
        "trained; SymmetricInfoNCE) via subprocess into its environment. "
        "Metric = in-batch retrieval top1 on held-out **real** windows "
        f"(batch {rows[0].get('batch_size')}, random line ≈ {random_line:.4f}).\n"
    )

    lines.append("## Pipe-probe ranking (mean±std over seeds)\n")
    lines.append("| protocol | stream | val_top1 (S4 real) | test_top1 (S5 real) | n |")
    lines.append("|---|---|---|---|---|")
    for key in sorted(pipe, key=lambda k: (_PROTO_ORDER.get(k[0], 9), k[1])):
        a = pipe[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {_ms(a['val'])} | {_ms(a['test'])} | {len(a['val'])} |"
        )
    lines.append("")

    # probe vs pipe TSTR ranking consistency
    pipe_tstr = {s: _mean(a["test"]) for (p, s), a in pipe.items() if p == "tstr"}
    probe_tstr = {s: _mean(v) for (p, s), v in probe.items() if p == "tstr"}
    common = sorted(set(pipe_tstr) & set(probe_tstr))
    lines.append("## Probe (sim2real) vs pipe (sim2pipe): TSTR consistency\n")
    if common:
        lines.append("| stream | probe R@1 | pipe test_top1 |")
        lines.append("|---|---|---|")
        for s in sorted(common, key=lambda s: -probe_tstr[s]):
            lines.append(f"| {s} | {probe_tstr[s]:.4f} | {pipe_tstr[s]:.4f} |")
        probe_rank = [s for s in sorted(common, key=lambda s: -probe_tstr[s])]
        pipe_rank = [s for s in sorted(common, key=lambda s: -pipe_tstr[s])]
        agree = probe_rank == pipe_rank
        pipe_spread = max(pipe_tstr.values()) - min(pipe_tstr.values())
        lines.append("")
        lines.append(f"- probe order: {' > '.join(probe_rank)}")
        lines.append(f"- pipe order:  {' > '.join(pipe_rank)}")
        lines.append(
            f"- **rank agreement: {agree}**; pipe TSTR spread = {pipe_spread:.4f} "
            f"(vs random line {random_line:.4f})."
        )
        if pipe_spread < random_line:
            lines.append(
                "- pipe TSTR streams are within one random-line of each other — "
                "**the real model does not distinguish the generators; the probe's "
                "ranking did not survive.**"
            )
    else:
        lines.append("_no common TSTR streams between probe and pipe ledgers._")
    lines.append("")

    out = bench_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
