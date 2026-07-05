"""02_gate_corpus — run the L0 gate over the parallel corpus (M1, implemented).

Reads gate thresholds from the benchmark spec (refuses to run while they are
null), gates every (sequence, synth stream) pair, writes gate_report.json/md
and records results in each sequence's meta.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.gate import gate_corpus, render_report_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", default="data/interim/sim2real/corpus")
    parser.add_argument("--benchmark", default="configs/sim2real/benchmarks/tc_rlowarm_w24_v1.json")
    parser.add_argument("--out", default="outputs/sim2real/gate")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    spec = json.loads(_abs(args.benchmark).read_text(encoding="utf-8"))
    thresholds = spec["gate"]

    report = gate_corpus(_abs(args.corpus_root), thresholds)

    out_dir = _abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "gate_report.md").write_text(render_report_md(report), encoding="utf-8")

    summary = report["summary"]
    print(f"[gate] sequences={summary['sequences']} streams={summary['streams']} "
          f"failed={summary['failed_streams']}")
    print(f"[gate] report -> {out_dir / 'gate_report.md'}")
    if summary["sequences"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
