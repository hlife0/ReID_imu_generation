"""06_run_matrix — execute every cell of a matrix file, resumably (M4, implemented).

Runs under the repo (system) python; each cell is a subprocess of the
CUDA-capable cell python (default: the adaptfm conda env — the data_generation
venv's torch is cu130 and the local driver only supports cu126). Cells are
distributed round-robin over --gpus with one worker per GPU.
"""

from __future__ import annotations

import argparse
import itertools
import json
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real.protocols import completed_cell_ids, make_cell_id

DEFAULT_CELL_PYTHON = "/home/hrli/miniconda3/envs/adaptfm/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", default="configs/sim2real/matrix_v1.yaml")
    parser.add_argument("--windows-root", default="data/interim/sim2real/windows")
    parser.add_argument("--out", default="outputs/sim2real")
    parser.add_argument("--cell-python", default=DEFAULT_CELL_PYTHON)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-all", action="store_true")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def expand_cells(matrix: dict) -> list:
    """Expand matrix cells x real_fraction sweeps x seeds into concrete jobs."""
    seeds = matrix["seeds"]
    jobs = []
    for cell in matrix["cells"]:
        fractions = cell.get("real_fraction", 1.0)
        if not isinstance(fractions, list):
            fractions = [fractions]
        for fraction, seed in itertools.product(fractions, seeds):
            jobs.append({
                "protocol": cell["protocol"],
                "train": list(cell["train"]),
                "pretrain": list(cell.get("pretrain") or []),
                "real_fraction": float(fraction),
                "seed": int(seed),
                "shuffle_pairs": False,
            })
    if matrix.get("controls", {}).get("shuffled_pairs"):
        jobs.append({"protocol": "trtr", "train": ["real"], "pretrain": [],
                     "real_fraction": 1.0, "seed": int(seeds[0]), "shuffle_pairs": True})
    return jobs


def job_cell_id(job: dict) -> str:
    return make_cell_id(
        job["protocol"], job["train"], job["seed"],
        pretrain_tokens=job["pretrain"] or None,
        real_fraction=job["real_fraction"],
        shuffle_pairs=job["shuffle_pairs"],
    )


def main() -> None:
    args = parse_args()
    matrix = yaml.safe_load(_abs(args.matrix).read_text(encoding="utf-8"))
    benchmark_id = matrix["benchmark"]
    results_jsonl = _abs(args.out) / benchmark_id / "results.jsonl"

    jobs = expand_cells(matrix)
    done = set() if args.rerun_all else completed_cell_ids(results_jsonl)
    pending = [j for j in jobs if job_cell_id(j) not in done]
    print(f"[matrix] {len(jobs)} cells total, {len(jobs) - len(pending)} already in ledger, "
          f"{len(pending)} to run")
    if args.dry_run:
        for job in pending:
            print(f"[matrix]   pending: {job_cell_id(job)}")
        return

    gpu_pool: queue.Queue = queue.Queue()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    for gpu in gpus:
        gpu_pool.put(gpu)

    failures = []

    def run_job(job: dict) -> None:
        gpu = gpu_pool.get()
        cell_id = job_cell_id(job)
        try:
            cmd = [args.cell_python, str(REPO_ROOT / "scripts/sim2real/05_run_l2_cell.py"),
                   "--benchmark-id", benchmark_id,
                   "--windows-root", str(_abs(args.windows_root)),
                   "--out", str(_abs(args.out)),
                   "--protocol", job["protocol"],
                   "--train-sources", *job["train"],
                   "--real-fraction", str(job["real_fraction"]),
                   "--seed", str(job["seed"])]
            if job["pretrain"]:
                cmd += ["--pretrain-sources", *job["pretrain"]]
            if job["shuffle_pairs"]:
                cmd.append("--shuffle-pairs")
            completed = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                env={"CUDA_VISIBLE_DEVICES": gpu, "PATH": "/usr/bin:/bin"},
            )
            if completed.returncode != 0:
                failures.append((cell_id, completed.stderr[-1500:]))
                print(f"[matrix] FAIL {cell_id} (gpu {gpu})")
            else:
                row = json.loads(completed.stdout.strip().splitlines()[-1])
                print(f"[matrix] done {cell_id} (gpu {gpu}): "
                      f"R@1={row['r_at_1']} R@5={row['r_at_5']} wall={row['wall_s']}s")
        finally:
            gpu_pool.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(run_job, pending))

    print(f"[matrix] finished: {len(pending) - len(failures)} ok, {len(failures)} failed")
    for cell_id, err in failures:
        print(f"[matrix] FAILED {cell_id}:\n{err}\n")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
