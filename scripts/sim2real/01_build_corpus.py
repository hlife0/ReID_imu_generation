"""01_build_corpus — stage the parallel corpus for every available sequence.

Per sequence: real IMU from the per-subject aux cache, one SMPL-X forward via
_smplx_to_motion.py (external venv), then every generator adapter fanned out
over the requested configs. Resumable via --skip-existing. Sequences run in a
thread pool; the heavy lifting happens in subprocesses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sim2real import corpus as corpus_lib
from src.sim2real.contracts import config_hash as cfg_hash
from src.sim2real.contracts import file_sha1, synth_source, source_to_token
from src.sim2real.splits import load_split, subject_of_sequence

DEFAULT_CONFIGS = [
    "configs/sim2real/generators/globalpose_noise_full.json",
    "configs/sim2real/generators/globalpose_clean.json",
    "configs/sim2real/generators/humogen_default.json",
    "configs/sim2real/generators/naive_default.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="totalcapture")
    parser.add_argument("--sequences", default="all",
                        help="'all' or comma-separated names, e.g. S1_freestyle3,S2_walking1")
    parser.add_argument("--sensor", default="R_LowArm")
    parser.add_argument("--generator-configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--generator-python", default="/home/hrli/data_generation/.venv/bin/python")
    parser.add_argument("--amass-root", default=str(corpus_lib.DEFAULT_AMASS_ROOT))
    parser.add_argument("--lxhong-root", default=str(corpus_lib.DEFAULT_LXHONG_ROOT))
    parser.add_argument("--corpus-root", default="data/interim/sim2real/corpus")
    parser.add_argument("--aux-cache", default="data/interim/sim2real/raw_imu_cache")
    parser.add_argument("--split-file", default="configs/sim2real/splits/totalcapture_subject_v1.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-out", default="outputs/sim2real/corpus_build/summary.json")
    return parser.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def run_subprocess(cmd: list) -> str:
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout tail: {completed.stdout[-2000:]}\nstderr tail: {completed.stderr[-2000:]}"
        )
    return completed.stdout


def build_sequence(seq: str, smplx_path: Path, args, configs: list, split_name: str) -> dict:
    corpus_dir = _abs(args.corpus_root) / args.dataset / seq
    imu_dir = corpus_dir / "imu"
    imu_dir.mkdir(parents=True, exist_ok=True)
    row = {"sequence": seq, "split": split_name, "status": "ok", "streams": [], "errors": []}

    # 1. real IMU
    real_npz = imu_dir / "real.npz"
    if not (args.skip_existing and real_npz.exists()):
        real = corpus_lib.extract_real_imu(_abs(args.aux_cache), seq, args.sensor)
        real.save(real_npz)
        row["real_frames"] = real.num_frames
    else:
        from src.sim2real.contracts import ImuSequence
        row["real_frames"] = ImuSequence.load(real_npz).num_frames

    # 2. motion (one SMPL-X forward, external venv)
    motion_npz = corpus_dir / "motion.npz"
    if not (args.skip_existing and motion_npz.exists()):
        out = run_subprocess([
            args.generator_python, str(REPO_ROOT / "scripts/sim2real/_smplx_to_motion.py"),
            "--smplx", str(smplx_path), "--out", str(motion_npz),
        ])
        row["motion"] = json.loads(out.strip().splitlines()[-1])
    else:
        row["motion"] = {"out": str(motion_npz), "cached": True}

    # 3. generators
    for config_path, config in configs:
        generator = config["generator"]
        token = source_to_token(synth_source(generator, config))
        target_npz = imu_dir / f"{token}.npz"
        if args.skip_existing and target_npz.exists():
            row["streams"].append({"npz": str(target_npz), "config_hash": cfg_hash(config),
                                   "generator": generator, "cached": True})
            continue
        try:
            out = run_subprocess([
                args.generator_python,
                str(REPO_ROOT / "scripts/sim2real/generators" / generator / "generate.py"),
                "--motion", str(motion_npz), "--sensor", args.sensor,
                "--config", str(config_path), "--seed", str(args.seed),
                "--out", str(imu_dir),
            ])
            row["streams"].append(json.loads(out.strip().splitlines()[-1]))
        except RuntimeError as exc:
            row["errors"].append(f"{generator}/{cfg_hash(config)}: {exc}")
            row["status"] = "partial"

    # 4. meta.json
    meta = {
        "sequence": seq,
        "subject": subject_of_sequence(seq),
        "session": seq.split("_", 1)[1],
        "split": split_name,
        "sensor": args.sensor,
        "sources": {
            "smplx": {"path": str(smplx_path), "sha1": file_sha1(smplx_path)},
            "real_imu_cache_root": str(_abs(args.aux_cache)),
        },
        "real_fps": corpus_lib.REAL_IMU_FPS,
        "real_frames": row.get("real_frames"),
        "motion_frames": row.get("motion", {}).get("frames"),
        "motion_fps": row.get("motion", {}).get("fps"),
        "alignment": "tail",
        "streams": row["streams"],
        "errors": row["errors"],
        "gate": None,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (corpus_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return row


def main() -> None:
    args = parse_args()
    split = load_split(_abs(args.split_file))

    # sequence inventory: AMASS SMPL-X x lxhong real IMU
    smplx_map = corpus_lib.list_smplx_sequences(Path(args.amass_root))
    subjects = sorted({name.split("_", 1)[0].lower() for name in smplx_map})
    aux_available = {
        subject: corpus_lib.ensure_aux_cache(Path(args.lxhong_root), _abs(args.aux_cache), subject)
        for subject in subjects
    }

    requested = None if args.sequences == "all" else {s.strip() for s in args.sequences.split(",")}
    work, skipped = [], []
    for seq, smplx_path in sorted(smplx_map.items()):
        if requested is not None and seq not in requested:
            continue
        subject_lower, session = seq.split("_", 1)[0].lower(), seq.split("_", 1)[1]
        if session not in aux_available.get(subject_lower, set()):
            skipped.append({"sequence": seq, "reason": "no real AuxFields IMU in lxhong source"})
            continue
        work.append((seq, smplx_path))

    configs = []
    for path_str in args.generator_configs:
        path = _abs(path_str)
        configs.append((path, json.loads(path.read_text(encoding="utf-8"))))

    print(f"[corpus] {len(work)} sequences to build, {len(skipped)} skipped, "
          f"{len(configs)} generator configs, jobs={args.jobs}")

    rows = []
    def worker(item):
        seq, smplx_path = item
        try:
            return build_sequence(seq, smplx_path, args, configs, split.split_of_sequence(seq))
        except Exception as exc:  # keep the batch going; report at the end
            return {"sequence": seq, "status": "failed", "error": str(exc)[-2000:]}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(worker, work):
            state = row["status"]
            print(f"[corpus] {row['sequence']}: {state}")
            rows.append(row)

    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sensor": args.sensor,
        "seed": args.seed,
        "generator_configs": [str(p) for p, _ in configs],
        "sequences_ok": sum(1 for r in rows if r["status"] == "ok"),
        "sequences_partial": sum(1 for r in rows if r["status"] == "partial"),
        "sequences_failed": sum(1 for r in rows if r["status"] == "failed"),
        "skipped_no_real_imu": skipped,
        "rows": rows,
    }
    summary_path = _abs(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[corpus] ok={summary['sequences_ok']} partial={summary['sequences_partial']} "
          f"failed={summary['sequences_failed']} skipped={len(skipped)}")
    print(f"[corpus] summary -> {summary_path}")
    if summary["sequences_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
