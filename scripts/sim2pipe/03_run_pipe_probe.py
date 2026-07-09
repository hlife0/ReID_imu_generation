"""03_run_pipe_probe — drive the matrix through the main project's pipeline.

For each cell (protocol, imu_stream, motion_source, seed):
  1. compose a main-project ``sequences/`` dir realizing the protocol
     (train/val/test IMU stream chosen per subject), via export_protocol;
  2. slice it with the main repo's own ``src.data.slice.totalcapture``;
  3. train the main repo's ``src.engine.train`` (frozen MotionBERT+DeSPITE,
     alignment head only), train-source imu stats;
  4. eval on the held-out real test split with ``src.engine.eval``;
  5. append (val_top1, test_top1, random_line, ...) to the results ledger.

Ledger-resumable: cells already present are skipped. Runs the main modules
in the main project's own environment via bridge (subprocess).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.sim2real.splits import load_split
from src.sim2pipe.bridge import PipePaths, run_main_module
from src.sim2pipe.export import export_protocol
from src.sim2pipe.ledger import append_row, done_keys
from src.sim2pipe.overlay import render_overlay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matrix", default="configs/sim2pipe/matrix_pipe_v1.yaml")
    p.add_argument("--paths", default="configs/sim2pipe/paths.yaml")
    p.add_argument("--overlay-template", default="configs/sim2pipe/overlays/pipe_probe_tc.yaml")
    p.add_argument("--corpus-root", default="data/interim/sim2real/corpus/totalcapture")
    p.add_argument("--work-root", default="data/interim/sim2pipe/probe")
    p.add_argument("--outputs-root", default="outputs/sim2pipe")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--only", nargs="*", help="restrict to stream tokens (e.g. real synth_naive_8f7d9e76)")
    p.add_argument("--seeds", nargs="*", type=int, help="override matrix seeds")
    p.add_argument("--shuffle-control", action="store_true",
                   help="add --shuffle_video_in_batch to train (shuffled-pairs control)")
    p.add_argument("--destroy-pairing", action="store_true",
                   help="add --destroy_pairing_in_batch to train (GENUINE pairing-destruction "
                        "floor: shuffle video without relabeling targets). Requires the main "
                        "repo to have mainproj_patches/destroy_pairing_in_batch.patch applied. "
                        "Use a distinct --outputs-root since the ledger cell key ignores this flag.")
    p.add_argument("--imu-stats-json", default="",
                   help="normalization ablation: use these fixed imu stats for train+eval "
                        "instead of computing from the (synthetic) train source")
    p.add_argument("--run-tag-suffix", default="",
                   help="appended to each cell tag / work dir to keep ablation runs distinct")
    return p.parse_args()


def _abs(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_cell(args, paths: PipePaths, split, cell: dict, seed: int, ledger: Path) -> dict:
    protocol = cell["protocol"]
    imu_stream = cell["imu_stream"]
    motion_source = cell["motion_source"]
    stream_tok = Path(imu_stream).stem

    tag = f"{protocol}__{stream_tok}__{motion_source}__s{seed}"
    if args.shuffle_control:
        tag += "__shuf"
    if args.destroy_pairing:
        tag += "__destroy"
    if args.run_tag_suffix:
        tag += f"__{args.run_tag_suffix}"
    work = _abs(args.work_root) / tag
    slice_out = work / "slice"
    train_out = work / "train"

    # 1. compose sequences/ for this protocol
    export_protocol(
        _abs(args.corpus_root), protocol, imu_stream, motion_source, split, work,
        camera="cam1",
    )

    # 2. slice via the main repo (overlay injects root/out_dir/subjects)
    overlay_path = work / "overlay.yaml"
    render_overlay(
        _abs(args.overlay_template),
        {
            "project": f"sim2pipe_{tag}",
            "work_dir": str(work),
            "slice.root": str(work),
            "slice.out_dir": str(slice_out),
            "train.seed": seed,
            "slice.train_subjects": ",".join(split.train_subjects),
            "slice.val_subjects": ",".join(split.val_subjects),
            "slice.test_subjects": ",".join(split.test_subjects),
        },
        overlay_path,
    )
    r = run_main_module(paths, "src.data.slice.totalcapture", ["--config", str(overlay_path)],
                        log_path=work / "slice.log")
    if r.returncode != 0:
        raise RuntimeError(f"slice failed for {tag}; see {work/'slice.log'}")

    train_csv = slice_out / "windows_train.csv"
    val_csv = slice_out / "windows_val.csv"
    test_csv = slice_out / "windows_test.csv"

    # 3. train — frozen backbones, alignment head only, train-source stats
    mb_ckpt = paths.motionbert_ckpt
    train_argv = [
        "--train_csv", str(train_csv), "--val_csv", str(val_csv), "--data_root", str(slice_out),
        "--motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml",
        "--motionbert_ckpt", str(mb_ckpt), "--imu_ckpt", str(paths.imu_ckpt),
        "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers), "--compute_imu_stats",
        "--imu_sensor", "R_LowArm", "--repeat_single_sensor", "4",
        "--freeze_backbone_epochs", str(args.epochs),
        "--device", args.device, "--seed", str(seed),
        "--output_root", str(train_out), "--run_name", tag,
    ]
    if args.imu_stats_json:
        # normalization ablation: fixed stats override the train-source computation
        train_argv.remove("--compute_imu_stats")
        train_argv += ["--imu_stats_json", str(_abs(args.imu_stats_json))]
    mb_root = paths.extra.get("motionbert_root")
    if mb_root:
        train_argv += ["--motionbert_root", str(mb_root)]
    if args.shuffle_control:
        train_argv.append("--shuffle_video_in_batch")
    if args.destroy_pairing:
        train_argv.append("--destroy_pairing_in_batch")
    r = run_main_module(paths, "src.engine.train", train_argv, log_path=work / "train.log")
    if r.returncode != 0:
        raise RuntimeError(f"train failed for {tag}; see {work/'train.log'}")

    save_dir = train_out / tag
    train_metrics = _read_json(save_dir / "metrics.json")
    best_pt = save_dir / "best.pt"
    stats_json = save_dir / "imu_stats.json"

    # 4. eval on held-out real test split
    eval_json = work / "eval_test.json"
    eval_argv = [
        "--test_csv", str(test_csv), "--data_root", str(slice_out),
        "--checkpoint", str(best_pt), "--imu_stats_json", str(stats_json),
        "--motionbert_config", "configs/pose3d/MB_ft_h36m_global_lite.yaml",
        "--motionbert_ckpt", str(mb_ckpt),
        "--imu_sensor", "R_LowArm", "--repeat_single_sensor", "4",
        "--batch_size", str(args.batch_size), "--device", args.device,
        "--save_json", str(eval_json),
    ]
    if mb_root:
        eval_argv += ["--motionbert_root", str(mb_root)]
    r = run_main_module(paths, "src.engine.eval", eval_argv, log_path=work / "eval.log")
    if r.returncode != 0:
        raise RuntimeError(f"eval failed for {tag}; see {work/'eval.log'}")
    test_metrics = _read_json(eval_json)

    return {
        "protocol": protocol,
        "imu_stream": stream_tok,
        "motion_source": motion_source,
        "seed": seed,
        "val_top1": float(train_metrics.get("best_val_top1", -1.0)),
        "test_top1": float(test_metrics.get("top1", -1.0)),
        "test_num_windows": int(test_metrics.get("num_windows", 0)),
        "random_line": 1.0 / float(args.batch_size),
        "shuffle_control": bool(args.shuffle_control),
        "destroy_pairing": bool(args.destroy_pairing),
        "norm_source": "real_fixed" if args.imu_stats_json else "synth_trainsrc",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "work": str(work),
    }


def main() -> None:
    args = parse_args()
    paths = PipePaths.load(_abs(args.paths))
    matrix = yaml.safe_load(_abs(args.matrix).read_text(encoding="utf-8"))
    split = load_split(_abs(matrix["split"]))
    motion_source = matrix["motion_source"]
    seeds = args.seeds if args.seeds else matrix["seeds"]

    benchmark_id = matrix["benchmark_id"]
    ledger = _abs(args.outputs_root) / benchmark_id / "results.jsonl"
    done = done_keys(ledger)

    cells = []
    for c in matrix["cells"]:
        c = {**c, "motion_source": motion_source}
        if args.only and Path(c["imu_stream"]).stem not in args.only:
            continue
        cells.append(c)

    total = len(cells) * len(seeds)
    idx = 0
    for c in cells:
        for seed in seeds:
            idx += 1
            key = (c["protocol"], Path(c["imu_stream"]).stem, motion_source, seed)
            if key in done:
                print(f"[{idx}/{total}] skip (done): {key}")
                continue
            t0 = time.time()
            print(f"[{idx}/{total}] run: {key}")
            row = run_cell(args, paths, split, c, seed, ledger)
            row["seconds"] = round(time.time() - t0, 1)
            append_row(ledger, row)
            print(f"    val_top1={row['val_top1']:.4f} test_top1={row['test_top1']:.4f} "
                  f"(random {row['random_line']:.4f}) in {row['seconds']}s")

    print(f"Done. Ledger: {ledger}")


if __name__ == "__main__":
    main()
