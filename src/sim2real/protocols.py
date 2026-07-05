"""L2 protocol runners — M3 (trtr/tstr) + M4 (mix/pretrain_finetune), implemented.

Torch-dependent; runs in the external generator venv. One "cell" =
(protocol, train composition, seed) -> probe trained per the protocol,
evaluated on the REAL test shard, appended as one row to the results ledger.

Normalization rule per cell: channel stats are computed over the assembled
TRAINING set (union of its sources; for pretrain_finetune, the PRETRAIN set —
the model's input convention is fixed in stage A and must not change under
finetuning) and applied unchanged to val/test. Stats are recorded in the row.

Discipline encoded here, not left to habit: val/test shards are always real;
hyperparameters are fixed constants; the test shard is touched exactly once,
after training finishes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .contracts import git_sha, write_manifest
from .windows import load_shard

PROTOCOLS = ("trtr", "tstr", "mix", "pretrain_finetune")

FINETUNE_LR = 1e-4


def make_cell_id(protocol: str, train_tokens, seed: int, *,
                 pretrain_tokens=None, real_fraction: float = 1.0,
                 shuffle_pairs: bool = False) -> str:
    parts = [protocol, "+".join(sorted(train_tokens))]
    if pretrain_tokens:
        parts.append("pre_" + "+".join(sorted(pretrain_tokens)))
    if real_fraction != 1.0:
        parts.append(f"rf{real_fraction:g}")
    parts.append(f"s{seed}")
    if shuffle_pairs:
        parts.append("shuf")
    return "__".join(parts)


def _channel_stats(imu: np.ndarray) -> tuple:
    flat = imu.reshape(-1, imu.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0, dtype=np.float64)
    std = flat.std(axis=0, dtype=np.float64)
    std[std == 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize(imu: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((imu - mean) / std).astype(np.float32)


def _assemble(windows_dir: Path, tokens, real_fraction: float, seed: int) -> tuple:
    imu_parts, motion_parts = [], []
    for token in tokens:
        _, arrays = load_shard(windows_dir / f"train__{token}.npz")
        imu, motion = arrays["imu"], arrays["motion"]
        if token == "real" and real_fraction < 1.0:
            rng = np.random.default_rng(seed + 4242)
            keep = np.sort(rng.choice(len(imu), max(1, int(len(imu) * real_fraction)), replace=False))
            imu, motion = imu[keep], motion[keep]
        imu_parts.append(imu)
        motion_parts.append(motion)
    return np.concatenate(imu_parts, axis=0), np.concatenate(motion_parts, axis=0)


def run_cell(
    windows_dir: Path,
    results_jsonl: Path,
    cells_dir: Path,
    protocol: str,
    train_tokens,
    seed: int,
    *,
    pretrain_tokens=None,
    real_fraction: float = 1.0,
    shuffle_pairs: bool = False,
    device: str = "cuda",
) -> dict:
    import torch

    from .probe.retrieve import retrieval_metrics
    from .probe.train import train_probe

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    windows_dir = Path(windows_dir)
    t_start = time.time()

    _, val_arrays = load_shard(windows_dir / "val__real.npz")
    test_header, test_arrays = load_shard(windows_dir / "test__real.npz")

    if protocol == "pretrain_finetune":
        if not pretrain_tokens:
            raise ValueError("pretrain_finetune requires pretrain_tokens")
        pre_imu, pre_motion = _assemble(windows_dir, pretrain_tokens, 1.0, seed)
        mean, std = _channel_stats(pre_imu)
        val_imu = _normalize(val_arrays["imu"], mean, std)
        stage_a = train_probe(
            _normalize(pre_imu, mean, std), pre_motion, val_imu, val_arrays["motion"],
            seed=seed, device=device, shuffle_pairs=shuffle_pairs,
        )
        fin_imu, fin_motion = _assemble(windows_dir, train_tokens, real_fraction, seed)
        result = train_probe(
            _normalize(fin_imu, mean, std), fin_motion, val_imu, val_arrays["motion"],
            seed=seed + 1, device=device, shuffle_pairs=shuffle_pairs,
            lr=FINETUNE_LR, init_state=stage_a["state"],
        )
        train_windows = len(fin_imu)
        pretrain_windows = len(pre_imu)
    else:
        tr_imu, tr_motion = _assemble(windows_dir, train_tokens, real_fraction, seed)
        mean, std = _channel_stats(tr_imu)
        val_imu = _normalize(val_arrays["imu"], mean, std)
        result = train_probe(
            _normalize(tr_imu, mean, std), tr_motion, val_imu, val_arrays["motion"],
            seed=seed, device=device, shuffle_pairs=shuffle_pairs,
        )
        train_windows = len(tr_imu)
        pretrain_windows = 0

    test_imu = torch.tensor(_normalize(test_arrays["imu"], mean, std),
                            dtype=torch.float32, device=device)
    test_motion = torch.tensor(test_arrays["motion"], dtype=torch.float32, device=device)
    metrics = retrieval_metrics(result["imu_tower"], result["motion_tower"],
                                test_imu, test_motion, test_arrays["sequence"])

    cell_id = make_cell_id(protocol, train_tokens, seed, pretrain_tokens=pretrain_tokens,
                           real_fraction=real_fraction, shuffle_pairs=shuffle_pairs)
    cell_dir = Path(cells_dir) / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["state"], cell_dir / "best.pt")
    write_manifest(
        cell_dir,
        stage="l2_cell",
        config={
            "protocol": protocol, "train": sorted(train_tokens),
            "pretrain": sorted(pretrain_tokens) if pretrain_tokens else None,
            "real_fraction": real_fraction, "seed": seed, "shuffle_pairs": shuffle_pairs,
        },
        seed=seed,
        extra={"benchmark_id": test_header["benchmark_id"]},
    )

    row = {
        "cell_id": cell_id,
        "protocol": protocol,
        "train": sorted(train_tokens),
        "pretrain": sorted(pretrain_tokens) if pretrain_tokens else None,
        "real_fraction": real_fraction,
        "seed": seed,
        "shuffle_pairs": shuffle_pairs,
        "r_at_1": metrics["r_at_1"],
        "r_at_5": metrics["r_at_5"],
        "gallery_size": metrics["gallery_size"],
        "chance_r_at_1": metrics["chance_r_at_1"],
        "per_sequence_r1": metrics["per_sequence"],
        "val_r1": round(result["best_val_r1"], 4),
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "train_windows": train_windows,
        "pretrain_windows": pretrain_windows,
        "norm_stats": {"mean": mean.tolist(), "std": std.tolist()},
        "wall_s": round(time.time() - t_start, 1),
        "git_sha": git_sha(),
        "artifact_dir": str(cell_dir),
    }
    results_path = Path(results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def completed_cell_ids(results_jsonl: Path) -> set:
    path = Path(results_jsonl)
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["cell_id"])
    return ids
