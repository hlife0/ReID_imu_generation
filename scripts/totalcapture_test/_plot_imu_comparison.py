from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIELDS = ["quat0", "quat1", "quat2", "quat3", "acc_x", "acc_y", "acc_z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real vs synthetic IMU CSV comparison.")
    parser.add_argument("--real-csv", required=True)
    parser.add_argument("--synthetic-csv", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--title", default="Real vs Synthetic IMU")
    return parser.parse_args()


def load_csv(path: str) -> dict[str, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {field: np.array([float(row[field]) for row in rows], dtype=np.float32) for field in ["frame_idx", *FIELDS]}
    return data


def main() -> None:
    args = parse_args()
    real = load_csv(args.real_csv)
    synthetic = load_csv(args.synthetic_csv)

    fig, axes = plt.subplots(len(FIELDS), 1, figsize=(14, 16), sharex=True)
    fig.suptitle(args.title)
    for axis, field in zip(axes, FIELDS):
        axis.plot(real["frame_idx"], real[field], label="real", linewidth=1.2)
        axis.plot(synthetic["frame_idx"], synthetic[field], label="synthetic", linewidth=1.0)
        axis.set_ylabel(field)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("frame_idx")
    fig.tight_layout()
    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
