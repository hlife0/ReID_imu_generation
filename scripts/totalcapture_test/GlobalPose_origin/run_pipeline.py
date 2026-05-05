from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_imu_csv_pair, write_metrics_json

TARGET_SENSOR_ORDER = ("R_LowArm",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GlobalPose-origin TotalCapture reference workflow end-to-end.")
    parser.add_argument(
        "--output-root",
        default="outputs/totalcapture_test/GlobalPose_origin",
    )
    parser.add_argument(
        "--data-root",
        default="data",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed",
    )
    parser.add_argument(
        "--globalpose-python",
        default="/home/hrli/data_generation/.venv/bin/python",
        help="Python interpreter used for the heavy helper that depends on torch.",
    )
    parser.add_argument("--sequence-name", default="S1_freestyle3")
    parser.add_argument("--sensor-name", default=TARGET_SENSOR_ORDER[0])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    data_root = (REPO_ROOT / args.data_root).resolve() if not Path(args.data_root).is_absolute() else Path(args.data_root)
    processed_root = (REPO_ROOT / args.processed_root).resolve() if not Path(args.processed_root).is_absolute() else Path(args.processed_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    helper = Path(__file__).resolve().parent / "_run_pipeline_impl.py"
    plot_script = REPO_ROOT / "scripts" / "totalcapture_test" / "plot_imu_comparison.py"

    run_command(
        [
            args.globalpose_python,
            str(helper),
            "--processed-root",
            str(processed_root),
            "--output-root",
            str(output_root),
            "--seed",
            str(args.seed),
            "--sequence-name",
            args.sequence_name,
            "--sensor-name",
            args.sensor_name,
        ]
    )

    csv_dir = output_root / "csv"
    plots_dir = output_root / "plots"
    metrics_dir = output_root / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    sensor_metrics = {}
    sensor_order = [args.sensor_name]
    for sensor_name in sensor_order:
        paths = build_sensor_artifact_paths(
            repo_root=REPO_ROOT,
            data_root=data_root,
            output_root=output_root,
            sequence_name=args.sequence_name,
            sensor_name=sensor_name,
        )
        raw_csv = paths["raw_output_csv"]
        generated_csv = paths["generated_output_csv"]
        plot_png = paths["plot_png"]
        raw_vs_generated_json = paths["raw_vs_generated_json"]

        run_command(
            [
                sys.executable,
                str(plot_script),
                "--real-csv",
                str(raw_csv),
                "--synthetic-csv",
                str(generated_csv),
                "--output-png",
                str(plot_png),
                "--title",
                f"GlobalPose_origin {sensor_name}: raw vs generated",
                "--plot-python",
                args.globalpose_python,
            ]
        )
        sensor_metrics[sensor_name] = {
            "raw_vs_generated": evaluate_sensor_metrics(
                reference_csv=raw_csv,
                candidate_csv=generated_csv,
                output_json=raw_vs_generated_json,
                fps=60.0,
            ),
        }
    report_path = output_root / "report.md"
    report_path.write_text(build_report(output_root=output_root, seed=args.seed, sensor_metrics=sensor_metrics), encoding="utf-8")
    print(report_path)


def build_sensor_artifact_paths(
    repo_root: Path,
    data_root: Path,
    output_root: Path,
    sequence_name: str,
    sensor_name: str,
) -> dict[str, Path]:
    csv_dir = output_root / "csv"
    plots_dir = output_root / "plots"
    metrics_dir = output_root / "metrics"
    return {
        "raw_sample_dir": data_root / "raw" / "totalcapture" / sequence_name,
        "processed_triplet_dir": data_root / "processed" / "totalcapture_test" / sequence_name,
        "processed_video_mp4": data_root / "processed" / "totalcapture_test" / sequence_name / f"TC_{sequence_name}_cam1.mp4",
        "processed_imu_csv": data_root / "processed" / "totalcapture_test" / sequence_name / f"{sequence_name.lower()}_{sensor_name}.csv",
        "processed_smplx_npz": data_root / "processed" / "totalcapture_test" / sequence_name / f"{sequence_name.lower()}_smplx.npz",
        "raw_output_csv": csv_dir / f"{sensor_name}_raw.csv",
        "generated_output_csv": csv_dir / f"{sensor_name}_generated.csv",
        "plot_png": plots_dir / f"{sensor_name}_raw_vs_generated.png",
        "raw_vs_generated_json": metrics_dir / f"{sensor_name}_raw_vs_generated_metrics.json",
    }


def run_command(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or f"Command failed: {cmd}")


def evaluate_sensor_metrics(
    reference_csv: Path,
    candidate_csv: Path,
    output_json: Path,
    fps: float,
) -> dict[str, object]:
    metrics = evaluate_imu_csv_pair(
        real_csv=reference_csv,
        synthetic_csv=candidate_csv,
        fps=fps,
    )
    write_metrics_json(metrics, output_json)
    return metrics


def build_report(
    output_root: Path,
    seed: int,
    sensor_metrics: dict[str, dict[str, object]],
) -> str:
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    raw_sample_dir = Path(manifest["raw_sample_dir"])
    processed_triplet_dir = Path(manifest["processed_triplet_dir"])
    processed_smplx_npz = Path(manifest["processed_smplx_npz"])
    sensor_label = ", ".join(manifest["sensor_order"])
    lines = [
        "# TotalCapture 当前运行报告",
        "",
        "这份报告只描述当前主流程这一件事：",
        "`data/raw/totalcapture/S1_freestyle3/` 里的原始样例 -> ",
        "`data/processed/totalcapture_test/S1_freestyle3/` 标准三元组 -> ",
        "`R_LowArm` synthetic IMU 生成与对比。",
        "",
        "## 这次运行用到了什么",
        "",
        f"- 原始样例目录：`{raw_sample_dir}`",
        f"- 标准三元组目录：`{processed_triplet_dir}`",
        f"- 标准三元组里的视频：`{manifest['processed_triplet_dir']}/TC_S1_freestyle3_cam1.mp4`",
        f"- 标准三元组里的 IMU：`{manifest['processed_imu_csv']}`",
        f"- 标准三元组里的 SMPL-X：`{processed_smplx_npz}`",
        f"- 输出目录：`{output_root}`",
        f"- 随机种子：`{seed}`",
        f"- 当前只处理的传感器：`{sensor_label}`",
        "",
        "## 当前流程到底在干什么",
        "",
        "1. `prepare_triplet.py` 先准备标准三元组。",
        "2. 当前 generation 只读取三元组里的 `s1_freestyle3_smplx.npz`。",
        "3. 用 `smplx` 做人体前向，恢复每一帧的人体关节运动。",
        "4. 从关节运动里提取右手手腕传感器轨迹。",
        "5. 用当前主流程内置的 `GlobalPose` 官方 synthesis 核心生成 `generated` IMU。",
        "6. 把三元组里的 `R_LowArm.csv` 当作 `raw` 基准。",
        "7. 只做一组对比：`raw vs generated`。",
        "",
        "## 生成输出",
        "",
        f"- `raw` CSV：`{manifest['csv_files'][sensor_label]['raw_csv']}`",
        f"- `generated` CSV：`{manifest['csv_files'][sensor_label]['generated_csv']}`",
        f"- 对比图：`{display_path(output_root / 'plots' / f'{sensor_label}_raw_vs_generated.png')}`",
        f"- 指标 JSON：`{display_path(output_root / 'metrics' / f'{sensor_label}_raw_vs_generated_metrics.json')}`",
        "",
        "## 指标摘要",
        "",
        "| 传感器 | 对比 | Acc corr | Gyro corr | Acc RMSE | Gyro RMSE | Gyro mean 相对差 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sensor_name in manifest["sensor_order"]:
        metrics = sensor_metrics[sensor_name]["raw_vs_generated"]
        acc_corr = metrics["temporal_consistency"]["acc_magnitude_correlation"]
        gyro_corr = metrics["temporal_consistency"]["gyro_magnitude_correlation"]
        acc_rmse = metrics["motion_intensity"]["acc_magnitude_rmse"]
        gyro_rmse = metrics["motion_intensity"]["gyro_magnitude_rmse"]
        gyro_mean_rel = metrics["real_vs_synthetic"]["gyro_magnitude"]["relative_delta"]["mean"]
        lines.append(
            f"| {sensor_name} | raw vs generated | `{acc_corr:.4f}` | `{gyro_corr:.4f}` | `{acc_rmse:.4f}` | `{gyro_rmse:.4f}` | `{gyro_mean_rel:+.4f} ({gyro_mean_rel * 100:+.2f}%)` |"
        )

    lines.extend(
        [
            "",
            "## 原始与生成统计",
            "",
        ]
    )

    for sensor_name in manifest["sensor_order"]:
        metrics = sensor_metrics[sensor_name]["raw_vs_generated"]
        for signal_name, label in [("acc_magnitude", "Acc magnitude"), ("gyro_magnitude", "Gyro magnitude")]:
            summary = metrics["real_vs_synthetic"][signal_name]
            lines.extend(
                [
                    f"### {sensor_name} {label}",
                    "",
                    "| 统计量 | raw | generated | generated - raw | (generated - raw) / raw |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for stat_name in ["mean", "std", "min", "median", "max", "rms", "mean_square_energy"]:
                raw_value = summary["real"][stat_name]
                generated_value = summary["synthetic"][stat_name]
                delta_value = summary["delta"][stat_name]
                relative_value = summary["relative_delta"][stat_name]
                if relative_value is None:
                    relative_text = "`None`"
                else:
                    relative_text = f"`{relative_value:+.4f} ({relative_value * 100:+.2f}%)`"
                lines.append(
                    f"| {stat_name} | `{raw_value:.4f}` | `{generated_value:.4f}` | `{delta_value:+.4f}` | {relative_text} |"
                )
            lines.append("")

    lines.extend(
        [
            "## 事件一致性",
            "",
            "| 传感器 | 信号 | raw 峰数 | generated 峰数 | 平均误差(frames) | 平均误差(seconds) | 中位误差(frames) | 最大误差(frames) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for sensor_name in manifest["sensor_order"]:
        metrics = sensor_metrics[sensor_name]["raw_vs_generated"]["event_consistency"]
        for key, label in [("acc_peak_timing_error", "acc"), ("gyro_peak_timing_error", "gyro")]:
            item = metrics[key]
            lines.append(
                f"| {sensor_name} | {label} | `{item['real_peak_count']}` | `{item['synthetic_peak_count']}` | `{item['mean_abs_error_frames']:.4f}` | `{item['mean_abs_error_seconds']:.4f}` | `{item['median_abs_error_frames']:.4f}` | `{item['max_abs_error_frames']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## 频域结构",
            "",
            "| 传感器 | Acc PSD distance | Gyro PSD distance |",
            "| --- | ---: | ---: |",
        ]
    )
    for sensor_name in manifest["sensor_order"]:
        frequency = sensor_metrics[sensor_name]["raw_vs_generated"]["frequency_structure"]
        lines.append(
            f"| {sensor_name} | `{frequency['acc_magnitude_psd_distance']:.4f}` | `{frequency['gyro_magnitude_psd_distance']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## 窗口统计",
            "",
            "| 传感器 | 信号 | window_seconds | window_count | mean | std | max | energy | overall_rmse |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for sensor_name in manifest["sensor_order"]:
        window = sensor_metrics[sensor_name]["raw_vs_generated"]["window_statistics"]
        for signal_name, label in [("acc_magnitude", "acc"), ("gyro_magnitude", "gyro")]:
            fd = window["feature_distance"][signal_name]
            lines.append(
                f"| {sensor_name} | {label} | `{window['window_seconds']:.2f}` | `{window['window_count']}` | `{fd['mean']:.4f}` | `{fd['std']:.4f}` | `{fd['max']:.4f}` | `{fd['energy']:.4f}` | `{fd['overall_rmse']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## 怎么读这些数",
            "",
            "- 所有数值都以 `raw` 为基准进行比较。",
            "- `raw vs generated` 反映 SMPL-X 生成结果相对 raw 的偏差。",
            "- `Acc/Gyro corr` 越高越好，表示与 raw 的起伏节奏更一致。",
            "- `Acc/Gyro RMSE` 越低越好，表示与 raw 的强度更接近。",
            "- `Gyro mean 相对差` 是 `(candidate - raw) / raw`，越接近 0 越好。",
            "",
            "## 边界说明",
            "",
            "- 当前 generation 不依赖 `legacy`。",
            "- 当前 generation 只依赖 `processed` 三元组。",
            "- 当前评测只比较 `raw` 和 `generated`，没有别的隐藏数据源。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
