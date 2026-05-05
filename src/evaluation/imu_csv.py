from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ACC_FIELDS = ("acc_x", "acc_y", "acc_z")
GYRO_FIELDS = ("gyro_x", "gyro_y", "gyro_z")
IMU_FIELDS = (
    "frame_idx",
    "quat0",
    "quat1",
    "quat2",
    "quat3",
    *ACC_FIELDS,
    *GYRO_FIELDS,
    "mag_x",
    "mag_y",
    "mag_z",
)
WINDOW_FEATURES = ("mean", "std", "max", "energy")


def evaluate_imu_csv_pair(
    real_csv: str | Path,
    synthetic_csv: str | Path,
    fps: float,
    peak_min_distance_seconds: float = 0.25,
    peak_prominence_fraction: float = 0.10,
    window_seconds: float = 1.0,
    window_overlap: float = 0.5,
) -> dict[str, Any]:
    return evaluate_imu_pair(
        real_csv=real_csv,
        synthetic_csv=synthetic_csv,
        fps=fps,
        peak_min_distance_seconds=peak_min_distance_seconds,
        peak_prominence_fraction=peak_prominence_fraction,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
    )


def load_imu_csv(path: str | Path) -> dict[str, np.ndarray]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no IMU rows")

    missing = [field for field in IMU_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    return {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in IMU_FIELDS
    }


def vector_magnitude(data: dict[str, np.ndarray], fields: tuple[str, str, str]) -> np.ndarray:
    values = np.stack([data[field] for field in fields], axis=1)
    return np.linalg.norm(values, axis=1)


def rmse(real: np.ndarray, synthetic: np.ndarray) -> float:
    _require_same_shape(real, synthetic)
    return float(np.sqrt(np.mean(np.square(real - synthetic))))


def pearson_correlation(real: np.ndarray, synthetic: np.ndarray) -> float:
    _require_same_shape(real, synthetic)
    if np.array_equal(real, synthetic):
        return 1.0
    real_std = float(np.std(real))
    synthetic_std = float(np.std(synthetic))
    if real_std == 0.0 or synthetic_std == 0.0:
        return 0.0
    return float(np.corrcoef(real, synthetic)[0, 1])


def peak_timing_error(
    real: np.ndarray,
    synthetic: np.ndarray,
    fps: float,
    min_distance_seconds: float = 0.25,
    prominence_fraction: float = 0.10,
) -> dict[str, Any]:
    min_distance_frames = max(1, int(round(min_distance_seconds * fps)))
    real_peaks = find_peaks(
        real,
        min_distance_frames=min_distance_frames,
        prominence_fraction=prominence_fraction,
    )
    synthetic_peaks = find_peaks(
        synthetic,
        min_distance_frames=min_distance_frames,
        prominence_fraction=prominence_fraction,
    )
    errors = _match_peak_errors(real_peaks, synthetic_peaks)

    result: dict[str, Any] = {
        "method": "local_maxima_nearest_synthetic_peak_per_real_peak",
        "min_distance_frames": min_distance_frames,
        "prominence_fraction": prominence_fraction,
        "real_peak_count": int(len(real_peaks)),
        "synthetic_peak_count": int(len(synthetic_peaks)),
        "matched_peak_count": int(len(errors)),
        "real_peak_frames": [int(index + 1) for index in real_peaks],
        "synthetic_peak_frames": [int(index + 1) for index in synthetic_peaks],
    }
    if len(errors) == 0:
        result.update(
            {
                "mean_abs_error_frames": None,
                "median_abs_error_frames": None,
                "max_abs_error_frames": None,
                "mean_abs_error_seconds": None,
                "median_abs_error_seconds": None,
                "max_abs_error_seconds": None,
            }
        )
        return result

    errors_array = np.asarray(errors, dtype=np.float64)
    result.update(
        {
            "mean_abs_error_frames": float(np.mean(errors_array)),
            "median_abs_error_frames": float(np.median(errors_array)),
            "max_abs_error_frames": float(np.max(errors_array)),
            "mean_abs_error_seconds": float(np.mean(errors_array) / fps),
            "median_abs_error_seconds": float(np.median(errors_array) / fps),
            "max_abs_error_seconds": float(np.max(errors_array) / fps),
        }
    )
    return result


def find_peaks(
    signal: np.ndarray,
    min_distance_frames: int,
    prominence_fraction: float,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    if values.size == 1:
        return np.asarray([0], dtype=np.int64)

    signal_range = float(np.max(values) - np.min(values))
    threshold = float(np.min(values) + prominence_fraction * signal_range)
    candidates: list[int] = []
    if values[0] > values[1] and values[0] >= threshold:
        candidates.append(0)
    for index in range(1, values.size - 1):
        if values[index] >= values[index - 1] and values[index] > values[index + 1] and values[index] >= threshold:
            candidates.append(index)
    if values[-1] > values[-2] and values[-1] >= threshold:
        candidates.append(values.size - 1)

    if not candidates:
        return np.asarray([], dtype=np.int64)

    strongest_first = sorted(candidates, key=lambda index: values[index], reverse=True)
    selected: list[int] = []
    for index in strongest_first:
        if all(abs(index - chosen) >= min_distance_frames for chosen in selected):
            selected.append(index)
    return np.asarray(sorted(selected), dtype=np.int64)


def psd_distance(
    real: np.ndarray,
    synthetic: np.ndarray,
    fps: float,
    nperseg: int | None = None,
) -> dict[str, Any]:
    _require_same_shape(real, synthetic)
    real_psd = _welch_psd(real, fps=fps, nperseg=nperseg)
    synthetic_psd = _welch_psd(synthetic, fps=fps, nperseg=nperseg)
    eps = np.finfo(np.float64).eps
    real_psd = real_psd / max(float(np.sum(real_psd)), eps)
    synthetic_psd = synthetic_psd / max(float(np.sum(synthetic_psd)), eps)
    distance = rmse(np.log10(real_psd + eps), np.log10(synthetic_psd + eps))
    return {
        "method": "welch_log_normalized_psd_rmse",
        "distance": distance,
        "n_frequency_bins": int(real_psd.size),
    }


def window_feature_distance(
    real: np.ndarray,
    synthetic: np.ndarray,
    fps: float,
    window_seconds: float = 1.0,
    overlap: float = 0.5,
) -> dict[str, Any]:
    _require_same_shape(real, synthetic)
    window_size = max(1, int(round(window_seconds * fps)))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("window overlap must satisfy 0 <= overlap < 1")
    step_size = max(1, int(round(window_size * (1.0 - overlap))))
    real_features = _window_features(real, window_size=window_size, step_size=step_size)
    synthetic_features = _window_features(synthetic, window_size=window_size, step_size=step_size)
    window_count = min(real_features.shape[0], synthetic_features.shape[0])
    if window_count == 0:
        raise ValueError("signal is too short to compute window features")
    real_features = real_features[:window_count]
    synthetic_features = synthetic_features[:window_count]

    feature_distances = {}
    for feature_index, feature_name in enumerate(WINDOW_FEATURES):
        real_distribution = np.sort(real_features[:, feature_index])
        synthetic_distribution = np.sort(synthetic_features[:, feature_index])
        feature_distances[feature_name] = rmse(real_distribution, synthetic_distribution)

    feature_values = np.asarray(list(feature_distances.values()), dtype=np.float64)
    feature_distances["overall_rmse"] = float(np.sqrt(np.mean(np.square(feature_values))))
    return {
        "method": "sorted_window_feature_distribution_rmse",
        "window_size_frames": int(window_size),
        "step_size_frames": int(step_size),
        "window_count": int(window_count),
        "features": list(WINDOW_FEATURES),
        "feature_distance": feature_distances,
    }


def signal_comparison_summary(real: np.ndarray, synthetic: np.ndarray) -> dict[str, Any]:
    _require_same_shape(real, synthetic)
    real_summary = _signal_summary(real)
    synthetic_summary = _signal_summary(synthetic)
    return {
        "real": real_summary,
        "synthetic": synthetic_summary,
        "delta": {
            key: synthetic_summary[key] - real_summary[key]
            for key in real_summary
        },
        "relative_delta": {
            key: _relative_delta(real_summary[key], synthetic_summary[key])
            for key in real_summary
        },
    }


def evaluate_imu_pair(
    real_csv: str | Path,
    synthetic_csv: str | Path,
    fps: float,
    peak_min_distance_seconds: float = 0.25,
    peak_prominence_fraction: float = 0.10,
    window_seconds: float = 1.0,
    window_overlap: float = 0.5,
) -> dict[str, Any]:
    real = load_imu_csv(real_csv)
    synthetic = load_imu_csv(synthetic_csv)
    if real["frame_idx"].shape != synthetic["frame_idx"].shape:
        raise ValueError(
            f"real and synthetic frame counts differ: {real['frame_idx'].size} vs {synthetic['frame_idx'].size}"
        )

    acc_real = vector_magnitude(real, ACC_FIELDS)
    acc_synthetic = vector_magnitude(synthetic, ACC_FIELDS)
    gyro_real = vector_magnitude(real, GYRO_FIELDS)
    gyro_synthetic = vector_magnitude(synthetic, GYRO_FIELDS)

    acc_psd = psd_distance(acc_real, acc_synthetic, fps=fps)
    gyro_psd = psd_distance(gyro_real, gyro_synthetic, fps=fps)
    acc_window = window_feature_distance(
        acc_real,
        acc_synthetic,
        fps=fps,
        window_seconds=window_seconds,
        overlap=window_overlap,
    )
    gyro_window = window_feature_distance(
        gyro_real,
        gyro_synthetic,
        fps=fps,
        window_seconds=window_seconds,
        overlap=window_overlap,
    )

    return _json_safe(
        {
            "metadata": {
                "real_csv": str(real_csv),
                "synthetic_csv": str(synthetic_csv),
                "frame_count": int(real["frame_idx"].size),
                "fps": float(fps),
                "signals": {
                    "acc_magnitude": list(ACC_FIELDS),
                    "gyro_magnitude": list(GYRO_FIELDS),
                },
            },
            "real_vs_synthetic": {
                "acc_magnitude": signal_comparison_summary(acc_real, acc_synthetic),
                "gyro_magnitude": signal_comparison_summary(gyro_real, gyro_synthetic),
            },
            "motion_intensity": {
                "acc_magnitude_rmse": rmse(acc_real, acc_synthetic),
                "gyro_magnitude_rmse": rmse(gyro_real, gyro_synthetic),
            },
            "temporal_consistency": {
                "acc_magnitude_correlation": pearson_correlation(acc_real, acc_synthetic),
                "gyro_magnitude_correlation": pearson_correlation(gyro_real, gyro_synthetic),
            },
            "event_consistency": {
                "acc_peak_timing_error": peak_timing_error(
                    acc_real,
                    acc_synthetic,
                    fps=fps,
                    min_distance_seconds=peak_min_distance_seconds,
                    prominence_fraction=peak_prominence_fraction,
                ),
                "gyro_peak_timing_error": peak_timing_error(
                    gyro_real,
                    gyro_synthetic,
                    fps=fps,
                    min_distance_seconds=peak_min_distance_seconds,
                    prominence_fraction=peak_prominence_fraction,
                ),
            },
            "frequency_structure": {
                "method": acc_psd["method"],
                "acc_magnitude_psd_distance": acc_psd["distance"],
                "gyro_magnitude_psd_distance": gyro_psd["distance"],
                "acc_n_frequency_bins": acc_psd["n_frequency_bins"],
                "gyro_n_frequency_bins": gyro_psd["n_frequency_bins"],
            },
            "window_statistics": {
                "method": acc_window["method"],
                "window_seconds": float(window_seconds),
                "overlap": float(window_overlap),
                "window_size_frames": acc_window["window_size_frames"],
                "step_size_frames": acc_window["step_size_frames"],
                "window_count": min(acc_window["window_count"], gyro_window["window_count"]),
                "features": list(WINDOW_FEATURES),
                "feature_distance": {
                    "acc_magnitude": acc_window["feature_distance"],
                    "gyro_magnitude": gyro_window["feature_distance"],
                },
            },
        }
    )


def write_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _welch_psd(signal: np.ndarray, fps: float, nperseg: int | None = None) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.size < 2:
        return np.asarray([0.0], dtype=np.float64)
    segment_size = min(values.size, nperseg or 256)
    segment_size = max(2, segment_size)
    step_size = max(1, segment_size // 2)
    window = np.hanning(segment_size)
    window_power = max(float(np.sum(window**2)), np.finfo(np.float64).eps)

    spectra = []
    for start in range(0, values.size - segment_size + 1, step_size):
        segment = values[start : start + segment_size]
        segment = segment - np.mean(segment)
        fft_values = np.fft.rfft(segment * window)
        spectra.append(np.square(np.abs(fft_values)) / (fps * window_power))
    if not spectra:
        segment = values - np.mean(values)
        window = np.hanning(values.size)
        window_power = max(float(np.sum(window**2)), np.finfo(np.float64).eps)
        fft_values = np.fft.rfft(segment * window)
        spectra.append(np.square(np.abs(fft_values)) / (fps * window_power))
    return np.mean(np.stack(spectra, axis=0), axis=0)


def _window_features(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    windows = []
    if values.size < window_size:
        windows.append(values)
    else:
        for start in range(0, values.size - window_size + 1, step_size):
            windows.append(values[start : start + window_size])

    features = []
    for window in windows:
        features.append(
            [
                float(np.mean(window)),
                float(np.std(window)),
                float(np.max(window)),
                float(np.mean(np.square(window))),
            ]
        )
    return np.asarray(features, dtype=np.float64)


def _signal_summary(signal: np.ndarray) -> dict[str, float]:
    values = np.asarray(signal, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "mean_square_energy": float(np.mean(np.square(values))),
    }


def _relative_delta(real_value: float, synthetic_value: float) -> float | None:
    if real_value == 0.0:
        return None
    return float((synthetic_value - real_value) / real_value)


def _match_peak_errors(real_peaks: np.ndarray, synthetic_peaks: np.ndarray) -> list[int]:
    if len(real_peaks) == 0 or len(synthetic_peaks) == 0:
        return []
    synthetic_indices = [int(index) for index in synthetic_peaks]
    errors: list[int] = []
    for real_peak in real_peaks:
        nearest = min(synthetic_indices, key=lambda synthetic_peak: abs(synthetic_peak - int(real_peak)))
        errors.append(abs(nearest - int(real_peak)))
    return errors


def _require_same_shape(real: np.ndarray, synthetic: np.ndarray) -> None:
    if real.shape != synthetic.shape:
        raise ValueError(f"signals must have the same shape, got {real.shape} and {synthetic.shape}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
