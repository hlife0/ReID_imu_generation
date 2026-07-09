"""TotalCapture raw-IMU reader + quaternion helpers used by the eval pipelines.

Trimmed to what the two evaluation pipelines and the generation protocol still
use:
  - ``load_totalcapture_raw_sensor_stream`` — read the real ``R_LowArm`` Xsens
    stream for the corpus real-IMU layer (``src/sim2real/corpus.py``).
  - ``matrix_to_quaternion_wxyz`` — rotation matrix -> wxyz quat (globalpose gen).
  - ``enforce_quaternion_continuity`` — sign-flip hemisphere fixup (all gens).

The legacy TotalCapture staging / GT / calibration helpers were removed with the
old generation workflow; recover them from ``archive/full-history-20260709`` if
ever needed.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

# Sensor column order in the raw TotalCapture ``.sensors`` stream.
GLOBALPOSE_SENSOR_ORDER = ("L_LowArm", "R_LowArm", "L_LowLeg", "R_LowLeg", "Head", "Pelvis")


def _sequence_parts(sequence_name: str) -> tuple[str, str, str]:
    subject_raw, session = sequence_name.split("_", 1)
    return subject_raw.upper(), subject_raw.lower(), session


def _session_to_aux_name(session: str) -> str:
    if session.startswith("acting"):
        return session.capitalize()
    return session


def _iter_sensor_lines_from_path(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield line


def _iter_sensor_lines_from_archive(archive_path: Path, member_name: str):
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = archive.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Could not find {member_name} in {archive_path}")
        for raw_line in extracted:
            yield raw_line.decode("utf-8")


def parse_totalcapture_sensor_stream(
    path: str | Path | object,
    sensor_names: list[str] | tuple[str, ...] = GLOBALPOSE_SENSOR_ORDER,
) -> dict[str, object]:
    sensor_names = list(sensor_names)
    sensor_to_index = {name: index for index, name in enumerate(sensor_names)}
    if isinstance(path, (str, Path)):
        csv_path = Path(path)
        iterator = csv_path.open(encoding="utf-8")
        should_close = True
    else:
        csv_path = Path("<stream>")
        iterator = iter(path)
        should_close = False

    def next_line() -> str:
        try:
            return next(iterator)
        except StopIteration:
            return ""

    try:
        header = next_line().strip().split()
        if len(header) < 2:
            raise ValueError(f"Invalid TotalCapture sensor header in {csv_path}")
        num_sensors = int(header[0])
        num_frames = int(header[1])

        quat = np.zeros((num_frames, len(sensor_names), 4), dtype=np.float64)
        acc = np.zeros((num_frames, len(sensor_names), 3), dtype=np.float64)
        gyro = np.zeros((num_frames, len(sensor_names), 3), dtype=np.float64)
        mag = np.zeros((num_frames, len(sensor_names), 3), dtype=np.float64)

        for frame_index in range(num_frames):
            frame_line = next_line()
            if frame_line == "":
                raise ValueError(f"Unexpected end of file in {csv_path}")
            _ = int(frame_line.strip())
            seen = set()
            for _sensor_index in range(num_sensors):
                line = next_line()
                if line == "":
                    raise ValueError(f"Unexpected end of file in {csv_path}")
                parts = line.strip().split("\t")
                name = parts[0].split()[0]
                if name not in sensor_to_index:
                    continue
                if len(parts) < 14:
                    raise ValueError(f"Expected 14 columns for {name} in {csv_path}")
                idx = sensor_to_index[name]
                quat[frame_index, idx] = np.asarray([float(value) for value in parts[1:5]], dtype=np.float64)
                acc[frame_index, idx] = np.asarray([float(value) for value in parts[5:8]], dtype=np.float64)
                gyro[frame_index, idx] = np.asarray([float(value) for value in parts[8:11]], dtype=np.float64)
                mag[frame_index, idx] = np.asarray([float(value) for value in parts[11:14]], dtype=np.float64)
                seen.add(name)
            missing = [name for name in sensor_names if name not in seen]
            if missing:
                raise ValueError(f"Missing sensors in frame {frame_index + 1}: {missing}")
    finally:
        if should_close:
            iterator.close()

    return {
        "frame_count": num_frames,
        "sensor_names": sensor_names,
        "quat": quat,
        "acc": acc,
        "gyro": gyro,
        "mag": mag,
    }


def load_totalcapture_raw_sensor_stream(
    imu_source_root: str | Path,
    sequence_name: str,
    sensor_name: str,
) -> dict[str, np.ndarray]:
    source_root = Path(imu_source_root)
    _, subject, session = _sequence_parts(sequence_name)
    aux_name = _session_to_aux_name(session)
    direct = source_root / subject / f"{aux_name}_Xsens_AuxFields.sensors"
    archive = source_root / f"{subject}_Gyro_Mag.tar.gz"
    member = f"{subject}/{aux_name}_Xsens_AuxFields.sensors"
    if direct.exists():
        lines = _iter_sensor_lines_from_path(direct)
    elif archive.exists():
        lines = _iter_sensor_lines_from_archive(archive, member)
    else:
        raise FileNotFoundError(f"Could not locate raw richer IMU for {sequence_name} in {source_root}")

    parsed = parse_totalcapture_sensor_stream(lines, sensor_names=[sensor_name])
    return {
        "quat": np.asarray(parsed["quat"], dtype=np.float64),
        "acc": np.asarray(parsed["acc"], dtype=np.float64),
        "gyro": np.asarray(parsed["gyro"], dtype=np.float64),
        "mag": np.asarray(parsed["mag"], dtype=np.float64),
    }


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), np.finfo(np.float64).eps)


def enforce_quaternion_continuity(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"expected quaternion array of shape [N, 4], got {values.shape}")
    for frame_index in range(1, values.shape[0]):
        if float(np.dot(values[frame_index - 1], values[frame_index])) < 0.0:
            values[frame_index] *= -1.0
    return values
