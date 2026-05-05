from __future__ import annotations

import csv
import shutil
import tarfile
from pathlib import Path

import numpy as np


GLOBALPOSE_SENSOR_ORDER = ("L_LowArm", "R_LowArm", "L_LowLeg", "R_LowLeg", "Head", "Pelvis")
_GLOBALPOSE_GT_MAPPING = {
    "L_LowArm": "LeftForeArm",
    "R_LowArm": "RightForeArm",
    "L_LowLeg": "LeftLeg",
    "R_LowLeg": "RightLeg",
    "Head": "Head",
    "Pelvis": "Hips",
}
_GLOBAL_TO_MODEL_MATRIX = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
_BONE_TO_MODEL_MATRIX = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _sequence_parts(sequence_name: str) -> tuple[str, str, str]:
    subject_raw, session = sequence_name.split("_", 1)
    return subject_raw.upper(), subject_raw.lower(), session


def _session_to_aux_name(session: str) -> str:
    if session.startswith("acting"):
        return session.capitalize()
    return session


def _extract_member(archive_path: Path, member_name: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        extracted = archive.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Could not find {member_name} in {archive_path}")
        with dst.open("wb") as handle:
            shutil.copyfileobj(extracted, handle)


def stage_totalcapture_raw_sample(
    totalcapture_source_root: str | Path,
    imu_source_root: str | Path,
    raw_root: str | Path,
    sequence_name: str,
    camera_names: tuple[str, ...] = ("cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8"),
) -> dict[str, str]:
    totalcapture_root = Path(totalcapture_source_root)
    imu_root = Path(imu_source_root)
    raw_base = Path(raw_root)
    subject_upper, subject_lower, session = _sequence_parts(sequence_name)
    sequence_dir = raw_base / "totalcapture" / f"{subject_upper}_{session}"
    if sequence_dir.exists():
        shutil.rmtree(sequence_dir)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    video_archive = totalcapture_root / f"{subject_lower}_{session}.tar.gz"
    for camera_name in camera_names:
        _extract_member(
            video_archive,
            f"{session}/TC_{subject_upper}_{session}_{camera_name}.mp4",
            sequence_dir / f"TC_{subject_upper}_{session}_{camera_name}.mp4",
        )

    imu_archive = imu_root / f"{subject_lower}_Gyro_Mag.tar.gz"
    aux_name = _session_to_aux_name(session)
    _extract_member(
        imu_archive,
        f"{subject_lower}/{aux_name}_Xsens_AuxFields.sensors",
        sequence_dir / f"{subject_lower}_{session}_Xsens_AuxFields.sensors",
    )

    imu_meta_archive = totalcapture_root / f"{subject_upper}_imu.tar.gz"
    if not imu_meta_archive.exists():
        imu_meta_archive = imu_root / f"{subject_upper}_imu.tar.gz"
    _extract_member(
        imu_meta_archive,
        f"{subject_lower}/{subject_lower}_{session}_Xsens.sensors",
        sequence_dir / f"{subject_lower}_{session}_Xsens.sensors",
    )
    _extract_member(
        imu_meta_archive,
        f"{subject_lower}/{subject_lower}_{session}_calib_imu_bone.txt",
        sequence_dir / f"{subject_lower}_{session}_calib_imu_bone.txt",
    )
    _extract_member(
        imu_meta_archive,
        f"{subject_lower}/{subject_lower}_{session}_calib_imu_ref.txt",
        sequence_dir / f"{subject_lower}_{session}_calib_imu_ref.txt",
    )

    gt_archive = totalcapture_root / f"{subject_lower}_vicon_pos_ori.tar.gz"
    _extract_member(
        gt_archive,
        f"{subject_upper}/{session}/gt_skel_gbl_pos.txt",
        sequence_dir / "gt_skel_gbl_pos.txt",
    )
    _extract_member(
        gt_archive,
        f"{subject_upper}/{session}/gt_skel_gbl_ori.txt",
        sequence_dir / "gt_skel_gbl_ori.txt",
    )

    return {
        "sequence_dir": str(sequence_dir),
        "raw_aux_sensors": str(sequence_dir / f"{subject_lower}_{session}_Xsens_AuxFields.sensors"),
        "raw_xsens_sensors": str(sequence_dir / f"{subject_lower}_{session}_Xsens.sensors"),
        "raw_calib_bone": str(sequence_dir / f"{subject_lower}_{session}_calib_imu_bone.txt"),
        "raw_calib_ref": str(sequence_dir / f"{subject_lower}_{session}_calib_imu_ref.txt"),
        "gt_pos": str(sequence_dir / "gt_skel_gbl_pos.txt"),
        "gt_ori": str(sequence_dir / "gt_skel_gbl_ori.txt"),
    }


def prepare_totalcapture_processed_triplet(
    raw_sample_dir: str | Path,
    processed_root: str | Path,
    sequence_name: str,
    sensor_name: str,
    smplx_source: str | Path,
    camera_name: str = "cam1",
) -> dict[str, str]:
    raw_dir = Path(raw_sample_dir)
    processed_base = Path(processed_root)
    subject_upper, subject_lower, session = _sequence_parts(sequence_name)
    triplet_dir = processed_base / "totalcapture_test" / f"{subject_upper}_{session}"
    if triplet_dir.exists():
        shutil.rmtree(triplet_dir)
    triplet_dir.mkdir(parents=True, exist_ok=True)

    video_src = raw_dir / f"TC_{subject_upper}_{session}_{camera_name}.mp4"
    video_dst = triplet_dir / video_src.name
    shutil.copy2(video_src, video_dst)

    imu_dst = triplet_dir / f"{subject_lower}_{session}_{sensor_name}.csv"
    write_single_sensor_csv_from_aux(
        aux_path=raw_dir / f"{subject_lower}_{session}_Xsens_AuxFields.sensors",
        sensor_name=sensor_name,
        output_csv=imu_dst,
    )

    smplx_src = Path(smplx_source)
    smplx_dst = triplet_dir / f"{subject_lower}_{session}_smplx.npz"
    shutil.copy2(smplx_src, smplx_dst)

    return {
        "triplet_dir": str(triplet_dir),
        "video_mp4": str(video_dst),
        "imu_csv": str(imu_dst),
        "smplx_npz": str(smplx_dst),
    }


def parse_totalcapture_sensor_stream(
    path: str | Path | object,
    sensor_names: list[str] | tuple[str, ...] = GLOBALPOSE_SENSOR_ORDER,
) -> dict[str, object]:
    sensor_names = list(sensor_names)
    sensor_to_index = {name: index for index, name in enumerate(sensor_names)}
    if isinstance(path, (str, Path)):
        csv_path = Path(path)
        handle = csv_path.open(encoding="utf-8")
        should_close = True
        iterator = handle
    else:
        csv_path = Path("<stream>")
        handle = None
        should_close = False
        iterator = iter(path)

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


def write_single_sensor_csv_from_aux(
    aux_path: str | Path,
    sensor_name: str,
    output_csv: str | Path,
) -> None:
    parsed = parse_totalcapture_sensor_stream(aux_path, sensor_names=[sensor_name])
    dst = Path(output_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_idx",
                "quat0",
                "quat1",
                "quat2",
                "quat3",
                "acc_x",
                "acc_y",
                "acc_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "mag_x",
                "mag_y",
                "mag_z",
            ]
        )
        for frame_index in range(parsed["quat"].shape[0]):
            quat = parsed["quat"][frame_index, 0]
            acc = parsed["acc"][frame_index, 0]
            gyro = parsed["gyro"][frame_index, 0]
            mag = parsed["mag"][frame_index, 0]
            writer.writerow(
                [
                    frame_index + 1,
                    f"{quat[0]:.6f}",
                    f"{quat[1]:.6f}",
                    f"{quat[2]:.6f}",
                    f"{quat[3]:.6f}",
                    f"{acc[0]:.6f}",
                    f"{acc[1]:.6f}",
                    f"{acc[2]:.6f}",
                    f"{gyro[0]:.6f}",
                    f"{gyro[1]:.6f}",
                    f"{gyro[2]:.6f}",
                    f"{mag[0]:.6f}",
                    f"{mag[1]:.6f}",
                    f"{mag[2]:.6f}",
                ]
            )


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


def parse_totalcapture_gt_positions(path: str | Path) -> dict[str, np.ndarray]:
    gt_path = Path(path)
    with gt_path.open(encoding="utf-8") as handle:
        header = [token for token in handle.readline().strip().split("\t") if token]
        columns = {name: [] for name in header}
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = [token for token in line.split("\t") if token]
            if len(values) != len(header):
                raise ValueError(f"Unexpected joint column count in {gt_path}")
            for name, token in zip(header, values):
                xyz = np.asarray([float(value) for value in token.split()], dtype=np.float64) * 0.0254
                columns[name].append(_GLOBAL_TO_MODEL_MATRIX @ xyz)
    return {name: np.stack(values, axis=0) for name, values in columns.items()}


def parse_totalcapture_gt_orientations(path: str | Path) -> dict[str, np.ndarray]:
    gt_path = Path(path)
    with gt_path.open(encoding="utf-8") as handle:
        header = [token for token in handle.readline().strip().split("\t") if token]
        columns = {name: [] for name in header}
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = [token for token in line.split("\t") if token]
            if len(values) != len(header):
                raise ValueError(f"Unexpected joint column count in {gt_path}")
            for name, token in zip(header, values):
                quat = np.asarray([float(value) for value in token.split()], dtype=np.float64)
                rot = quaternion_wxyz_to_matrix(quat)
                columns[name].append(matrix_to_quaternion_wxyz(_GLOBAL_TO_MODEL_MATRIX @ rot @ _GLOBAL_TO_MODEL_MATRIX))
    return {name: np.stack(values, axis=0) for name, values in columns.items()}


def parse_totalcapture_calibration_rotations(
    path: str | Path,
    sensor_names: list[str] | tuple[str, ...] = GLOBALPOSE_SENSOR_ORDER,
    calibration_kind: str = "bone",
) -> np.ndarray:
    calib_path = Path(path)
    sensor_names = list(sensor_names)
    output = np.zeros((len(sensor_names), 3, 3), dtype=np.float64)
    sensor_to_index = {name: index for index, name in enumerate(sensor_names)}
    with calib_path.open(encoding="utf-8") as handle:
        num_sensors = int(handle.readline().strip())
        for _ in range(num_sensors):
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected end of file in {calib_path}")
            parts = line.split()
            name = parts[0]
            if name not in sensor_to_index:
                continue
            quat = np.asarray([float(parts[4]), float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            output[sensor_to_index[name]] = quaternion_wxyz_to_matrix(quat).T
    if calibration_kind == "bone":
        output = output @ _BONE_TO_MODEL_MATRIX
    elif calibration_kind == "ref":
        output = output @ _GLOBAL_TO_MODEL_MATRIX
    else:
        raise ValueError(f"Unsupported calibration kind: {calibration_kind}")
    return output


def build_globalpose_sensor_trajectories(
    gt_positions: dict[str, np.ndarray],
    gt_orientations: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for sensor_name in GLOBALPOSE_SENSOR_ORDER:
        gt_name = _GLOBALPOSE_GT_MAPPING[sensor_name]
        trajectories[sensor_name] = {
            "positions": gt_positions[gt_name],
            "quaternions": gt_orientations[gt_name],
        }
    return trajectories


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), np.finfo(np.float64).eps)
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


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
