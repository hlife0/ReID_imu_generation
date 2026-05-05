from __future__ import annotations

import csv
import shutil
import tarfile
from pathlib import Path

import numpy as np


def _sequence_parts(sequence_id: str) -> tuple[str, str, str]:
    subject, session = sequence_id.split("_", 1)
    return subject, subject.lower(), session


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _session_to_aux_name(session: str) -> str:
    if session.startswith("acting"):
        return session.capitalize()
    return session


def _extract_member(archive_path: Path, member_name: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Could not find {member_name} in {archive_path}")
        with dst.open("wb") as f:
            shutil.copyfileobj(extracted, f)


def _stage_video(
    video_source_root: Path,
    subject_upper: str,
    subject_lower: str,
    session: str,
    camera_name: str,
    dst: Path,
) -> None:
    direct = video_source_root / session / f"TC_{subject_upper}_{session}_{camera_name}.mp4"
    archive = video_source_root / f"{subject_lower}_{session}.tar.gz"
    member = f"{session}/TC_{subject_upper}_{session}_{camera_name}.mp4"
    if direct.exists():
        _copy_file(direct, dst)
        return
    if archive.exists():
        _extract_member(archive, member, dst)
        return
    raise FileNotFoundError(f"Could not locate video for {subject_upper}_{session} in {video_source_root}")


def _iter_sensor_lines_from_path(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield line


def _iter_sensor_lines_from_archive(archive_path: Path, member_name: str):
    with tarfile.open(archive_path, "r:gz") as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Could not find {member_name} in {archive_path}")
        for raw_line in extracted:
            yield raw_line.decode("utf-8")


def _write_richer_single_sensor_csv(lines, dst: Path, sensor_name: str) -> None:
    iterator = iter(lines)
    header = next(iterator).strip().split()
    if len(header) < 2:
        raise ValueError("Invalid Xsens header")
    num_sensors = int(header[0])
    num_frames = int(header[1])
    rows: list[list[str]] = []

    for _ in range(num_frames):
        frame_line = next(iterator, None)
        if frame_line is None:
            break
        frame_idx = int(frame_line.strip())
        sensor_lines = [next(iterator).rstrip("\n") for _ in range(num_sensors)]
        kept_line = None
        for line in sensor_lines:
            if line.split("\t")[0].split()[0] == sensor_name:
                kept_line = line
                break
        if kept_line is None:
            raise ValueError(f"Sensor {sensor_name} not found in frame")
        toks = kept_line.split("\t")
        if len(toks) != 14:
            raise ValueError(f"Expected richer AuxFields format with 14 columns, got {len(toks)}")
        rows.append(
            [
                str(frame_idx),
                *[f"{float(value):.6f}" for value in toks[1:5]],
                *[f"{float(value):.6f}" for value in toks[5:8]],
                *[f"{float(value):.6f}" for value in toks[8:11]],
                *[f"{float(value):.6f}" for value in toks[11:14]],
            ]
        )

    if len(rows) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, got {len(rows)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
        writer.writerows(rows)


def _stage_richer_imu(
    imu_source_root: Path,
    subject_lower: str,
    session: str,
    sensor_name: str,
    dst: Path,
) -> None:
    aux_name = _session_to_aux_name(session)
    direct = imu_source_root / subject_lower / f"{aux_name}_Xsens_AuxFields.sensors"
    archive = imu_source_root / f"{subject_lower}_Gyro_Mag.tar.gz"
    member = f"{subject_lower}/{aux_name}_Xsens_AuxFields.sensors"
    if direct.exists():
        _write_richer_single_sensor_csv(_iter_sensor_lines_from_path(direct), dst, sensor_name=sensor_name)
        return
    if archive.exists():
        _write_richer_single_sensor_csv(_iter_sensor_lines_from_archive(archive, member), dst, sensor_name=sensor_name)
        return
    raise FileNotFoundError(f"Could not locate richer IMU for {subject_lower}_{session} in {imu_source_root}")


def _write_smplx_npz(src: Path, dst: Path, sequence_id: str, subject: str, session: str) -> None:
    stageii = np.load(src, allow_pickle=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst,
        sequence_id=np.array(sequence_id),
        subject=np.array(subject),
        session=np.array(session),
        gender=stageii["gender"],
        surface_model_type=np.array("smplx"),
        mocap_frame_rate=stageii["mocap_frame_rate"],
        mocap_time_length=stageii["mocap_time_length"],
        trans=stageii["trans"],
        betas=stageii["betas"],
        num_betas=stageii["num_betas"],
        root_orient=stageii["root_orient"],
        pose_body=stageii["pose_body"],
        pose_hand=stageii["pose_hand"],
        pose_jaw=stageii["pose_jaw"],
        pose_eye=stageii["pose_eye"],
        poses=stageii["poses"],
        source_stageii_file=np.array(str(src)),
    )


def stage_totalcapture_test(
    video_source_root: Path,
    imu_source_root: Path,
    stageii_totalcapture_root: Path,
    data_root: Path,
    sequence_id: str,
    sensor_name: str = "R_LowArm",
    camera_name: str = "cam1",
) -> dict[str, str]:
    subject_upper, subject_lower, session = _sequence_parts(sequence_id)
    output_dir = data_root / "processed" / "totalcapture_test" / sequence_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.iterdir():
        if old_path.is_file() or old_path.is_symlink():
            old_path.unlink()
        elif old_path.is_dir():
            shutil.rmtree(old_path)

    stageii_src = stageii_totalcapture_root / subject_lower / f"{session}_stageii.npz"

    video_dst = output_dir / f"TC_{subject_upper}_{session}_{camera_name}.mp4"
    imu_dst = output_dir / f"{subject_lower}_{session}_{sensor_name}.csv"
    smplx_dst = output_dir / f"{subject_lower}_{session}_smplx.npz"

    _stage_video(
        video_source_root=video_source_root,
        subject_upper=subject_upper,
        subject_lower=subject_lower,
        session=session,
        camera_name=camera_name,
        dst=video_dst,
    )
    _stage_richer_imu(
        imu_source_root=imu_source_root,
        subject_lower=subject_lower,
        session=session,
        sensor_name=sensor_name,
        dst=imu_dst,
    )
    _write_smplx_npz(stageii_src, smplx_dst, sequence_id=sequence_id, subject=subject_upper, session=session)

    expected_files = {video_dst.name, imu_dst.name, smplx_dst.name}
    actual_files = {p.name for p in output_dir.iterdir()}
    unexpected = sorted(actual_files - expected_files)
    if unexpected:
        raise RuntimeError(f"Unexpected extra files in {output_dir}: {unexpected}")

    return {
        "sequence_id": sequence_id,
        "output_dir": str(output_dir),
        "video_mp4": str(video_dst),
        "imu_sensor_stream": str(imu_dst),
        "smplx_npz": str(smplx_dst),
        "sensor_name": sensor_name,
    }
