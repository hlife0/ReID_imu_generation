from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np


def _sequence_parts(sequence_id: str) -> tuple[str, str, str]:
    subject, session = sequence_id.split("_", 1)
    return subject, subject.lower(), session


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _filter_single_sensor_stream_to_csv(src: Path, dst: Path, sensor_name: str) -> None:
    with src.open(encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) < 2:
            raise ValueError(f"Invalid Xsens header in {src}")
        num_sensors = int(header[0])
        num_frames = int(header[1])
        rows: list[list[str]] = []

        for _ in range(num_frames):
            frame_line = f.readline()
            if not frame_line:
                break
            frame_idx = int(frame_line.strip())
            sensor_lines = [f.readline().rstrip("\n") for _ in range(num_sensors)]
            kept_line = None
            for line in sensor_lines:
                if line.split("\t")[0].split()[0] == sensor_name:
                    kept_line = line
                    break
            if kept_line is None:
                raise ValueError(f"Sensor {sensor_name} not found in frame of {src}")
            toks = kept_line.split("\t")
            rows.append(
                [
                    str(frame_idx),
                    *[f"{float(value):.6f}" for value in toks[1:5]],
                    *[f"{float(value):.6f}" for value in toks[5:8]],
                ]
            )

    if len(rows) != num_frames:
        raise ValueError(f"Expected {num_frames} frames in {src}, got {len(rows)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "quat0", "quat1", "quat2", "quat3", "acc_x", "acc_y", "acc_z"])
        writer.writerows(rows)


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
    raw_totalcapture_root: Path,
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

    video_src = raw_totalcapture_root / session / f"TC_{subject_upper}_{session}_{camera_name}.mp4"
    imu_src = raw_totalcapture_root / subject_lower / f"{subject_lower}_{session}_Xsens.sensors"
    stageii_src = stageii_totalcapture_root / subject_lower / f"{session}_stageii.npz"

    video_dst = output_dir / video_src.name
    imu_dst = output_dir / f"{subject_lower}_{session}_{sensor_name}.csv"
    smplx_dst = output_dir / f"{subject_lower}_{session}_smplx.npz"

    _copy_file(video_src, video_dst)
    _filter_single_sensor_stream_to_csv(imu_src, imu_dst, sensor_name=sensor_name)
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
