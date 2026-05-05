# Legacy TotalCapture Test Workflow

This document describes the archived legacy workflow kept for reference.

## Scope

The legacy workflow implements a single worked example around one `TotalCapture` sequence:

- sequence: `S1_freestyle3`
- sensor: `R_LowArm`
- body representation: `SMPL-X`

## Inputs

The workflow uses three upstream sources:

1. video source root
   - current default: `/data/lxhong/totalcapture`
   - source file example: `freestyle3/TC_S1_freestyle3_cam1.mp4`

2. richer IMU source root
   - current default: `/data/lxhong`
   - source archive example: `s1_Gyro_Mag.tar.gz`
   - source member example: `s1/freestyle3_Xsens_AuxFields.sensors`

3. staged body-motion source root
   - current default: `/data/luoyizhang/HuMoGen/data/AMASS/TotalCapture`
   - source file example: `s1/freestyle3_stageii.npz`

## Step 1: Stage the three-file processed sample

Entry point:

- `scripts/legacy/totalcapture_test/prepare_sample.py`

Core logic:

- `src/legacy/totalcapture_test.py`

Outputs:

- `data/legacy/processed/totalcapture_test/S1_freestyle3/TC_S1_freestyle3_cam1.mp4`
- `data/legacy/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm.csv`
- `data/legacy/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_smplx.npz`

### Real IMU CSV format

The staged real IMU CSV contains one row per frame and one sensor only.

Columns:

- `frame_idx`
- `quat0`, `quat1`, `quat2`, `quat3`
- `acc_x`, `acc_y`, `acc_z`
- `gyro_x`, `gyro_y`, `gyro_z`
- `mag_x`, `mag_y`, `mag_z`

The current implementation expects the richer `Xsens_AuxFields.sensors` format and extracts only the chosen sensor.

## Step 2: Synthesize IMU from SMPL-X

Entry point:

- `scripts/legacy/totalcapture_test/synthesize_imu.py`

Current backend chain:

- `scripts/legacy/totalcapture_test/_synthesize_imu_existing.py`
- official `smplx` Python package for `SMPL-X` forward
- local `data_generation` utilities for joint projection and wrist sensor trajectory
- `third-party/GlobalPose/articulate/utils/imu/simulation.py` for IMU acceleration, angular velocity, and magnetic field

Output:

- `data/legacy/interim/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm_synthetic.csv`

### Synthetic IMU CSV format

The synthetic IMU CSV uses the same column layout as the staged real IMU CSV:

- `frame_idx`
- `quat0`, `quat1`, `quat2`, `quat3`
- `acc_x`, `acc_y`, `acc_z`
- `gyro_x`, `gyro_y`, `gyro_z`
- `mag_x`, `mag_y`, `mag_z`

This keeps real and synthetic outputs directly comparable.

The current synthetic path writes richer IMU fields, but the real and synthetic values are not expected to be on exactly the same physical calibration convention. The current workflow is intended for inspectable comparison, not final scientific validation.

## Step 3: Plot real vs synthetic comparison

Entry point:

- `scripts/legacy/totalcapture_test/plot_imu_comparison.py`

Helper:

- `scripts/totalcapture_test/_plot_imu_comparison.py`

Output:

- `outputs/legacy/totalcapture_test/S1_freestyle3/r_lowarm_real_vs_synthetic.png`

The figure overlays all quaternion, acceleration, gyroscope, and magnetic channels.

## Current limitation

This workflow is archived under `legacy` because the repository now uses the `GlobalPose_origin` workflow as the main maintained path. It remains available for historical comparison and reproducibility.
