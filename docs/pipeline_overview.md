# Synthetic IMU Pipeline Overview

This repository is organized to make the synthetic IMU pipeline explicit and scientifically inspectable. At the moment, the repository contains one concrete `TotalCapture`-based workflow plus background research notes.

## Terminology convention

Within this repository, `mocap`, `Vicon`, and `IMU` are treated as already-stored raw-data formats or representations. `Processing techniques` refers only to operations that transform one stored representation into another or derive new outputs from it.

## Current Implemented Pipeline Stages

1. stage one processed `TotalCapture` sample from richer raw IMU, video, and `SMPL-X`
2. synthesize a single-sensor IMU sequence from the staged `SMPL-X`
3. compare real and synthetic IMU with overlay plots

See `docs/totalcapture_test_workflow.md` for the file-level workflow.

## Where to inspect intermediate results

- `data/interim/`: serialized intermediate states and cached pipeline outputs
- `outputs/`: plots, metrics, logs, and experiment summaries
- `scripts/totalcapture_test/plot_imu_comparison.py`: current comparison entry point
- `scripts/totalcapture_test/synthesize_imu.py`: current synthesis entry point

## Design Principle

Every transformation that materially changes the synthetic IMU signal should eventually be easy to inspect, save, disable, or swap for ablation.
