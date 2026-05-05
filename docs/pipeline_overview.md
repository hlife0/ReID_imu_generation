# Synthetic IMU Pipeline Overview

This repository is organized to make the synthetic IMU pipeline explicit and scientifically inspectable. At the moment, the repository contains one maintained `GlobalPose_origin`-based `TotalCapture` workflow plus an archived legacy staged workflow.

## Terminology convention

Within this repository, `mocap`, `Vicon`, and `IMU` are treated as already-stored raw-data formats or representations. `Processing techniques` refers only to operations that transform one stored representation into another or derive new outputs from it.

## Current Implemented Pipeline Stages

1. read the copied official `TotalCapture` reference sample
2. generate a synthetic `R_LowArm` IMU stream with the maintained `GlobalPose_origin` workflow
3. compare real and synthetic IMU with plots and summary metrics

See `docs/totalcapture_test_workflow.md` for the current maintained path.

The previous staged `S1_freestyle3` sample workflow is archived under `docs/legacy/totalcapture_test_workflow.md`.

## Where to inspect intermediate results

- `data/interim/`: serialized intermediate states and cached pipeline outputs
- `outputs/`: plots and run-specific summaries
- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`: current main workflow entry point
- `scripts/totalcapture_test/evaluate_imu_metrics.py`: current metrics entry point
- `scripts/legacy/totalcapture_test/`: archived legacy pipeline wrappers

## Design Principle

Every transformation that materially changes the synthetic IMU signal should eventually be easy to inspect, save, disable, or swap for ablation.
