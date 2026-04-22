# Synthetic IMU Pipeline Overview

This repository is organized to make the synthetic IMU pipeline explicit and scientifically inspectable.

## Terminology convention

Within this repository, `mocap`, `Vicon`, and `IMU` are treated as already-stored raw-data formats or representations. `Processing techniques` refers only to operations that transform one stored representation into another or derive new outputs from it.

## Intended pipeline stages

1. Load and normalize motion or body data.
2. Define body frames, coordinate conventions, and sensor placement.
3. Derive kinematic quantities needed for IMU synthesis.
4. Generate idealized accelerometer and gyroscope signals.
5. Apply optional alignment, filtering, calibration, and noise models.
6. Compare generated signals against references and summarize metrics.

## Where to inspect intermediate results

- `data/interim/`: serialized intermediate states and cached pipeline outputs
- `outputs/`: plots, metrics, logs, and experiment summaries
- `scripts/inspect_case.py`: targeted single-case inspection entry point

## Initial design principle

Every transformation that materially changes the synthetic IMU signal should eventually be easy to inspect, save, disable, or swap for ablation.
