# Synthetic IMU Research Repository

This repository is a lightweight, script-first workspace for synthetic IMU research from structured human motion data such as SMPL-X, mocap, or equivalent body-motion representations.

The repository is intentionally organized to make the generation pipeline transparent and inspectable rather than black-boxed. The goal is to support:

- synthetic IMU generation from motion or body data
- inspection of intermediate computations and assumptions
- analysis and evaluation against reference targets
- method improvement through alignment, filtering, calibration, and noise design
- ablations, comparisons, plots, and metrics

## Working Principles

- Keep core reusable logic in `src/`
- Run research workflows from `scripts/`
- Save inspectable intermediates under `data/interim`
- Save run artifacts under `outputs/`
- Keep notebooks secondary to scripts
- Prefer small, explicit modules over deep abstraction

## Terminology Boundary

This repository treats already-collected `mocap`, `Vicon`, and `IMU` inputs as data formats or raw-data representations, not as a discussion about acquisition hardware or sensing systems.

In this repository:

- `mocap`, `Vicon`, and `IMU` refer to stored raw data formats or data sources that already exist on disk.
- `SMPL`, `SMPL-H`, and `SMPL-X` refer to body-motion representations or processed output formats.
- `processing techniques` means operations that transform one stored data format into another representation or generate derived outputs from it.

Examples of `processing techniques` in this repository:

- mocap or Vicon text files -> SMPL-family parameters
- raw IMU streams -> filtered or aligned IMU
- body-motion representations -> synthetic IMU
- one raw format -> another processed format used for analysis or evaluation

See `docs/repo_conventions.md` for the explicit repository convention.

## Repository Layout

```text
repo/
├── src/                # Reusable research logic
├── scripts/            # Runnable entry points
├── experiments/        # Experiment-specific notes or configs
├── data/
│   ├── raw/            # Original motion/body inputs
│   ├── reference/      # Real IMU or other evaluation references
│   ├── interim/        # Inspectable intermediate results
│   └── processed/      # Reusable processed outputs
├── outputs/            # Plots, metrics, tables, logs, run artifacts
├── tests/              # Lightweight validation checks
├── docs/               # Method notes and conventions
└── notebooks/          # Optional exploratory analysis
```

## Suggested Workflow

1. Put motion or body data under `data/raw/`.
2. Place reference targets for evaluation under `data/reference/`.
3. Implement reusable synthesis and evaluation logic under `src/`.
4. Call that logic from `scripts/` for generation, inspection, and evaluation.
5. Save intermediate pipeline states under `data/interim`.
6. Save figures, metrics, and run-specific outputs under `outputs/`.

## First Files To Extend

- `src/motion_io.py`
- `src/frames.py`
- `src/alignment.py`
- `src/metrics.py`
- `src/totalcapture_test.py`
- `scripts/generate_imu.py`
- `scripts/inspect_case.py`
- `scripts/evaluate_signals.py`
- `scripts/totalcapture_test/prepare_sample.py`
