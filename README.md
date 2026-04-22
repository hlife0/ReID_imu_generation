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
├── src/                        # Reusable research logic
├── scripts/                    # Runnable entry points
│   └── totalcapture_test/      # Current concrete workflow
├── data/                       # Ignored runtime data and references
├── outputs/                    # Plots and run artifacts
├── tests/                      # Lightweight validation checks
├── docs/                       # Method notes and conventions
└── third-party/                # External reference repositories
```

## Suggested Workflow

1. Put motion or body data under `data/raw/`.
2. Place reference targets for evaluation under `data/reference/`.
3. Implement reusable synthesis and evaluation logic under `src/`.
4. Call that logic from `scripts/` for generation, inspection, and evaluation.
5. Save intermediate pipeline states under `data/interim`.
6. Save figures, metrics, and run-specific outputs under `outputs/`.

## Current Primary Files

- `src/totalcapture_test.py`
- `scripts/totalcapture_test/prepare_sample.py`
- `scripts/totalcapture_test/synthesize_imu.py`
- `scripts/totalcapture_test/plot_imu_comparison.py`
- `docs/repo_conventions.md`
