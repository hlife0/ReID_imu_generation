# Synthetic IMU Research Repository

This repository is a lightweight, script-first workspace for synthetic IMU research from structured human motion data such as SMPL-X, mocap, Vicon, and equivalent body-motion representations.

The repository is intentionally organized to make the generation pipeline transparent and inspectable rather than black-boxed.

## Current Implemented Workflow

The repository currently contains one concrete, fully wired example workflow built around `TotalCapture`.

Implemented path:

1. stage one `TotalCapture` sample into `data/processed/totalcapture_test/...`
2. keep exactly three processed sample files:
   - one raw video `.mp4`
   - one real single-sensor IMU `.csv`
   - one `SMPL-X` body-motion `.npz`
3. synthesize a matching single-sensor IMU sequence from `SMPL-X`
4. compare real and synthetic IMU with an overlay plot

Current entry points:

- `scripts/totalcapture_test/prepare_sample.py`
- `scripts/totalcapture_test/synthesize_imu.py`
- `scripts/totalcapture_test/plot_imu_comparison.py`

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
├── src/                        # Current reusable project logic
├── scripts/
│   └── totalcapture_test/      # TotalCapture test workflow entry points
├── data/                       # Ignored runtime data and references
├── outputs/                    # Plots and run artifacts
├── tests/                      # Lightweight validation checks
├── docs/                       # Current workflow docs and background notes
└── third-party/                # External reference repositories
```

## Current Data Products

- `data/processed/totalcapture_test/S1_freestyle3/`
  - staged three-file sample
- `data/interim/totalcapture_test/S1_freestyle3/`
  - synthetic IMU derived from the staged `SMPL-X`
- `outputs/totalcapture_test/S1_freestyle3/`
  - real vs synthetic comparison plots

## Current Primary Files

- `src/totalcapture_test.py`
- `scripts/totalcapture_test/prepare_sample.py`
- `scripts/totalcapture_test/synthesize_imu.py`
- `scripts/totalcapture_test/plot_imu_comparison.py`
- `docs/totalcapture_test_workflow.md`
- `docs/repo_conventions.md`
