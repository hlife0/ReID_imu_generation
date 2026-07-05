# Synthetic IMU Research Repository

This repository is a lightweight, script-first workspace for synthetic IMU research from structured human motion data such as SMPL-X, mocap, Vicon, and equivalent body-motion representations.

The repository is intentionally organized to make the current generation pipeline transparent and inspectable rather than black-boxed.

## Current Implemented Workflow

The repository currently treats `GlobalPose_origin` as the main maintained `TotalCapture` workflow.

Implemented path:

1. stage a raw `TotalCapture` sample under `data/raw/totalcapture/S1_freestyle3/`
2. extract the matching raw `R_LowArm` IMU stream directly from the local `lxhong` TotalCapture raw files
3. synthesize a matching `R_LowArm` IMU sequence from `SMPL-X stageii` with the `GlobalPose_origin` pipeline
4. evaluate `raw vs generated` and save plots, metrics, and a report

Current entry points:

- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`
- `scripts/totalcapture_test/evaluate_imu_metrics.py`
- `scripts/totalcapture_test/plot_imu_comparison.py`

## Sim-to-Real Evaluation Subsystem (WIP)

`src/sim2real/` and `scripts/sim2real/` host a downstream benchmark that
ranks the generation pipelines by usefulness on a real task
(train-on-synthetic, test-on-real IMU->motion retrieval on TotalCapture)
instead of signal similarity alone.

- Design and construction plan: `docs/sim2real_design.md`
- Findings + handoff memo: `docs/sim2real_findings_v1.md`
- Progress / resume notes: `docs/sim2real_progress.md`
- Frozen data split (do not edit): `configs/sim2real/splits/totalcapture_subject_v1.json`

Status: benchmark v1 (`tc_rlowarm_w24_v1`) complete — corpus, gate, windows,
L1, and the full 46-cell L2 matrix. Headline: the naive kinematics baseline
transfers ~10x better than the full GlobalPose realism stack (TSTR), and
real+naive mixed training beats real-only by +37% relative; signal-frame
semantics dominate noise realism.

## Legacy Workflow

The previous staged three-file `S1_freestyle3` workflow is still preserved under `legacy/`:

- `src/legacy/totalcapture_test.py`
- `scripts/legacy/totalcapture_test/prepare_sample.py`
- `scripts/legacy/totalcapture_test/synthesize_imu.py`
- `scripts/legacy/totalcapture_test/plot_imu_comparison.py`
- `docs/legacy/totalcapture_test_workflow.md`

## Working Principles

- Keep core reusable logic in `src/`
- Run research workflows from `scripts/`
- Save inspectable intermediates under `data/interim`
- Save run artifacts under `outputs/`
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
│   ├── totalcapture_test/      # Current maintained workflow entry points
│   └── legacy/totalcapture_test/  # Archived legacy staged workflow
├── data/                       # Ignored runtime data and references
├── outputs/                    # Plots and run artifacts
├── tests/                      # Lightweight validation checks
├── docs/                       # Current workflow docs and background notes
└── third-party/                # External reference repositories
```

## Current Data Products

- `data/raw/totalcapture/S1_freestyle3/`
  - staged raw `TotalCapture` sample with video, raw IMU, calibration, and GT files
- `data/raw/totalcapture_test/GlobalPose_origin/s1_freestyle3/R_LowArm_raw.csv`
  - raw `R_LowArm` IMU extracted from the local `lxhong` TotalCapture source
- `outputs/totalcapture_test/GlobalPose_origin/`
  - `R_LowArm` raw/generated CSVs, comparison plot, metrics JSON, and report
- `data/legacy/processed/totalcapture_test/S1_freestyle3/`
  - archived legacy staged three-file sample
- `data/legacy/interim/totalcapture_test/S1_freestyle3/`
  - archived legacy synthetic IMU CSV
- `outputs/legacy/totalcapture_test/S1_freestyle3/`
  - archived legacy comparison outputs

## Current Primary Files

- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`
- `src/evaluation/imu_csv.py`
- `scripts/totalcapture_test/plot_imu_comparison.py`
- `docs/totalcapture_test_workflow.md`
- `docs/legacy/totalcapture_test_workflow.md`
- `docs/repo_conventions.md`
