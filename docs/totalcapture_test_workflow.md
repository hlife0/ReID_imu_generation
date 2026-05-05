# TotalCapture Workflow

This repository now treats `GlobalPose_origin` as the main maintained `TotalCapture` workflow.

Detailed walkthrough:

- `docs/totalcapture_test_workflow_detailed.md`

## Current Main Path

Entry point:

- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`

Inputs:

- staged raw sample under `data/raw/totalcapture/S1_freestyle3/`
- raw IMU source from the local `lxhong` TotalCapture dataset
- `SMPL-X stageii` source from `/data/luoyizhang/HuMoGen/data/AMASS/TotalCapture/`

Outputs:

- `data/raw/totalcapture/S1_freestyle3/`
- `data/raw/totalcapture_test/GlobalPose_origin/s1_freestyle3/R_LowArm_raw.csv`
- `outputs/totalcapture_test/GlobalPose_origin/csv/R_LowArm_generated.csv`
- `outputs/totalcapture_test/GlobalPose_origin/metrics/R_LowArm_raw_vs_generated_metrics.json`
- `outputs/totalcapture_test/GlobalPose_origin/plots/R_LowArm_raw_vs_generated.png`
- `outputs/totalcapture_test/GlobalPose_origin/report.md`

The maintained workflow compares `raw` and `generated` IMU only for `R_LowArm`.

## Legacy Path

The previous staged three-file `S1_freestyle3` pipeline is still available under `legacy/`:

- `scripts/legacy/totalcapture_test/prepare_sample.py`
- `scripts/legacy/totalcapture_test/synthesize_imu.py`
- `scripts/legacy/totalcapture_test/plot_imu_comparison.py`
- `src/legacy/totalcapture_test.py`
- `docs/legacy/totalcapture_test_workflow.md`

Its default data/output roots also live under `data/legacy/` and `outputs/legacy/`.
