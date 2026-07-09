# Synthetic IMU Evaluation

Two pipelines that judge synthetic IMU generators by how useful their output is
on a real downstream task — IMU→motion retrieval on TotalCapture (single sensor
`R_LowArm`, 24-frame windows, frozen subject split S1-S3 / S4 / S5) — rather than
by signal similarity alone.

## IMU generation protocol

Every generator obeys one contract: standardized motion input (`MotionSequence`)
→ standardized 13-channel IMU output (`ImuSequence`), via a fixed CLI, so both
pipelines rank any generator with no generator-specific code.

- Spec: `docs/imu_generation_protocol.md`
- Harness: `src/sim2real/gen_common.py::run_generator`
- Reference generator (self-contained, numpy-only): `scripts/sim2real/generators/naive/generate.py`
- Conformance test: `tests/test_generation_protocol.py`

## sim2real — probe-level benchmark

Self-contained contrastive probe (small two-tower encoder, trained from scratch)
ranking generators via TRTR / TSTR / mix protocols. Judgment = IMU→motion
full-gallery retrieval R@1.

- Code: `src/sim2real/`, `scripts/sim2real/`
  (01 corpus → 01b estskel → 01c alignment → 02 gate → 03 windows → 06 matrix → 07 report)
- Generators: `scripts/sim2real/generators/{naive,humogen,globalpose}`
- Frozen split (do not edit): `configs/sim2real/splits/totalcapture_subject_v1.json`

## sim2pipe — main-project-level benchmark

Same corpus, split, and judgment, but the encoder is the main project's real
model (frozen MotionBERT + DeSPITE + trained alignment head), driven via a
subprocess file contract — no main-project code is vendored.

- Code: `src/sim2pipe/`, `scripts/sim2pipe/`
  (01 export → 02 gate → 03 pipe-probe matrix → 04 full-gallery eval → 05 report)
- Main-project patches (applied on demand, then reverted): `mainproj_patches/`

## Running

Machine-specific paths live in `configs/*/paths.yaml` (gitignored; copy from
`paths.yaml.example`). Generators and training run in an external venv with
torch; `outputs/` and `data/` hold gitignored working artifacts.
