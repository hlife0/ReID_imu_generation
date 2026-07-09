# Synthetic IMU Evaluation Repository

This repository hosts two evaluation pipelines that judge synthetic IMU
generators by their usefulness on a real downstream task (IMU->motion
retrieval on TotalCapture, single sensor `R_LowArm`, 24-frame windows,
frozen subject split S1-S3 train / S4-S5 test), instead of signal
similarity alone.

Historical exploration (legacy generation workflows, signal-level metrics,
surveys) is preserved on the `archive/full-history-20260709` branch.

## IMU generation protocol

All generators obey one contract — standardized motion input (`MotionSequence`)
→ standardized 13-channel IMU output (`ImuSequence`), via a fixed CLI — so both
pipelines can rank any generator with no generator-specific code. Spec:
`docs/imu_generation_protocol.md`. Harness: `src/sim2real/gen_common.py::run_generator`.
Reference implementation (self-contained, dependency-free):
`scripts/sim2real/generators/naive/generate.py`. Conformance guard:
`tests/test_generation_protocol.py`.

## sim2real — probe-level benchmark

Self-contained contrastive probe (small two-tower encoder, trained from
scratch) ranking generators via TRTR / TSTR / mix protocols. Judgment =
IMU->motion full-gallery retrieval R@1 (1005 test windows, chance ~0.001).

- Code: `src/sim2real/`, `scripts/sim2real/` (01 corpus -> 01b estskel ->
  01c alignment -> 02 gate -> 06 matrix)
- Generators under test: `scripts/sim2real/generators/{naive,humogen,globalpose}`
  (run in the external `data_generation` venv; globalpose loads
  `third-party/GlobalPose`)
- Design: `docs/sim2real_design.md` · Findings: `docs/sim2real_findings_v1.md`,
  `docs/sim2real_findings_estskel.md` · Progress: `docs/sim2real_progress.md`
- Frozen split (do not edit): `configs/sim2real/splits/totalcapture_subject_v1.json`

## sim2pipe — main-project-level benchmark

Same corpus, split, and judgment, but the encoder is the main project's
real model (frozen MotionBERT + frozen DeSPITE + trained alignment head),
driven via a subprocess file contract — no main-project code is vendored.

- Code: `src/sim2pipe/`, `scripts/sim2pipe/` (01 export -> 02 gate ->
  03 pipe-probe matrix -> 04 full-gallery eval -> 05 report)
- Main-project patches (applied on demand, then reverted): `mainproj_patches/`
- Design: `docs/sim2pipe_design.md` · Findings: `docs/sim2pipe_findings_v1.md`
  · Progress: `docs/sim2pipe_progress.md` · Handoff: `docs/sim2pipe_handoff_g4.md`

## Headline results (lagfix baselines, 2026-07)

- sim2real (estskel): TRTR 0.0414; only the naive generator transfers
  (TSTR 0.021); real+naive mix +32% over TRTR (survives a same-size
  real+real control).
- sim2pipe (full-gallery judgment): TRTR 0.0872; all three generators
  collapse to the pairing-destruction floor (TSTR naive = 0.0000) — the
  frozen real-convention encoder exposes coordinate-frame/semantics gaps
  the probe adapts around. Fixing signal semantics (F2) is the
  prerequisite for any synthetic-data gain at pipeline level.

## Conventions

See `docs/repo_conventions.md`. Machine-specific paths live in
`configs/*/paths.yaml` (gitignored; see `paths.yaml.example`).
