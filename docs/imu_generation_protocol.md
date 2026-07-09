# IMU Generation Protocol

Every synthetic-IMU generator in this repo — and every one added in the future —
conforms to a single contract: **one standardized motion input → one standardized
IMU output**, exchanged as files so generators can run in whatever environment
their physics needs (external venv, torch, an external research checkout) while
the evaluation pipelines stay agnostic.

This is the contract the two evaluation pipelines (`sim2real`, `sim2pipe`)
consume. If a generator conforms here, both pipelines can rank it with no
generator-specific code.

## Standardized input — `MotionSequence`

A generator reads exactly one motion file (`motion.npz`), loaded via
`src/sim2real/contracts.py::MotionSequence`:

| field         | type            | meaning                                            |
|---------------|-----------------|----------------------------------------------------|
| `joints`      | `(T, J, 3)` f32 | world-space joint positions over time              |
| `fps`         | float           | sample rate (TotalCapture corpus: 60.0)            |
| `joint_layout`| str             | joint-name convention (`data_generation_pipeline_v1`) |

The generator does **not** consume raw SMPL-X, video, or dataset-specific formats.
Producing `motion.npz` from a source dataset is a corpus-build concern
(`scripts/sim2real/01_build_corpus.py` + `_smplx_to_motion.py`), upstream of and
separate from the generator.

The sensor 6DoF trajectory (world position + orientation of the mounted sensor)
is derived from `joints` by `src/sim2real/geom.py::compute_sensor_trajectory`,
shared by all generators. A generator receives `(motion, positions (T,3),
quat_wxyz (T,4))` and never re-derives placement itself.

## Standardized output — `ImuSequence` (13 channels)

A generator writes exactly one IMU stream (`ImuSequence`), a fixed 13-channel
layout (`contracts.IMU_CHANNELS_13`):

```
quat0 quat1 quat2 quat3   (w, x, y, z, sensor-to-world)
acc_x acc_y acc_z         (m/s^2, sensor frame)
gyro_x gyro_y gyro_z      (rad/s, sensor frame)
mag_x mag_y mag_z         (magnetometer; zeros if the generator has no model)
```

plus header metadata: `fps`, `source` (`synth/<generator>/<config_hash>`),
`sensor` (e.g. `R_LowArm`), and a free `meta` dict. Files are named by stream
identity: `synth_<generator>_<cfg8>.npz` + a `.manifest.json` provenance record
(resolved config, config hash, input file hashes, seed, git sha).

Which downstream channels are actually used is the evaluation's choice — the
current benchmarks feed only the 6 acc+gyro channels to the probe — but a
generator always emits the full 13.

## CLI contract

Each generator is `scripts/sim2real/generators/<name>/generate.py` with a fixed
interface (dispatched by directory name in `01_build_corpus.py`):

```
generate.py --motion <motion.npz> --sensor R_LowArm \
            --config <config.json> --seed <N> --out <dir>
```

- `--config` is JSON carrying `"generator": "<name>"` (checked) plus generator-
  specific parameters. Config identity (its 8-hex `config_hash`) names the output
  stream, so any parameter change produces a distinct, traceable artifact.
- On success the generator prints one JSON receipt line to stdout
  (`{npz, manifest, source, config_hash, frames}`).

## Adding a generator

The harness `src/sim2real/gen_common.py::run_generator` implements the entire
contract — argument parsing, config load + name check, motion+trajectory load,
`ImuSequence`/manifest write, and the stdout receipt. A new generator is just its
synthesis core plus one call:

```python
from sim2real.gen_common import run_generator

def synthesize(motion, positions, quat_wxyz, config, seed):
    # ... produce arrays ...
    return {"quat": quat, "acc": acc, "gyro": gyro, "mag": mag,
            "fps": float(...), "extra_meta": {...}}

if __name__ == "__main__":
    run_generator("my_generator", synthesize)
```

Going through `run_generator` means conformance is structural: the output is a
valid `ImuSequence` by construction. `tests/test_generation_protocol.py` guards
this against the naive reference.

## The reference: `naive`

`scripts/sim2real/generators/naive/generate.py` is the canonical conforming
generator and the intended starting point for reading or writing one. It is
**fully self-contained** — pure finite-difference kinematics on `numpy` + repo
code (`geom`, `globalpose_origin_adapter`), no external checkout — so it is the
readable, dependency-free demonstration of the protocol:

- double-difference sensor world position → specific force (minus gravity) →
  rotate into the sensor frame = accel;
- finite-difference sensor orientation = gyro;
- zero magnetometer; no realism modules; deterministic (seed-independent).

The other two generators wrap heavier physics but obey the same I/O contract:
`globalpose` (torch + in-repo `third-party/GlobalPose`, realism switches) and
`humogen` (an external HuMoGen synthesis module, path from config).
