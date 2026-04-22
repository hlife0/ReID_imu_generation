# IMU Synthesis Methods and Reference Repositories

This note summarizes several open-source references that are directly useful for building a transparent synthetic IMU generation pipeline from structured human motion data.

## Main takeaway

There are relatively few public repositories that expose the synthetic IMU generation process clearly.

The most common open implementation pattern is:

1. recover body vertices and joint orientations from SMPL or AMASS
2. choose a fixed set of virtual sensor locations
3. use finite differences on sensor positions to synthesize acceleration
4. use body or joint rotations as synthetic orientation
5. optionally normalize the signals into a root-relative frame for downstream learning

Only a smaller subset of repositories explicitly model raw gyroscope, magnetic field, calibration error, or stochastic sensor noise.

Another important observation is that many later repositories are not fully independent implementations. Several of them inherit or lightly modify the same synthetic AMASS preprocessing pattern first popularized by DIP and then cleaned up by TransPose.

## Upstream ancestor: official AMASS repository

- Repository: https://github.com/nghorbani/amass
- Local clone: `third-party/amass`
- Official site: https://amass.is.tue.mpg.de/

### Why it feels like the ancestor

This intuition is basically right. AMASS is the upstream unification layer that made later synthetic IMU pipelines practical at scale. It standardizes large amounts of optical mocap data into a common body-model representation, which is exactly the kind of structured motion input later used for synthetic IMU generation.

### What it actually provides

- a unified archive of human motion represented with SMPL+H and related body-model conventions
- tools and tutorials for loading, preparing, and visualizing AMASS data
- compatibility notes for SMPL and SMPL-X style body models

### What it does not directly provide

- a full synthetic IMU generation pipeline
- virtual sensor placement logic
- accelerometer or gyroscope synthesis code
- IMU noise or calibration models

### Why I did not initially list it with the IMU repositories

Because I was grouping repositories by whether they directly implement the IMU synthesis step. In that narrower sense, AMASS is the upstream motion source rather than the downstream IMU generator.

But in the broader scientific genealogy, AMASS absolutely deserves to be called a key ancestor:

- AMASS makes large-scale synthetic IMU generation possible
- many later repositories synthesize IMU from AMASS sequences
- many later preprocessing conventions are organized around AMASS frame rates, body models, and sequence structure

## Repository 1: DIP / dip18

- Repository: https://github.com/eth-ait/dip18
- Local clone: `third-party/dip18`
- Key file: `third-party/dip18/data_synthesis/genSynData.py`

### What it does

This is a foundational and relatively minimal synthetic IMU pipeline. It generates synthetic IMU orientation and acceleration from SMPL motion sequences.

### Key implementation choices

- Fixed virtual sensor vertices are manually specified by `VERTEX_IDS`.
- Sensor orientations are taken from selected SMPL global rigid transforms.
- Sensor accelerations are computed by second-order finite differences on selected vertex trajectories.
- Motion is resampled to a target frame rate before synthesis.

### Why it matters

This implementation is easy to understand and is a good baseline for a transparent first version of a research repository.

### Main limitations

- very manual
- little modularization
- no explicit gyroscope synthesis
- no explicit sensor noise, bias drift, or calibration model
- limited handling of frame conventions and validation

## Repository 2: TransPose

- Repository: https://github.com/Xinyu-Yi/TransPose
- Local clone: `third-party/TransPose`
- Key files: `third-party/TransPose/preprocess.py`, `third-party/TransPose/utils.py`

### What it does

TransPose includes a practical preprocessing pipeline that synthesizes IMU-like measurements from AMASS and prepares them for pose estimation.

### Key implementation choices

- Uses six fixed mesh vertices as virtual sensor locations.
- Uses six selected joints as sensor orientations.
- Aligns AMASS global coordinates to the target training convention.
- Computes acceleration from vertex trajectories with finite differences.
- Applies temporal smoothing in the acceleration synthesis step.
- Converts the resulting measurements into a root-relative representation before model input.

### Why it matters

Compared with DIP, this code is cleaner and closer to a reusable preprocessing pipeline. It is a good reference for how to separate:

- raw synthetic measurements in global frame
- normalized model inputs
- dataset preprocessing

### Main limitations

- still mainly oriented toward network training rather than signal-faithful IMU simulation
- acceleration and orientation are emphasized more than full raw sensor modeling
- noise and calibration effects are limited

## Repository 3: PIP

- Repository: https://github.com/Xinyu-Yi/PIP
- Local clone: `third-party/PIP`
- Key files: `third-party/PIP/preprocess.py`, `third-party/PIP/readme.md`

### What it does

PIP largely follows the TransPose-style preprocessing route, but its TotalCapture processing is more careful and is especially useful for evaluation-oriented research.

### Key implementation choices

- keeps the same AMASS to synthetic IMU pattern using selected vertices and selected joint orientations
- preprocesses raw DIP-IMU and TotalCapture for evaluation
- removes mean acceleration bias on TotalCapture by comparing real IMU acceleration with vertex-based synthetic acceleration

### Why it matters

If your repository wants to evaluate synthetic IMU quality against real measurements, this acceleration bias correction idea is worth studying even if you do not adopt the rest of the PIP stack.

## Repository 4: GlobalPose

- Repository: https://github.com/Xinyu-Yi/GlobalPose
- Local clone: `third-party/GlobalPose`
- Key files:
  - `third-party/GlobalPose/imu_synthesis.py`
  - `third-party/GlobalPose/articulate/utils/imu/simulation.py`
  - `third-party/GlobalPose/process.py`

### What it does

GlobalPose contains the most explicit open implementation of raw IMU synthesis among the cloned references. It synthesizes accelerometer, gyroscope, and magnetic-field measurements from SMPL trajectories, then injects calibration error and stochastic noise.

### Key implementation choices

- Uses a dedicated `IMUSimulator` with a 6DoF sensor trajectory interface.
- Computes acceleration in the sensor frame from world-frame second derivatives minus gravity.
- Computes angular velocity from time derivatives of rotation matrices.
- Computes magnetic field from a fixed world magnetic vector.
- Adds random-walk style perturbations to virtual sensor position and orientation.
- Adds Gaussian sensor noise and random walk noise to raw signals.
- Supports optional ESKF-based orientation estimation from raw signals.
- Simulates a calibration stage before returning the final measurements.

### Why it matters

This repository is highly relevant if the goal is not only to generate signals, but also to study assumptions, intermediate steps, and how noise and calibration design change the result.

### Main limitations

- heavier dependency stack
- some components are tied to the authors' custom tooling
- codebase is larger and less lightweight than a minimal research repo should be

## Repository 5: PNP

- Repository: https://github.com/Xinyu-Yi/PNP
- Local clone: `third-party/PNP`
- Key files: `third-party/PNP/process.py`, `third-party/PNP/readme.md`

### What it does

PNP is more useful as a data formatting and evaluation reference than as a pure synthetic IMU generator. It parses real TotalCapture IMU, calibration files, and SMPL-aligned pose references into a structured representation.

### Key implementation choices

- explicitly stores calibration transforms such as `RIM`, `RSB`, and `RIS`
- reads real TotalCapture orientation, acceleration, gyroscope, and magnetic field
- aligns the official TotalCapture frames and calibration files to SMPL conventions
- generates a structured representation that keeps raw measurements and reference pose aligned

### Why it matters

For your repository, PNP is relevant because it shows how to keep evaluation data transparent instead of burying calibration inside opaque preprocessing.

## Repository 6: TIP

- Repository: https://github.com/jyf588/transformer-inertial-poser
- Local clone: `third-party/TIP`
- Key files:
  - `third-party/TIP/data-gen-and-viz-bullet-new.py`
  - `third-party/TIP/preprocess_DIP_TC_new.py`
  - `third-party/TIP/README.md`

### What it does

TIP takes a noticeably different route from the vertex-difference family. It uses URDF and PyBullet-based forward kinematics to synthesize IMU readings and also synthesizes extra physical constraints used by the method.

### Key implementation choices

- sensor placement is specified in a URDF rather than only by SMPL vertex indices
- synthetic IMU readings are generated from simulated joint trajectories in PyBullet
- uses finite differences on joint positions from the simulator to get acceleration
- mixes synthesized AMASS data with real DIP and TotalCapture data
- relies on preprocessed TotalCapture signals distributed in DIP format rather than parsing the official dataset alone

### Why it matters

This is a useful reference if you want a sensor-placement abstraction that is more interpretable than hard-coded vertex IDs, or if you want to explore physics-inspired alternatives to mesh-vertex acceleration synthesis.

## Repository 7: IMUPoser

- Repository: https://github.com/FIGLAB/IMUPoser
- Local clone: `third-party/IMUPoser`
- Key files:
  - `third-party/IMUPoser/scripts/1. Preprocessing/1. preprocess_all.py`
  - `third-party/IMUPoser/README.md`

### What it does

IMUPoser keeps the general AMASS-to-synthetic-IMU pipeline but adapts it for consumer-device placements such as phones, watches, and earbuds.

### Key implementation choices

- still synthesizes acceleration from mesh-vertex trajectories
- changes virtual sensor locations and corresponding joint selections
- stores both real measured signals and synthetic vertex-based counterparts for DIP

### Why it matters

This repository shows that the same transparent pipeline can support different sensor layouts without changing its core logic.

## Repository 8: MobilePoser

- Repository: https://github.com/SPICExLAB/MobilePoser
- Local clone: `third-party/MobilePoser`
- Key files:
  - `third-party/MobilePoser/mobileposer/process.py`
  - `third-party/MobilePoser/README.md`

### What it does

MobilePoser is a later-generation preprocessing codebase that extends the IMUPoser direction. It supports a mobile-device-oriented sensor layout, uses 30 FPS, adds contact labels, and includes explicit TotalCapture processing.

### Key implementation choices

- downsamples to 30 FPS
- synthesizes acceleration with the updated target frame rate
- computes foot-contact labels from joint motion
- handles TotalCapture using AMASS TotalCapture poses plus real TotalCapture IMU
- applies acceleration bias correction against vertex-based synthetic acceleration

### Why it matters

This is a strong example of how to keep a script-first research pipeline while evolving sensor layout, dataset mix, and intermediate supervision targets.

## Repository 9: WheelPoser

- Repository: https://github.com/axle-lab/WheelPoser
- Local clone: `third-party/WheelPoser`
- Key files:
  - `third-party/WheelPoser/scripts/1. Preprocessing/1.1 preprocess_all.py`
  - `third-party/WheelPoser/README.md`

### What it does

WheelPoser adapts the same synthetic IMU preprocessing template to a four-sensor setting and to wheelchair-user data.

### Key implementation choices

- changes the sensor layout to four sensors
- filters implausible AMASS motions before synthesis
- reuses the same core steps: AMASS alignment, forward kinematics, selected virtual sensors, finite-difference acceleration

### Why it matters

This repo is useful evidence that the core synthetic IMU pipeline should be designed around configurable sensor layout rather than a fixed six-sensor assumption.

## Datasets and non-repo resources worth separating from code repositories

### TotalCapture official dataset

- Official page: https://cvssp.org/data/totalcapture/

This is not a synthetic IMU repository, but it is one of the most important evaluation references. The official dataset provides synchronized video, IMU, and Vicon labels. Multiple repositories above parse its raw IMU data, calibration files, and Vicon ground truth in different ways.

### DIP-IMU official dataset

- Official page: https://dip.is.tue.mpg.de/

DIP-IMU is critical because it provides real sparse IMU measurements together with reference poses and was explicitly introduced to support learning with both real and synthesized IMU data.

### AMASS official dataset

- Official page: https://amass.is.tue.mpg.de/

AMASS is not an IMU dataset, but it is the dominant source used for synthetic IMU generation from SMPL or SMPL-H motion. Many repositories synthesize IMU from AMASS and then evaluate on DIP-IMU or TotalCapture.

## Important lineage observation

Several repositories are best understood as variations of the same preprocessing family:

- `dip18` introduces the simple vertex-difference baseline
- `TransPose` cleans it into a more reusable preprocessing pipeline
- `PIP` adds stronger TotalCapture preprocessing and bias correction
- `IMUPoser`, `MobilePoser`, and `WheelPoser` adapt the same idea to new sensor layouts and tasks

The repositories that stand out as genuinely different in synthesis philosophy are:

- `GlobalPose`, because it explicitly simulates raw accelerometer, gyroscope, magnetic field, calibration error, and stochastic noise
- `TIP`, because it moves sensor placement into URDF plus PyBullet rather than staying purely in a mesh-vertex viewpoint

## Recommended design implications for this repository

Based on the references above, a transparent synthetic IMU repository should keep the following stages explicit and separately inspectable:

1. motion loading and resampling
2. body-model forward kinematics
3. virtual sensor placement
4. sensor trajectory generation
5. raw acceleration and angular velocity synthesis
6. gravity handling and coordinate transforms
7. optional filtering, calibration, and noise injection
8. reference alignment and evaluation
9. saving intermediate results for inspection

It should also make the sensor layout an explicit research object. Hard-coding vertex IDs is acceptable for a baseline, but later work strongly suggests keeping sensor placement configurable.

## Suggested baseline strategy

For a first implementation, the best research tradeoff is:

- start with a DIP or TransPose style deterministic baseline
- keep every intermediate tensor savable
- then incrementally add GlobalPose-style noise, calibration, and gyroscope synthesis as optional modules

This staged approach keeps the pipeline understandable while still allowing later realism improvements.

## Additional adjacent repository

### DynaIP

- Repository: https://github.com/dx118/dynaip

DynaIP is not primarily a synthetic AMASS-to-IMU codebase in the same style as the repositories above. It is still useful because it processes multiple public Xsens motion datasets and DIP-IMU into a unified training pipeline. This makes it more relevant for cross-dataset evaluation and robustness benchmarking than for the initial synthesis implementation itself.
