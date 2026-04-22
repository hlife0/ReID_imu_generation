# Third-Party References

This directory is reserved for local clones of external repositories that are useful for studying synthetic IMU generation and related evaluation pipelines.

These repositories are ignored by Git by default so that reference code does not get mixed into the main research history of this repository.

## Cloned repositories

- `amass/`
  - Repository: https://github.com/nghorbani/amass
  - Why it is useful: the upstream official AMASS repository. It is the canonical source for understanding how unified motion data are represented and loaded before later projects synthesize IMU from them.
  - Key files: `README.md`, `src/amass/data/prepare_data.py`, `notebooks/`

- `dip18/`
  - Repository: https://github.com/eth-ait/dip18
  - Why it is useful: foundational open-source synthetic IMU generation code used by DIP. Simple and direct.
  - Key file: `data_synthesis/genSynData.py`

- `TransPose/`
  - Repository: https://github.com/Xinyu-Yi/TransPose
  - Why it is useful: practical AMASS-to-IMU preprocessing pipeline with clearer organization than DIP.
  - Key files: `preprocess.py`, `utils.py`

- `PIP/`
  - Repository: https://github.com/Xinyu-Yi/PIP
  - Why it is useful: extends the TransPose-style preprocessing line and includes a more careful TotalCapture preprocessing path with acceleration bias removal.
  - Key files: `preprocess.py`, `readme.md`

- `PNP/`
  - Repository: https://github.com/Xinyu-Yi/PNP
  - Why it is useful: less about synthetic AMASS IMU itself, more about a well-structured representation of real IMU measurements, calibration matrices, and TotalCapture parsing.
  - Key files: `process.py`, `readme.md`

- `GlobalPose/`
  - Repository: https://github.com/Xinyu-Yi/GlobalPose
  - Why it is useful: the most explicit raw IMU synthesis implementation among the cloned repositories, including acceleration, gyroscope, magnetic field, calibration error, random walk noise, and optional ESKF.
  - Key files: `imu_synthesis.py`, `articulate/utils/imu/simulation.py`, `process.py`

- `TIP/`
  - Repository: https://github.com/jyf588/transformer-inertial-poser
  - Why it is useful: provides a different synthesis philosophy based on URDF plus PyBullet forward kinematics, and mixes synthetic AMASS data with real DIP/TotalCapture data.
  - Key files: `data-gen-and-viz-bullet-new.py`, `preprocess_DIP_TC_new.py`, `README.md`

- `IMUPoser/`
  - Repository: https://github.com/FIGLAB/IMUPoser
  - Why it is useful: a clean later-generation research codebase that keeps synthetic AMASS IMU generation while adapting the virtual sensor layout to phone/watch/earbud style placements.
  - Key files: `scripts/1. Preprocessing/1. preprocess_all.py`, `README.md`

- `MobilePoser/`
  - Repository: https://github.com/SPICExLAB/MobilePoser
  - Why it is useful: extends the IMUPoser-style preprocessing line, adds 30 FPS processing, contact labels, and explicit TotalCapture handling with a mobile-device-oriented sensor layout.
  - Key files: `mobileposer/process.py`, `README.md`

- `WheelPoser/`
  - Repository: https://github.com/axle-lab/WheelPoser
  - Why it is useful: shows how the same synthetic IMU preprocessing pattern can be adapted to a different population and a different sparse sensor layout.
  - Key files: `scripts/1. Preprocessing/1.1 preprocess_all.py`, `README.md`
