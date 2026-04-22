# Repository Conventions

This document defines the terminology boundary for this repository.

## Hard scope boundary

This repository does **not** treat `mocap`, `Vicon`, or `IMU` as a discussion about sensing hardware, capture systems, or acquisition technology.

Within this repository, those terms are used only in the following sense:

- as already-collected raw data
- as stored data formats
- as on-disk data representations that later steps will transform

In other words, once data have been captured and written to disk, the repository discusses them as data formats or raw-data structures.

## Category rules used in this repository

### Data formats / data representations

The following should be treated as data formats or stored representations in repository discussions:

- `mocap`
- `Vicon`
- `IMU`
- `SMPL`
- `SMPL-X`
- any text, binary, `npz`, `pkl`, `pt`, video, or sensor stream file already saved on disk

For this repository's purposes, terms like `Vicon` or `IMU` refer to the raw stored data and their file structure, not to the physical capture device.

### Processing techniques

`processing techniques` means any method that takes already-collected data in one format and:

- converts them into another format
- aligns them
- calibrates them
- filters them
- denoises them
- fits a body model to them
- generates synthetic outputs from them
- extracts derived variables from them

Examples:

- `Vicon -> SMPL-X`
- `raw IMU -> calibrated IMU`
- `SMPL-X -> synthetic IMU`
- `raw video + IMU -> aligned windows`

## Current workflow-specific convention

The current `totalcapture_test` workflow standardizes real and synthetic IMU into a common CSV layout:

- `frame_idx`
- quaternion: `quat0..quat3`
- acceleration: `acc_x..acc_z`
- gyroscope: `gyro_x..gyro_z`
- magnetic field: `mag_x..mag_z`

Any new real/synthetic IMU path added to this repository should either use this exact layout or document why it differs.

### Datasets

Datasets are collections of samples stored in one or more of the formats above.

Examples:

- `TotalCapture`
- `DIP-IMU`
- `AMASS`

## Practical interpretation rule

If a term can be interpreted either as:

- the real-world capture technology
- or the stored raw data format used by this repository

then this repository defaults to the second interpretation.

So in normal repository discussion:

- `Vicon` means the stored `Vicon` ground-truth files
- `IMU` means the stored IMU stream files
- `mocap` means stored raw motion-capture data

unless a document explicitly says it is discussing acquisition hardware or sensing setup.
