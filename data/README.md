# Data Layout

This repository keeps data outside git by default and currently uses the following working split:

- `processed/`
  - staged workflow inputs ready to use directly
  - now mainly reserved for ad hoc staging; the previous `S1_freestyle3` sample has been archived under `legacy/processed/`
- `interim/`
  - derived intermediate products generated from processed inputs
  - now mainly reserved for ad hoc intermediates; the previous `S1_freestyle3` synthetic CSV has been archived under `legacy/interim/`
- `legacy/`
  - archived data products from older workflows kept for reproducibility
  - current archived example: the older staged `TotalCapture` `S1_freestyle3` sample and its synthetic IMU CSV

The current maintained workflow stages a raw `TotalCapture` sample into `raw/totalcapture/S1_freestyle3/`, extracts raw IMU into `raw/totalcapture_test/GlobalPose_origin/`, and writes run artifacts under `outputs/totalcapture_test/GlobalPose_origin/`.
