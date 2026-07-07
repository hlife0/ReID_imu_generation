# sim2real per-sequence IMU<->motion alignment (naive_bridge_lagscan_v1)

`imu_motion_lag = k` means `real[i] <-> motion[i+k]`. Estimated by acc-magnitude
sliding Pearson against the deterministic naive synth stream (motion timebase).

| sequence | real | motion | lag | corr@lag |
|---|---|---|---|---|
| S1_acting1 | 4115 | 4114 | -1 | 0.9212 |
| S1_acting2 | 3704 | 3703 | -1 | 0.1199 |
| S1_acting3 | 2770 | 2769 | -1 | 0.8217 |
| S1_freestyle1 | 2313 | 2311 | -1 | 0.9549 |
| S1_freestyle2 | 2464 | 2464 | -1 | 0.8502 |
| S1_freestyle3 | 2040 | 2040 | +0 | 0.9751 |
| S1_rom1 | 4892 | 4891 | +0 | 0.8813 |
| S1_rom2 | 6777 | 6776 | -1 | 0.8975 |
| S1_rom3 | 5944 | 5942 | -1 | 0.9705 |
| S1_walking1 | 3671 | 3671 | +0 | 0.9266 |
| S1_walking2 | 3189 | 3189 | +0 | 0.583 |
| S1_walking3 | 3397 | 3397 | +0 | 0.3899 |
| S2_acting1 | 3387 | 3386 | -1 | 0.9354 |
| S2_acting2 | 3808 | 3809 | +0 | 0.9727 |
| S2_acting3 | 4828 | 4827 | -2 | 0.875 |
| S2_freestyle2 | 2092 | 2090 | -3 | 0.8712 |
| S2_rom1 | 6419 | 6421 | +2 | 0.672 |
| S2_rom2 | 5939 | 5938 | +1 | 0.8326 |
| S2_rom3 | 5993 | 5994 | +0 | 0.7808 |
| S2_walking1 | 3518 | 3518 | -1 | 0.8451 |
| S2_walking2 | 3575 | 3575 | +0 | 0.9443 |
| S2_walking3 | 3553 | 3551 | -1 | 0.879 |
| S3_acting2 | 3712 | 3712 | -1 | 0.9625 |
| S3_rom1 | 5700 | 5699 | -1 | 0.9618 |
| S3_rom2 | 5319 | 5319 | +0 | 0.976 |
| S3_rom3 | 4858 | 4859 | -1 | 0.9722 |
| S3_walking1 | 3538 | 3538 | -1 | 0.6314 |
| S3_walking2 | 3239 | 3238 | -1 | 0.8999 |
| S3_walking3 | 3388 | 3387 | -1 | 0.9781 |
| S4_freestyle1 | 2020 | 2019 | -1 | 0.8557 |
| S4_freestyle3 | 2826 | 2826 | -1 | 0.9118 |
| S4_rom3 | 5430 | 5430 | -1 | 0.8868 |
| S4_walking2 | 3376 | 3377 | +1 | 0.9192 |
| S5_freestyle1 | 3073 | 3073 | +0 | 0.9805 |
| S5_freestyle3 | 3561 | 3605 | +44 | 0.9409 |
| S5_rom3 | 5758 | 5756 | -1 | 0.9513 |
| S5_walking2 | 3738 | 3737 | +0 | 0.6703 |

## Warnings

- S1_acting2: bridge corr 0.120 < 0.3
