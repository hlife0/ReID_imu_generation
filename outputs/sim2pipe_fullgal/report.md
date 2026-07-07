# sim2pipe full-gallery judgment (sim2real-aligned protocol)

Main-project encoder, sim2real judgment: IMU->motion retrieval over the
FULL test gallery (1005 windows, chance 0.000995). Same window grid and split as tc_rlowarm_w24_estskel_v1.

| protocol | stream | destroy | R@1 mean±std (n) |
|---|---|---|---|
| mix | synth_naive_8f7d9e76 |  | 0.0886±0.0138 (3) |
| trtr | real |  | 0.0872±0.0083 (3) |
| trtr | real | yes | 0.0010±0.0010 (3) |
| tstr | synth_naive_8f7d9e76 |  | 0.0000±0.0000 (3) |
