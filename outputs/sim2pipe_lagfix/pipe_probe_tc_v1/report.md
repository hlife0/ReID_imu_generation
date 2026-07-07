# sim2pipe report — pipe_probe_tc_v1

Synthetic IMU judged by the **main project's own model** (IMUVideoMatcher: frozen MotionBERT + frozen DeSPITE, alignment head trained; SymmetricInfoNCE) via subprocess into its environment. Metric = in-batch retrieval top1 on held-out **real** windows (batch 64, random line ≈ 0.0156).

## Pipe-probe ranking (mean±std over seeds)

| protocol | stream | val_top1 (S4 real) | test_top1 (S5 real) | n |
|---|---|---|---|---|
| trtr | real | 0.2672±0.0061 | 0.2976±0.0085 | 3 |
| tstr | synth_naive_8f7d9e76 | 0.0129±0.0032 | 0.0072±0.0018 | 3 |
| mix | synth_naive_8f7d9e76 | 0.2778±0.0072 | 0.2934±0.0158 | 3 |

## Probe (sim2real) vs pipe (sim2pipe): TSTR consistency

| stream | probe R@1 | pipe test_top1 |
|---|---|---|
| synth_naive_8f7d9e76 | 0.0156 | 0.0072 |

- probe order: synth_naive_8f7d9e76
- pipe order:  synth_naive_8f7d9e76
- **rank agreement: True**; pipe TSTR spread = 0.0000 (vs random line 0.0156).
- pipe TSTR streams are within one random-line of each other — **the real model does not distinguish the generators; the probe's ranking did not survive.**
