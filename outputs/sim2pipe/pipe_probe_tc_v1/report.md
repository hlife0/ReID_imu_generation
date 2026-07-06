# sim2pipe report — pipe_probe_tc_v1

Synthetic IMU judged by the **main project's own model** (IMUVideoMatcher: frozen MotionBERT + frozen DeSPITE, alignment head trained; SymmetricInfoNCE) via subprocess into its environment. Metric = in-batch retrieval top1 on held-out **real** windows (batch 64, random line ≈ 0.0156).

## Pipe-probe ranking (mean±std over seeds)

| protocol | stream | val_top1 (S4 real) | test_top1 (S5 real) | n |
|---|---|---|---|---|
| trtr | real | 0.2714±0.0083 | 0.2518±0.0055 | 3 |
| tstr | synth_globalpose_42c7b0f2 | 0.0145±0.0046 | 0.0098±0.0020 | 3 |
| tstr | synth_humogen_e8cdbc46 | 0.0110±0.0012 | 0.0128±0.0016 | 3 |
| tstr | synth_naive_8f7d9e76 | 0.0140±0.0043 | 0.0127±0.0020 | 3 |
| mix | synth_globalpose_42c7b0f2 | 0.2699±0.0060 | 0.2601±0.0010 | 3 |
| mix | synth_humogen_e8cdbc46 | 0.2417±0.0034 | 0.2374±0.0102 | 3 |
| mix | synth_naive_8f7d9e76 | 0.2604±0.0061 | 0.2406±0.0136 | 3 |

## Probe (sim2real) vs pipe (sim2pipe): TSTR consistency

| stream | probe R@1 | pipe test_top1 |
|---|---|---|
| synth_naive_8f7d9e76 | 0.0172 | 0.0127 |
| synth_humogen_e8cdbc46 | 0.0020 | 0.0128 |
| synth_globalpose_42c7b0f2 | 0.0013 | 0.0098 |

- probe order: synth_naive_8f7d9e76 > synth_humogen_e8cdbc46 > synth_globalpose_42c7b0f2
- pipe order:  synth_humogen_e8cdbc46 > synth_naive_8f7d9e76 > synth_globalpose_42c7b0f2
- **rank agreement: False**; pipe TSTR spread = 0.0030 (vs random line 0.0156).
- pipe TSTR streams are within one random-line of each other — **the real model does not distinguish the generators; the probe's ranking did not survive.**
