# sim2real L2 Report — tc_rlowarm_w24_v1

Testbed: IMU→motion retrieval, 24-frame windows, subject-disjoint split, test gallery 1005 real windows (chance R@1 = 0.0010).

## Anchors

- **TRTR (train real, test real): R@1 = 0.0607 ± 0.0144, R@5 = 0.1831 ± 0.0258** — 61× above chance.
- Shuffled-pairs control: R@1 = 0.002 (chance 0.0010) — **at chance, no leakage** ✅

## TSTR generator ranking (train synthetic, test real)

| generator (train source) | R@1 | R@5 | sim-to-real gap (TRTR−TSTR R@1) |
|---|---|---|---|
| synth_naive_8f7d9e76 | 0.0212 ± 0.0012 | 0.0853 ± 0.0037 | 0.0395 |
| synth_globalpose_42c7b0f2 | 0.0020 ± 0.0008 | 0.0063 ± 0.0029 | 0.0587 |
| synth_humogen_e8cdbc46 | 0.0017 ± 0.0005 | 0.0050 ± 0.0029 | 0.0590 |
| synth_globalpose_8026be9d | 0.0013 ± 0.0012 | 0.0090 ± 0.0016 | 0.0594 |

## Mix (real + synthetic as augmentation)

| train composition | R@1 | R@5 | Δ vs TRTR |
|---|---|---|---|
| real+synth_naive_8f7d9e76 | 0.0597 ± 0.0062 | 0.1798 ± 0.0133 | -0.0010 |
| real+synth_globalpose_42c7b0f2 | 0.0531 ± 0.0116 | 0.1867 ± 0.0276 | -0.0076 |
| real+real | 0.0488 ± 0.0077 | 0.1536 ± 0.0059 | -0.0119 |

## Sim-boost curve (synthetic pretraining vs real-data budget)

| real fraction | real-only R@1 | pretrain synth_globalpose_42c7b0f2 R@1 (boost) | pretrain synth_naive_8f7d9e76 R@1 (boost) |
|---|---|---|---|
| 0.1 | 0.0067 ± 0.0017 | 0.0017 ± 0.0005 (-0.0050) | 0.0119 ± 0.0021 (+0.0053) |
| 0.25 | 0.0209 ± 0.0043 | 0.0090 ± 0.0037 (-0.0119) | 0.0070 ± 0.0016 (-0.0139) |
| 1 | 0.0607 ± 0.0144 | 0.0385 ± 0.0117 (-0.0222) | 0.0438 ± 0.0106 (-0.0169) |

## Notes

- All values mean ± std over seeds; ledger: `results.jsonl` (single source of truth).
- Normalization: channel stats from the training source, applied unchanged to val/test (honest TSTR).
- val = S4 (hyperparameter surface), test = S5 (touched once per cell).
