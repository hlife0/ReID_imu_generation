# sim2real L2 Report — tc_rlowarm_w24_estskel_v1

Testbed: IMU→motion retrieval, 24-frame windows, subject-disjoint split, test gallery 1005 real windows (chance R@1 = 0.0010).

## Anchors

- **TRTR (train real, test real): R@1 = 0.0414 ± 0.0111, R@5 = 0.1403 ± 0.0123** — 42× above chance.
- Shuffled-pairs control: R@1 = 0.0 (chance 0.0010) — **at chance, no leakage** ✅

## TSTR generator ranking (train synthetic, test real)

| generator (train source) | R@1 | R@5 | sim-to-real gap (TRTR−TSTR R@1) |
|---|---|---|---|
| synth_naive_8f7d9e76 | 0.0156 ± 0.0012 | 0.0803 ± 0.0062 | 0.0259 |
| synth_globalpose_8026be9d | 0.0020 ± 0.0008 | 0.0090 ± 0.0021 | 0.0394 |
| synth_humogen_e8cdbc46 | 0.0020 ± 0.0016 | 0.0090 ± 0.0029 | 0.0394 |
| synth_globalpose_42c7b0f2 | 0.0013 ± 0.0012 | 0.0063 ± 0.0017 | 0.0401 |

## Mix (real + synthetic as augmentation)

| train composition | R@1 | R@5 | Δ vs TRTR |
|---|---|---|---|
| real+synth_naive_8f7d9e76 | 0.0547 ± 0.0014 | 0.1685 ± 0.0074 | +0.0133 |
| real+synth_globalpose_42c7b0f2 | 0.0488 ± 0.0008 | 0.1496 ± 0.0121 | +0.0074 |
| real+real | 0.0375 ± 0.0034 | 0.1267 ± 0.0074 | -0.0040 |

## Sim-boost curve (synthetic pretraining vs real-data budget)

| real fraction | real-only R@1 | pretrain synth_globalpose_42c7b0f2 R@1 (boost) | pretrain synth_naive_8f7d9e76 R@1 (boost) |
|---|---|---|---|
| 0.1 | 0.0103 ± 0.0062 | 0.0023 ± 0.0005 (-0.0079) | 0.0136 ± 0.0033 (+0.0033) |
| 0.25 | 0.0152 ± 0.0033 | 0.0023 ± 0.0005 (-0.0129) | 0.0110 ± 0.0043 (-0.0043) |
| 1 | 0.0414 ± 0.0111 | 0.0335 ± 0.0150 (-0.0079) | 0.0302 ± 0.0060 (-0.0113) |

## Notes

- All values mean ± std over seeds; ledger: `results.jsonl` (single source of truth).
- Normalization: channel stats from the training source, applied unchanged to val/test (honest TSTR).
- val = S4 (hyperparameter surface), test = S5 (touched once per cell).
