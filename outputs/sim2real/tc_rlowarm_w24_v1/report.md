# sim2real L2 Report — tc_rlowarm_w24_v1

Testbed: IMU→motion retrieval, 24-frame windows, subject-disjoint split, test gallery 1005 real windows (chance R@1 = 0.0010).

## Anchors

- **TRTR (train real, test real): R@1 = 0.0395 ± 0.0066, R@5 = 0.1472 ± 0.0071** — 40× above chance.
- Shuffled-pairs control: R@1 = 0.001 (chance 0.0010) — **at chance, no leakage** ✅

## TSTR generator ranking (train synthetic, test real)

| generator (train source) | R@1 | R@5 | sim-to-real gap (TRTR−TSTR R@1) |
|---|---|---|---|
| synth_naive_8f7d9e76 | 0.0226 ± 0.0033 | 0.0816 ± 0.0059 | 0.0169 |
| synth_humogen_e8cdbc46 | 0.0023 ± 0.0005 | 0.0103 ± 0.0020 | 0.0371 |
| synth_globalpose_42c7b0f2 | 0.0013 ± 0.0005 | 0.0083 ± 0.0005 | 0.0381 |
| synth_globalpose_8026be9d | 0.0013 ± 0.0009 | 0.0110 ± 0.0029 | 0.0381 |

## Mix (real + synthetic as augmentation)

| train composition | R@1 | R@5 | Δ vs TRTR |
|---|---|---|---|
| real+synth_naive_8f7d9e76 | 0.0541 ± 0.0045 | 0.1871 ± 0.0069 | +0.0146 |
| real+synth_globalpose_42c7b0f2 | 0.0385 ± 0.0061 | 0.1555 ± 0.0151 | -0.0010 |

## Sim-boost curve (synthetic pretraining vs real-data budget)

| real fraction | real-only R@1 | pretrain synth_globalpose_42c7b0f2 R@1 (boost) | pretrain synth_naive_8f7d9e76 R@1 (boost) |
|---|---|---|---|
| 0.1 | 0.0087 ± 0.0012 | 0.0037 ± 0.0012 (-0.0050) | 0.0169 ± 0.0028 (+0.0082) |
| 0.25 | 0.0196 ± 0.0046 | 0.0060 ± 0.0035 (-0.0136) | 0.0060 ± 0.0016 (-0.0136) |
| 1 | 0.0395 ± 0.0066 | 0.0435 ± 0.0038 (+0.0040) | 0.0408 ± 0.0100 (+0.0013) |

## Notes

- All values mean ± std over seeds; ledger: `results.jsonl` (single source of truth).
- Normalization: channel stats from the training source, applied unchanged to val/test (honest TSTR).
- val = S4 (hyperparameter surface), test = S5 (touched once per cell).
