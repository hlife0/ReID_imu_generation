# sim2real L2 Report — tc_rlowarm_w24_estskel_v1

Testbed: IMU→motion retrieval, 24-frame windows, subject-disjoint split, test gallery 1005 real windows (chance R@1 = 0.0010).

## Anchors

- **TRTR (train real, test real): R@1 = 0.0362 ± 0.0066, R@5 = 0.1207 ± 0.0026** — 36× above chance.
- Shuffled-pairs control: R@1 = 0.001 (chance 0.0010) — **at chance, no leakage** ✅

## TSTR generator ranking (train synthetic, test real)

| generator (train source) | R@1 | R@5 | sim-to-real gap (TRTR−TSTR R@1) |
|---|---|---|---|
| synth_naive_8f7d9e76 | 0.0172 ± 0.0039 | 0.0763 ± 0.0143 | 0.0189 |
| synth_humogen_e8cdbc46 | 0.0020 ± 0.0008 | 0.0090 ± 0.0014 | 0.0342 |
| synth_globalpose_42c7b0f2 | 0.0013 ± 0.0005 | 0.0090 ± 0.0024 | 0.0348 |
| synth_globalpose_8026be9d | 0.0010 ± 0.0008 | 0.0080 ± 0.0008 | 0.0352 |

## Mix (real + synthetic as augmentation)

| train composition | R@1 | R@5 | Δ vs TRTR |
|---|---|---|---|
| real+synth_naive_8f7d9e76 | 0.0521 ± 0.0047 | 0.1585 ± 0.0176 | +0.0159 |
| real+synth_globalpose_42c7b0f2 | 0.0458 ± 0.0122 | 0.1552 ± 0.0134 | +0.0096 |

## Sim-boost curve (synthetic pretraining vs real-data budget)

| real fraction | real-only R@1 | pretrain synth_globalpose_42c7b0f2 R@1 (boost) | pretrain synth_naive_8f7d9e76 R@1 (boost) |
|---|---|---|---|
| 0.1 | 0.0116 ± 0.0020 | 0.0030 ± 0.0008 (-0.0086) | 0.0110 ± 0.0029 (-0.0006) |
| 0.25 | 0.0169 ± 0.0028 | 0.0070 ± 0.0032 (-0.0099) | 0.0163 ± 0.0095 (-0.0006) |
| 1 | 0.0362 ± 0.0066 | 0.0242 ± 0.0082 (-0.0120) | 0.0308 ± 0.0070 (-0.0053) |

## Notes

- All values mean ± std over seeds; ledger: `results.jsonl` (single source of truth).
- Normalization: channel stats from the training source, applied unchanged to val/test (honest TSTR).
- val = S4 (hyperparameter surface), test = S5 (touched once per cell).
