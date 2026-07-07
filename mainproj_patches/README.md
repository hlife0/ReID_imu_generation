# mainproj_patches — 对主项目 Rd-id-Project 的按需补丁

本目录存放 sim2pipe 评测所需、但必须作用在**主项目**（`Rd-id-Project`）源码上的改动。
主项目是他人（justlanxuan）仓库，本仓库**不**直接向其提交——改动以补丁形式保存在这里，
按需临时套用、用完还原，主项目工作树与远端始终保持原状。

## destroy_pairing_in_batch.patch

给 `Rd-id-Project/src/engine/train.py` 增加一个合法的配对破坏负控开关
`--destroy_pairing_in_batch`：打乱 batch 内 video 但**不**重标记检索 target，使每条 IMU
训练时对着错误 video，对齐头学不到一致映射，held-out 测试落到冻结编码器地板。

- 与主项目既有 `--shuffle_video_in_batch`（会重标记、**保留**正确配对，只是位置偏置控制、
  不是配对破坏）互斥且有 guard；默认关闭，不带该 flag 时主项目行为完全不变。
- 纯新增（+25 行），针对 `origin/egohumans`（基线 commit `6f4cebd`）。

### 为什么不直接提交到主项目
sim2pipe 的裁判是主项目真身（冻结 DeSPITE+对齐头），合法地板必须在主项目 train 循环里测，
但按仓库约定本方向的产出只落在 `ReID_imu_generation`。故改动留作补丁，不污染主项目历史/远端。

### 用法（临时套用 → 复现 → 还原）
```bash
MAIN=/home/hrli/ReID-workspace/Rd-id-Project
# 套用
git -C "$MAIN" apply /home/hrli/ReID-workspace/ReID_imu_generation/mainproj_patches/destroy_pairing_in_batch.patch
# 复现真实地板（sim2pipe 驱动的 --destroy-pairing 会向 train 传 --destroy_pairing_in_batch）
/home/hrli/data_generation/.venv/bin/python scripts/sim2pipe/03_run_pipe_probe.py \
    --only real --destroy-pairing --seeds 0 42 123 \
    --outputs-root outputs/sim2pipe_true_floor --device cuda:2
# 还原（主项目回到与远端逐字一致）
git -C "$MAIN" apply -R /home/hrli/ReID-workspace/ReID_imu_generation/mainproj_patches/destroy_pairing_in_batch.patch
```
或用 `mainproj_patches/apply.sh apply|revert|check`。

### 已记录的结果（无需重跑即可引用）
真实配对破坏地板 test_top1 = **0.0201 ± 0.0034**（real 流 3 seeds，`outputs/sim2pipe_true_floor/`）。
train_top1 在 destroy 下钉在 0.0155≈随机（对照 shuffle 版爬到 0.82），确认配对确被破坏。
结论见 `docs/sim2pipe_findings_v1.md`。
