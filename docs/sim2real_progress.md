# 🔄 sim2real 子系统进展与恢复节点

> 设计文档：`docs/sim2real_design.md`；发现与交接：`docs/sim2real_findings_v1.md`。

## 1. 当前所处阶段

- **M0–M4 全部完成，M5 核心问题已回答，M6 交付物已产出（2026-07-05，单日）。**
- 基准 `tc_rlowarm_w24_v1` 完整可复现：语料 → 门禁 → 窗口 → L1 → L2 矩阵 → 报告。

## 2. 已完成与关键数字

- ✅ M1 语料：37 序列（AMASS∩lxhong，S1:12/S2:10/S3:7/S4:4/S5:4）× 4 合成流
  （globalpose noise_full/clean、humogen、naive）+ 真实流；`01_build_corpus.py`，
  8 并发全量约 2 分钟；SMPL-X 前向每序列仅一次。
- ✅ M1 门禁：规则 = acc 与 gyro 幅值相关**同时**低于 0.10 才判配对断裂；148 流全 PASS，
  16 条单通道弱（10 条 humogen，重力轴问题）记录为 weak_channels 保留进基准。
- ✅ M2 窗口：train 7353 窗 × 5 源 / val 849 / test 1005；泄漏检查干净；
  L1 表（Fréchet：globalpose_clean 448 < noise_full 517 < naive 1138 < humogen 1485；
  2026-07-07 更正：原文写作 "globalpose 448 < clean 517"，clean/noise_full 标签反了——
  config_hash 复算真值为 42c7b0f2=clean、8026be9d=noise_full）。
- ✅ M3/M4 矩阵：46 格全部完成（4×4090 并行，单格 3–12 秒）。
  **TRTR 0.0395±0.0066（40×随机线）；TSTR：naive 0.0226 ≫ humogen 0.0023 ≈ globalpose 0.0013；
  mix 真实+naive 0.0541（+37% 相对 TRTR）；打乱配对对照 = 随机线 ✅。**
- ✅ M5 核心问题：clean 与 noise_full 的 TSTR 同样为随机线 ⇒ 迁移失败源于帧语义
  （自由加速度 vs 传感器系比力），与噪声堆栈无关。完整逐开关消融留待 F2 变体。
- ✅ M6：`report.md`（自动聚合）+ `sim2real_findings_v1.md`（解读/交接/AMASS 计划）。

## 3. 运行环境备忘

- 生成侧 venv：`/home/hrli/data_generation/.venv/bin/python`（torch cu130，**CUDA 不可用**，
  驱动只到 12.6——仅用于 SMPL-X 前向与合成，CPU 足够）。
- L2 训练环境：`/home/hrli/miniconda3/envs/adaptfm/bin/python`（torch 2.10+cu128，CUDA ✅），
  是 `06_run_matrix.py --cell-python` 的默认值。
- `src` 命名冲突：data_generation 的 `src` 是常规包，会遮蔽本仓库 `src` 命名空间——
  venv 侧脚本一律把 `REPO_ROOT/src` 入 path、以 `sim2real.*` 导入（脚本头有注释）。

## 4. 当前阻塞

- 无。

## 5. 下一步行动（v2 迭代候选，按优先级）

- [ ] **F2 验证**：globalpose 增加"传感器系比力输出"配置开关，重跑 TSTR——
      预期大幅回升；随后做完整噪声逐开关消融（真 M5）。
- [ ] `humogen_yup.json`：修正重力轴的变体配置（保留 origin 对照）。
- [ ] 与主项目对接口（custom IMU 7 维格式/采样率），启动 AMASS 扩产（用当前胜者 naive）。
- [ ] 可选：w100 变体 benchmark（`tc_rlowarm_w100_v1`，只需新 spec 文件）；
      L1 换 probe 编码器复检 L1↔L2 相关性；sim-boost 低比例段加 seeds。
- [ ] 向 hlife0/ReID_imu_generation 提交本次全部改动（工作树尚未 commit）。
