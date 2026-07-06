# Sim-to-Pipe 评估子系统设计草案 v0（sim2pipe）

> 状态：**v1 定稿（2026-07-06，hrli 拍板）**。本文档是 `src/sim2pipe/` + `scripts/sim2pipe/` 的唯一设计依据（对齐 `sim2real_design.md` 的角色）。
> 已拍板决策：① 边界方式=subprocess 文件契约，不搬模型代码不改主项目；② 不等 F2，现有四条生成流直接上，F2 出来后作为新格子追加；③ **本期范围只到 P0+P1**，P2（custom 微调 FrameAcc）另行立项。
> 前置阅读：`docs/sim2real_design.md`、`docs/sim2real_findings_v1.md`、`docs/sim2real_findings_estskel.md`。

## 1. 目标与动机

**一句话：让主项目自己的量具（IMUVideoMatcher + FrameAcc）而不是我们自建的 probe 来裁决合成 IMU 的质量。**

sim2real 用一个小型双塔 probe 回答了"哪个生成器的数据下游最能打"（naive ≫ humogen ≈ globalpose），但 probe 终究是我们自己造的尺子。主项目（`Rd-id-Project`）的真实消费方是 `IMUVideoMatcher`（DeSPITE LSTM IMU 塔 + MotionBERT 骨架塔 + SymmetricInfoNCE），最终指标是 custom 数据上的 **FrameAcc**（E10b SOTA 0.613±0.010，G4 目标 ≥0.75）。sim2pipe 要回答三个递进的问题：

1. **排名复验**：sim2real 的 probe 排名（naive 第一、比力语义决定迁移）换成主项目的真实模型后还成立吗？
2. **绝对收益**：合成数据通过主项目 I5 钩子（`--init_alignment_ckpt` 合成预训练 → custom 微调）能把 FrameAcc 从 0.613 抬多少？这是能直接写进 G4 账本的数字。
3. **交付格式定版**：为主项目产出即插即用的合成数据包（48 维格式 + 窗口 CSV + 预训练 checkpoint），把 AMASS 扩产的出口格式一次定死。

**与 sim2real 的关系：平行而非替代。** sim2real 是快速迭代档（probe 分钟级，生成器改一版当天出排名）；sim2pipe 是最终裁决档（跑主项目全模型，小时级，只对 sim2real 筛出的胜者跑）。这正是 sim2real 设计里"L1 预测 L2 就只跑 L1"逻辑的再上一层：**L2(probe) 预测 L3(pipe) 就只定期跑 L3。**

## 2. 核心设计决策：数据出境、命令出境、结果回境

**不把主项目模型搬进本仓库，也不改主项目代码。** 主项目已有现成的合成 IMU 消费接口（`configs/G_synthetic_data_E*.yaml`：`preprocess.imu_source: synthetic` + `synthetic_imu_root` glob `*.npz`；训练侧 `--init_alignment_ckpt`/`--imu_ckpt`；评测侧 `src.engine.eval_synchronous` 产 FrameAcc）。sim2pipe 与主项目的边界和生成器契约同构——**文件级 + subprocess**：

- **出境（export）**：把 sim2real 语料转成主项目吃的格式，落到 `data/interim/sim2pipe/export/`；
- **出境（drive）**：用主项目自己的 conda 环境（`test_reid`）subprocess 调 `python -m src.engine.train` / `eval_synchronous`，配置通过我们生成的 yaml overlay 传入；
- **回境（ledger）**：解析主项目输出的 `metrics.json`/log，追加进本仓库 `outputs/sim2pipe/results.jsonl` 账本，报告在本仓库聚合。

这样主项目对 sim2pipe 完全无感知（零 PR 依赖），环境差异（本仓库 adaptfm torch vs 主项目 test_reid torch 2.1+cu118）被 subprocess 边界诚实隔离。

## 3. 两级评测（P1 / P2）

### P1：pipe-probe —— 主项目模型 × sim2real 协议（TotalCapture）

把 sim2real 的自建 probe 换成主项目真身，其余全部复用：同一份 37 序列语料、同一冻结切分 `totalcapture_subject_v1`（主项目 `TotalCaptureAdapter` 本就支持 subject 级切分，配置对齐 S1–S3/S4/S5 即可）、同一 TRTR/TSTR/mix 协议族。

- IMU 侧：13 通道 → 48 维（quat→9 维旋转矩阵 + 3 维 acc，R_LowArm 复制 ×4 槽位，即主项目 `repeat_single_sensor=4` 单传感器模式）；
- 骨架侧：**直接用 estskel 变体的 `motion_estimated.npz`**——`gen_estimated.py` 本就重定向到 H36M17，与 MotionBERT 输入 `(T,17,3)` 天然对齐（这是 estskel 工作的直接回报）；
- 指标：主项目 `val_top1`/`test_top1`（batch 内检索），语义与 sim2real probe R@1 同族，可直接对照排名。
- **回答问题 1。** 无需视频、无需 custom 数据，成本 ≈ 训练若干次主项目模型（小时级 × 格子数，矩阵必须比 sim2real 的 46 格小得多，见 §6）。

### P2：full-pipe —— 合成预训练 → custom 微调 → FrameAcc（**本期不做，另行立项**）

> 2026-07-06 决策：sim2pipe v1 范围只到 P0+P1（回答排名复验问题）。P2 依赖 custom 数据链路与主项目实验纪律（seed 纪律、SOTA 表规则、与 justlanxuan 对流程），收益数字关系到 G4 账本，单独立项更干净。以下保留作为立项时的起点：

1. 用主项目在导出的合成 TotalCapture（未来 AMASS）语料上预训练 alignment checkpoint；
2. `--init_alignment_ckpt` 接 E10b 协议（per-video 7:3，`segment_frames=1800`）在 custom 真实数据上微调；
3. `eval_synchronous` 出 FrameAcc + HOTA，**6 seeds [0,42,123,1,2,3]**（G4 纪律）；
4. 对照组：无合成预训练的 from-scratch 微调，必须复现 0.613±0.010（健全性锚点，等价 sim2real 的 TRTR 锚点——复现不了说明测试床坏，一切增益数字作废）。

- **回答问题 2，产出可直接进主项目 `experiments/G4` 账本和 SOTA 表的数字。**

### 门禁 P0（进入 P1 前必须过）

1. **格式往返**：导出的 npz 能被主项目 preprocess/slice 无警告吃下，窗口数与 spec 推算一致；
2. **真实流锚点**：把我们的 `real.npz`（TotalCapture 真实 IMU）走一遍导出→主项目训练，val_top1 显著高于随机线——这验证的是 quat 约定（wxyz/xyzw、全局/传感器坐标系）和重采样（60Hz→主项目 fps）没有搞错，**防止转换 bug 冒充生成器差距**（sim2real L0 门禁的同款教训）；
3. **打乱对照**：错配训练回落随机线。

## 4. 目录结构

```
ReID_imu_generation/
├── docs/sim2pipe_design.md            # 本文档
├── src/sim2pipe/
│   ├── __init__.py
│   ├── convert.py                     # 13ch ImuSequence → 48 维；quat→rotmat；重采样（核心纯函数，可单测）
│   ├── export.py                      # 语料/切分 → 主项目 synthetic_imu_root 布局 + 窗口物料
│   ├── overlay.py                     # 生成主项目 yaml overlay（路径/切分/seed/协议 → 配置文件）
│   ├── bridge.py                      # subprocess 驱动主项目 CLI；捕获退出码与产物路径
│   └── ledger.py                      # 解析 metrics.json / multi_seed_summary → results.jsonl 行
├── scripts/sim2pipe/
│   ├── 01_export_corpus.py            # 全语料导出（--generators --split ...）
│   ├── 02_gate_export.py              # P0 门禁三件套
│   ├── 03_run_pipe_probe.py           # P1：读 matrix yaml，账本续跑
│   ├── 04_run_pretrain_finetune.py    # P2：预训练 → custom 微调 → FrameAcc
│   └── 05_report.py                   # 聚合 + 与 sim2real 排名对照表
├── configs/sim2pipe/
│   ├── paths.yaml                     # 主项目仓库路径、test_reid python、外部 ckpt 路径（机器特定，git 只进 example）
│   ├── matrix_pipe_v1.yaml            # P1 矩阵（协议 × 生成器 × seed）
│   └── overlays/                      # 主项目 yaml 模板（P1 totalcapture / P2 custom E10b）
├── data/interim/sim2pipe/export/      # 导出物料（不进 git）
├── outputs/sim2pipe/<benchmark_id>/   # results.jsonl + report.md（进 git，同 sim2real 惯例）
└── tests/
    ├── test_sim2pipe_convert.py       # quat→rotmat 往返、通道顺序、重采样确定性
    └── test_sim2pipe_export_contract.py
```

复用不重写：切分与泄漏检查直接 import `src.sim2real.splits`；语料读取用 `src.sim2real.contracts`；账本/续跑惯例照抄 sim2real（`results.jsonl` 追加式、启动跳过已有格子）。

## 5. 风险与开放问题（定稿前需拍板）

| # | 问题 | 现状/倾向 |
|---|---|---|
| R1 | **quat/坐标系约定**：主项目 48 维里 9 维旋转矩阵的参考系（全局？传感器→骨盆？）与我们 quat 的约定是否一致 | P0 真实流锚点兜底；实现前先读 `Rd-id-Project/src/data/preprocess/custom.py` 的 `convert_single_imu_to_48` 确认 |
| R2 | **acc 语义（比力 vs 自由加速度）**：主项目 custom 真实 IMU 是比力，naive 生成器输出是什么帧语义——F2 问题在 pipe 侧同样存在 | P1 矩阵里保留 clean/现状两档；F2 变体出来后优先进 sim2pipe |
| R3 | **采样率**：语料 60Hz，主项目 custom 链路是 100Hz 低通后重采样到视频 fps（~30Hz），窗口 24@30Hz≈0.8s | 导出时统一重采样到主项目 target fps，写进 export manifest |
| R4 | **外部路径**：MotionBERT/despite ckpt 在 `/home/fzliang/...`，机器特定 | 全部收进 `configs/sim2pipe/paths.yaml`，缺失时 02 门禁直接报清单 |
| R5 | **算力**：主项目训练一格远贵于 probe 一格 | P1 矩阵瘦身：协议 {trtr, tstr, mix} × 生成器 {naive, globalpose_clean, humogen} × 3 seeds ≈ 21 格封顶；P2 只跑胜者 ×6 seeds |
| R6 | **P2 依赖 custom 数据与主项目实验纪律**（SOTA 表更新规则、experiments 目录归属 justlanxuan） | P2 的数字先落本仓库账本，进主项目 G4 账本前与 justlanxuan 对流程 |

## 6. 里程碑（草案）

| 里程碑 | 内容 | DoD | 预估 |
|---|---|---|---|
| **N0** | 骨架 + convert/export 契约 + paths.yaml + 本文档定稿 | 转换单测绿；02 门禁能报缺失依赖清单 | 0.5 天 |
| **N1** | P0 门禁三件套通过 | 真实流锚点 val_top1 ≫ 随机线，打乱=随机线 | 1–2 天 |
| **N2** | P1 pipe-probe 矩阵 | 排名表 + 与 sim2real probe 排名的一致性判定 | 2–3 天（算力主导） |
| **N3** | 报告 + 交付格式说明 | report.md + 与 sim2real 排名对照 + 48 维出口格式文档 | 1 天 |

> P2（合成预训练→custom 微调→FrameAcc，6 seeds）已移出本期范围，另行立项；原 N3 里程碑内容见 §3 P2 小节。

## 7. 刻意不做的事

- **不 vendor 主项目模型代码**：环境不同、演进不同步，subprocess 文件契约是诚实边界（同生成器契约的理由）。
- **不在 P1 就上视频/FrameAcc**：TotalCapture 无主项目视频链路，P1 用 top1 检索即可回答排名问题；FrameAcc 只在 P2 的 custom 上出现。
- **不替代 sim2real**：日常生成器迭代仍走 probe；sim2pipe 只在"要对外报数/定交付格式/验排名"时跑。
- **矩阵不贪大**：P1 ≤ 21 格起步，先验证 L2(probe)↔L3(pipe) 排名相关性，相关则以后更省。
