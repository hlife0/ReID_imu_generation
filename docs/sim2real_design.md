# Sim-to-Real 评估子系统设计与施工计划（sim2real）

> 本文档是 `src/sim2real/` + `scripts/sim2real/` 子系统的唯一设计依据。
> 施工进度见第 2 节里程碑清单与 `docs/sim2real_progress.md`。
> 决策记录：本子系统**内建于本仓库**而非新建仓库（理由见 §11）；若未来出现第二个消费者（如主项目用本基准评测 MoBInd 合成 IMU），因命名空间独立，抽取为独立仓库是机械操作。

## 1. 目标与动机

**一句话：让"下游任务"而不是"信号相似度"来裁决合成 IMU 的质量。**

现有 L0 信号级指标（RMSE / Pearson / PSD，`src/evaluation/imu_csv.py`）无法回答团队真正关心的问题：*哪条生成 pipeline 的合成数据训练出的模型在真实 IMU 上最能打？* 本子系统建立以 **TSTR（Train on Synthetic, Test on Real）** 为核心的评估协议，产出：

1. 三条生成 pipeline（GlobalPose_origin / HuMoGen_origin / naive_kinematics）的**下游迁移排名**；
2. 每条 pipeline 的 **sim-to-real gap**（= TRTR − TSTR）；
3. **sim-boost 曲线**：合成预训练在不同真实数据预算下带来的增益——直接对接主项目 G4 的 I4（数据增强）与 I5（预训练）假设，回答"瓶颈是不是数据量"。

**测试床：** TotalCapture。每条序列同时有真实 IMU（R_LowArm）与 SMPL-X 动作，把同一条动作喂给各生成器即得到*同一动作、真假多版 IMU 的平行语料*。传感器锁定 `R_LowArm`（与主项目 custom 场景一致，结论可直接换算）。

## 2. 施工顺序（里程碑与完成判据）★

> 本节是执行主线。每个里程碑有明确 DoD（Definition of Done），未达 DoD 不进入下一阶段。
> 完成后在本节勾选并同步更新 `docs/sim2real_progress.md`。

### 总览

| 里程碑 | 内容 | DoD（完成判据） | 预估 | 状态 |
|---|---|---|---|---|
| **M0** | 骨架 + 数据契约 + 冻结切分 + 本文档 | 契约/切分测试全绿 | 0.5 天 | ✅ 2026-07-05 |
| **M1** | 平行语料：全序列三元组 + 三生成器扇出 + L0 门禁 | 全序列门禁表，排除名单带原因 | 2–3 天 | ✅ 2026-07-05（37 序列×4 流，148 流全 PASS） |
| **M2** | 窗口物化 + L1 分布指标 | 第一张 L1 排名表（3 生成器 × 3 指标） | 2 天 | ✅ 2026-07-05（train 7353×5 源，泄漏干净） |
| **M3** | 检索 probe + TRTR/TSTR + 健全性锚点 | 头版表格：TRTR 上界 + 3 生成器 TSTR，3 seeds mean±std | 3–4 天 | ✅ 2026-07-05（TRTR 40×随机线；打乱对照=随机线） |
| **M4** | 矩阵驱动器 + mix/finetune + 真实比例扫描 | sim-boost 曲线图 | 2–3 天 | ✅ 2026-07-05（46 格，`report.md`） |
| **M5** | GlobalPose 噪声开关消融 | 噪声模块 × 下游指标消融表 | 顺带 | 🔶 核心问题已答（clean≈full ⇒ 语义而非噪声）；逐开关消融待 F2 比力变体 |
| **M6** | 报告 + 交接 | report.md + 给主项目的交接备忘 + AMASS 扩产计划 | 1 天 | ✅ 2026-07-05（`docs/sim2real_findings_v1.md`） |

> 执行明细与结果数字见 `docs/sim2real_progress.md`；发现解读见 `docs/sim2real_findings_v1.md`。
> 本表状态为准；下方各里程碑小节保留立项时的任务分解，不再逐项勾选。

### M0：骨架与契约（✅ 已完成 2026-07-05）

- [x] 目录骨架：`src/sim2real/`、`scripts/sim2real/`、`configs/sim2real/`
- [x] `contracts.py`：MotionSequence / ImuSequence npz 契约、config_hash、manifest 读写
- [x] `splits.py`：切分注册表加载校验、subject 解析、`find_leakage`
- [x] 冻结切分 `configs/sim2real/splits/totalcapture_subject_v1.json`（S1–S3 训 / S4 验 / S5 测）
- [x] 全部 7 个编号脚本与 3 个生成器适配器的 CLI 契约（stub）
- [x] benchmark spec `tc_rlowarm_w24_v1.json`、生成器配置 clean/noise_full、`matrix_v1.yaml`
- [x] 测试：`test_sim2real_contracts.py`、`test_no_split_leakage.py`

### M1：平行语料与 L0 门禁

- [ ] 枚举 lxhong TotalCapture 源中实际可用的序列（S1–S5 × acting/freestyle/rom/walking × takes），缺失记录在案
- [ ] `prepare_triplet.py` 参数化序列名；`corpus.py` + `01_build_corpus.py` 批量驱动
- [ ] 实现三个生成器适配器（globalpose 进程内包 `_run_pipeline_impl.py` 核心并接 switches 开关；humogen/naive subprocess 到外部 venv）
- [ ] 每个生成产物写 `manifest.json`；**用真实 config_hash 回填 `matrix_v1.yaml` 的 TBD 占位符**
- [ ] 确认真实 IMU 采样率与 SMPL-X `mocap_frame_rate`；时间对齐偏移写进每序列 `meta.json`
- [ ] `gate.py` + `02_gate_corpus.py`；在已知良好的 S1_freestyle3 上**标定 gate 阈值**并写回 benchmark spec（阈值为 null 时门禁拒绝运行）
- **DoD：全序列门禁表（PASS/FAIL + 原因），失败序列的排除名单成文。**

### M2：窗口物化与 L1 指标

- [ ] `windows.py` + `03_build_windows.py`：物化 `windows/<benchmark_id>/`，`spec.json` 记录归一化统计量（**只来自训练源**）与语料清单
- [ ] 物化后自动跑 `find_leakage` 断言；新增 `tests/test_windowing_determinism.py`
- [ ] `embed.py`（probe encoder 适配；SIE 适配可选）+ `dist_metrics.py`（Fréchet / MMD / C2ST-AUC）
- [ ] `04_run_l1.py` 产出 L1 排名表
- **DoD：第一张 L1 排名表（3 生成器 × 3 指标）落盘 `outputs/sim2real/`。**

### M3：检索 probe 与 TSTR 头版表格

- [ ] `probe/`：小型双塔（IMU 塔 + motion 塔）+ 对称 InfoNCE + val R@1 早停；**probe 容量全矩阵固定不变**（它是量具不是研究对象）
- [ ] `05_run_l2_cell.py` 支持 trtr / tstr；结果追加 `results.jsonl`
- [ ] 健全性锚点①：TRTR R@1 显著高于随机水平（gallery 随机线一并报出），否则先修测试床
- [ ] 健全性锚点②：`--shuffle-pairs` 对照回落到随机水平，否则排查泄漏
- **DoD：头版表格 —— TRTR 上界 + 3 生成器 TSTR，各 3 seeds mean±std。这是第一个对外可汇报的成果。**

### M4：完整矩阵与 sim-boost 曲线

- [ ] `protocols.py` 补 mix / pretrain_finetune（含 real_fraction 子采样）
- [ ] `06_run_matrix.py`：读 matrix yaml（引入 PyYAML），**账本续跑**（跳过 results.jsonl 已有格子）
- [ ] `07_report.py` + `report.py`：聚合表、sim-to-real gap、sim-boost 曲线图
- **DoD：sim-boost 曲线图 —— 直接回答主项目"瓶颈是不是数据量"。**

### M5：噪声真实感消融（依托 M1–M4，几乎免费）

- [ ] 以 `globalpose_clean.json` ↔ `globalpose_noise_full.json` 为两端，逐开关生成中间配置
- [ ] 复用同一套语料→窗口→L1/L2 链路
- [ ] 顺带检查 **L1 与 L2 排名的相关性**：若 L1 能预测 L2，此后生成器迭代只跑 L1、定期 L2 确认
- **DoD：噪声模块 × 下游指标消融表。**

### M6：报告与交接

- [ ] `outputs/sim2real/<benchmark_id>/report.md` 定稿
- [ ] 给主项目的交接备忘：最优生成器、gap 数值、建议的预训练数据格式
- [ ] AMASS 扩产计划：用胜出 pipeline 从 AMASS 批量生成腕部单 IMU 预训练集（对接 G4/I5）
- **DoD：报告 + 备忘 + 扩产计划三件套。**

## 3. 三层评估金字塔

| 层 | 问题 | 手段 | 成本 | 位置 |
|---|---|---|---|---|
| L0 | 信号对上了吗？ | RMSE / Pearson / PSD / 峰值时序 | 秒级 | `src/evaluation/imu_csv.py`（复用），`gate.py` 包装为门禁 |
| L1 | 分布像吗？ | 冻结 encoder 嵌入 + Fréchet / MMD / C2ST-AUC | 分钟级 | `embed.py` + `dist_metrics.py` |
| L2 | 下游有用吗？ | 检索 probe + TSTR 协议族 | 单格分钟~几十分钟 | `probe/` + `protocols.py` |

L0 是门禁：防止"坐标系/标定/对齐坏了"被误诊为"sim-to-real gap 大"。L1 是快速迭代档。L2 是最终裁决。

## 4. 下游任务与协议

**任务：** 窗口级 IMU→motion 检索（给一个 IMU 窗口，从 gallery 中检索配对的骨架运动窗口），指标 R@1 / R@5 + 每序列分解。选它因为：(a) 就是主项目 FrameAcc 的底层能力；(b) TotalCapture 平行语料免费提供正样本对；(c) 训练规模小，撑得起矩阵。

**协议（`protocols.py`）：**

| 协议 | 训练 | 测试 | 含义 |
|---|---|---|---|
| `trtr` | 真实 | 真实 | 上界参考 + 健全性锚点 |
| `tstr` | 合成 | 真实 | 主排名；gap = TRTR − TSTR |
| `mix` | 真实+合成 | 真实 | 主项目 I4（增强）对应物 |
| `pretrain_finetune` | 合成预训练 → 真实(比例)微调 | 真实 | 主项目 I5 对应物；扫 real_fraction 得 sim-boost 曲线 |

**纪律（写进代码不靠自觉）：** 超参只准对 val（S4）调；test（S5）只在最终评估触碰。val/test 分片永远只有真实数据。

## 5. 数据契约

### 层 A：序列级平行语料（M1 产出）

```
data/interim/sim2real/corpus/totalcapture/<S1_freestyle3>/
├── motion.npz                        # MotionSequence: joints[T,J,3], fps, joint_layout
├── imu/
│   ├── real.npz                      # ImuSequence, source='real'
│   └── synth_<gen>_<cfg8>.npz        # 每 (生成器×配置) 一个（token 命名，配套 .manifest.json）
└── meta.json                         # subject/action/take、源文件哈希、对齐偏移、门禁结果
```

### 层 B：窗口级 benchmark（M2 产出）

```
data/interim/sim2real/windows/tc_rlowarm_w24_v1/
├── spec.json            # 窗口/步长/通道/归一化统计量/切分哈希/语料 config_hash 清单
├── train__real.npz      # 每 (split, source) 一个分片：<split>__<source_token>.npz
├── train__synth_globalpose_<cfg8>.npz
├── val__real.npz        # val/test 永远只有 real
└── test__real.npz
```

**两条铁律（由 `windows.py`/`splits.py` 强制，不靠约定）：**
1. **切分按受试者隔离且先于生成冻结。** 检索任务里动作内容就是半个标签；按受试者切一刀消除序列级与 take 级泄漏。冻结文件进 git，改动=新建 `_v2` 文件。物化结束自动跑 `find_leakage` 断言。
2. **归一化统计量只从训练源计算，并原样施加到 val/test**（TSTR 的诚实部署假设）。混用各自统计量测的是归一化差异而非生成质量——TSTR 基准最经典的隐形杀手。统计量记录于 `spec.json`。

## 6. 生成器契约（文件级，环境解耦）

**约束：** HuMoGen_origin / naive_kinematics 的合成核心跑在外部 venv（subprocess），GlobalPose_origin 跑在仓库环境，评估侧还需要 torch 环境——所以生成器接口**必须是文件级契约**，不做进程内统一框架。

统一 CLI（`scripts/sim2real/generators/<name>/generate.py`）：

```
generate.py --motion <motion.npz> --sensor R_LowArm --config <cfg.json> --seed N --out <dir>
```

产出 `imu.npz`（ImuSequence，source = `synth/<gen>/<cfg8>`）+ `manifest.json`（解析后配置、config_hash、输入哈希、seed、git sha）。适配器内部实现自由；现有三个 `run_pipeline.py` 保留为单序列 demo/调试入口。

**配置身份：** `config_hash = sha1(canonical_json(config))[:8]`（`contracts.config_hash`）。GlobalPose 的 5 个真实感开关（安装误差 RBS / 位置随机游走 / 姿态随机游走 / 传感器噪声 / 含噪积分姿态）由配置的 `switches` 块声明，M5 消融免费获得。

## 7. 实验矩阵与结果账本

- 矩阵定义：`configs/sim2real/matrix_v1.yaml`（cell = 协议 × 训练组成 × seed；TBD 哈希在 M1 后回填）。
- 账本：`outputs/sim2real/<benchmark_id>/results.jsonl`，每格一行 JSON（协议、来源、config_hash、seed、R@1/R@5、随机线、耗时、artifact 路径、git sha）。**追加式；聚合只读账本；账本进 git。**
- 续跑：`06_run_matrix.py` 启动时跳过账本已有格子——共享 GPU 上被杀是常态，断点续跑是一等公民。
- seeds 固定 `[0, 42, 123]`；一切结论报 mean±std（本领域 seed 方差足以淹没结论，主项目 G3/E2 已验证过）。

## 8. 健全性锚点（内建于协议）

1. **TRTR 必须显著强**（随 R@1 一并报出 1/gallery 随机线）。弱则测试床坏（对齐/窗口/probe），所有 TSTR 作废。矩阵第一格。
2. **打乱配对对照**（`--shuffle-pairs`）：错配训练必须回落随机线，否则有泄漏（如窗口重叠跨切分边界）。每 benchmark 在 M3 跑一次。
3. **L0 门禁不是装饰**：阈值在 M1 用已知良好样本标定后写入 benchmark spec；阈值为 null 时 `02_gate_corpus.py` 拒绝运行。

## 9. 目录结构与现有资产复用

```
src/sim2real/          contracts / splits / corpus / gate / windows / embed /
                       dist_metrics / probe/{model,train,retrieve} / protocols / report
scripts/sim2real/      01_build_corpus … 07_report（编号风格对齐主项目 A1/A2 习惯）
scripts/sim2real/generators/{globalpose,humogen,naive}/generate.py
configs/sim2real/      splits/ generators/ benchmarks/ matrix_v1.yaml
data/interim/sim2real/ corpus/ windows/          （不进 git）
outputs/sim2real/      results.jsonl report.md   （账本与报告进 git）
tests/                 test_sim2real_contracts / test_no_split_leakage / (M2+) determinism
```

| 现有资产 | 用法 |
|---|---|
| `src/evaluation/imu_csv.py` | 原封不动，`gate.py` 调用（L0） |
| `scripts/totalcapture_test/prepare_triplet.py` | 参数化序列名，被 `01_build_corpus.py` 驱动 |
| 三个 `_run_pipeline_impl.py` | 被生成器适配器薄壳包住 |
| `plot_imu_comparison.py` | 复用进门禁报告 |
| `run_manifest.json` 惯例 | 升级为统一 manifest 契约（`contracts.write_manifest`） |

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| 坐标系/标定/对齐错误冒充 sim-to-real gap | L0 门禁 + M1 强制标定阈值；失败序列排除并记录原因 |
| fps 假设错误（spec 里 60Hz 是假设） | M1 实测真实流与 `mocap_frame_rate`；M2 统一重采样到 target_fps |
| 在测试受试者上调参 | S5 只许最终评估触碰，写入协议层；切分文件冻结 |
| seed 方差淹没结论 | 全部 ≥3 seeds，只看 mean±std |
| 共享 GPU 任务被杀 | results.jsonl 账本续跑 |
| probe 容量变成混淆变量 | probe 架构与训练预算全矩阵冻结 |

## 11. 刻意不做的事

- **不新建仓库**：评估与生成在迭代期强耦合（config hash / manifest / 数据契约跨侧流动），n=1 开发者跨仓库同步是纯摩擦。命名空间独立已为未来抽取留好接缝。
- **不引入 hydra / 工作流引擎 / 插件注册机制**：3 个生成器、几十个矩阵格子，argparse + json + 一个驱动脚本足够（仓库自身原则："small, explicit modules over deep abstraction"）。
- **不把三条 pipeline 重构进同一进程**：它们本就跑在不同 venv，文件契约是诚实的边界。
- **L2 出结果前不新增第 4 条生成 pipeline**：先分胜负，再铺宽度。
