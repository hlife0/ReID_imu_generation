# sim2real 基准 v1 —— 发现与交接备忘（2026-07-05）

> 数据出处：`outputs/sim2real/tc_rlowarm_w24_v1/`（`results.jsonl` 为唯一事实源，
> `report.md` 为自动聚合，本文是解读）。测试床：TotalCapture，IMU→骨架窗口检索，
> 24 帧窗 / stride 16 / 单 R_LowArm / 按受试者切分（S1–S3 训 / S4 验 / S5 测），
> 测试 gallery 1005 窗，随机线 R@1 = 0.10%。3 seeds 报 mean±std。

## 一句话结论

**在"下游任务说了算"的评估下，最朴素的 naive 运动学基线以约 10 倍优势胜过
GlobalPose 全真实感管线与 HuMoGen；真实+naive 合成混合训练比纯真实训练高出
约 37%（相对）——合成 IMU 的价值瓶颈不在噪声真实感，而在输出信号的语义约定。**

## 核心数字

| 设置 | R@1 | 说明 |
|---|---|---|
| TRTR（真实训练上界） | 0.0395 ± 0.0066 | 随机线的 40 倍 |
| TSTR naive | **0.0226 ± 0.0033** | 达 TRTR 的 57%，最佳生成器 |
| TSTR humogen | 0.0023 ± 0.0005 | ≈ 随机线 2 倍 |
| TSTR globalpose（全噪声/clean） | 0.0013 ± 0.0007 | ≈ 随机线 |
| **mix：真实 + naive** | **0.0541 ± 0.0045** | **超过纯真实 +0.0146（+37% 相对）** |
| mix：真实 + globalpose | 0.0385 ± 0.0061 | 无增益 |
| 打乱配对对照 | 0.0010 | = 随机线，测试床无泄漏 ✅ |

## 四个发现

**F1：TSTR 排名与 L1 特征分布排名完全反转。**
L1（Fréchet/MMD，stats_v1 编码器）认为 globalpose 最接近真实分布（Fréchet 448 vs
naive 1138 vs humogen 1485），但下游迁移 naive 独占鳌头。**"分布像"≠"下游有用"；
用 L1 指标筛选生成器的捷径在本任务上被证伪**——迭代生成器必须看 L2。

**F2：GlobalPose 迁移失败的原因不是噪声堆栈，而是输出帧语义。**
clean（全部真实感模块关闭）与 noise_full 的 TSTR 同样 ≈ 随机线 ⇒ 罪魁不是噪声。
GlobalPose 输出的是"估计坐标系下的自由加速度 + 重力回加"（`a_model = R_est·a_sensor + g`），
而真实 Xsens 读数是传感器系比力；naive 恰好输出传感器系比力——L0 门禁早有伏笔
（naive 的 acc 幅值相关 0.975 vs globalpose 0.55~0.63）。
**可操作假设（下一步最高优先）：给 globalpose 增加"传感器系比力输出"的配置开关，
预期其 TSTR 大幅回升；届时噪声模块的消融（M5 完整版）才真正有意义。**

**F3：humogen 的加速度被重力轴约定破坏。**
其 origin 实现默认重力 [0,0,-9.81]（Z-up），但 pipeline 世界是 Y-up；acc 幅值相关
仅 0.30，10/16 条门禁弱通道都是它。修一行重力向量即可（保留 origin 行为的对照价值，
建议新增 `humogen_yup.json` 配置而非改默认）。

**F4：sim-boost（合成预训练）在低真实数据段有信号但不稳。**
10% 真实数据时 naive 预训练把 R@1 从 0.0087 提到 0.0169（约翻倍），但 25% 段为负。
mix（增强式，对应主项目 I4）目前比 pretrain→finetune（I5）更可靠。

## 给主项目（G4）的交接建议

1. **I4（增强）优先于 I5（预训练）**：本基准里"真实+合成混合训练"是唯一稳定为正的
   用法。给 custom 训练集混入 naive 风格合成腕部 IMU 是最先值得试的。
2. 合成数据的**信号语义对齐（帧约定、比力 vs 自由加速度、重力轴）比噪声真实感重要
   一个数量级**——把这条设计约束带进任何合成数据的使用。
3. 本基准窗口约定（w24/s16、单 R_LowArm）与 custom 场景一致，结论可直接类比。

## AMASS 扩产计划（草案）

- 生成器：当前胜者 naive（等 F2 的 globalpose 比力变体验证后再定终选）。
- 源：AMASS 全库（TotalCapture 之外的几十个子集），SMPL-X neutral，60Hz 重采样。
- 产物：`(motion.npz, synth imu.npz)` 平行语料，直接走本仓库 layer-A 契约；
  交付主项目侧的格式在对接口后再定（他们 cache 层吃 7 维/24 帧窗）。
- 复用 `01_build_corpus.py` 的骨架：把 sequence 枚举器换成 AMASS 目录遍历即可。

## 已知局限（下轮迭代）

- 绝对 R@1 偏低（0.4 秒窗、跨受试者、千级 gallery 本来就难）；排名间隔远大于
  seed 方差，结论稳，但若想要更高的绝对数，可加 w100 变体 benchmark（对应主项目
  的 100 帧设置）。
- L1 的 stats_v1 编码器与下游排名脱节（F1）；M3 之后可改用 probe 的 IMU 塔做 L1
  编码器再检验一次相关性。
- sim-boost 25% 段的负值需要更多 seeds 或更细的 fraction 网格确认。
