# sim2pipe 发现 v1（pipe_probe_tc_v1）

> 数据：`outputs/sim2pipe/pipe_probe_tc_v1/{results.jsonl,report.md}`（21 格，3 seeds）。
> 测试床：TotalCapture，冻结切分 `totalcapture_subject_v1`（S1–S3 训 / S4 验 / S5 测），
> 估计骨架（estskel）motion 侧，传感器 R_LowArm，窗口 w24。
> 裁判：主项目 `IMUVideoMatcher`（冻结 MotionBERT + 冻结 DeSPITE + 训练对齐头，SymmetricInfoNCE），
> 经 subprocess 在主项目 autism_test 环境跑其自带 `slice / train / eval`。指标 = 真实测试窗口的
> batch 内检索 top1（batch 64，理论随机线 0.0156）。

## 头版结论

**probe 的排名没有在真实 pipeline 幸存，且合成数据在真实模型下当前无迁移价值。**

| 协议 | 流 | test_top1（S5 真实） |
|---|---|---|
| TRTR | real | **0.2518 ± 0.0055** |
| TSTR | globalpose_clean | 0.0098 ± 0.0020 |
| TSTR | humogen | 0.0128 ± 0.0016 |
| TSTR | naive | 0.0127 ± 0.0020 |
| mix | real+globalpose_clean | 0.2601 ± 0.0010 |
| mix | real+humogen | 0.2374 ± 0.0102 |
| mix | real+naive | 0.2406 ± 0.0136 |

三条主线：

1. **TRTR 强、锚点健康**：0.2518，约 16× 理论随机线，远高于冻结编码器的零样本地板
   （打乱对照把 val 压在 ~0.11 不涨，见下）。测试床成立。
2. **三个生成器 TSTR 全部塌到随机线附近**（0.010–0.013，彼此差 0.003 < 随机线 0.0156，
   统计上不可区分）。这与 sim2real probe 完全相反——那里 naive TSTR R@1=0.0172（17× chance）
   明显胜出、humogen/globalpose 仅 2× chance。**真实模型下三者都失败，naive 的优势是 probe 的假象。**
3. **mix 无增益**：real+synth ≈ TRTR（naive/humogen 甚至略降，globalpose 持平在噪声内）。
   sim2real probe 里 mix real+naive 相对 TRTR +141%——**这个增益也没幸存。**

## 为什么 probe 高估：量具不同

sim2real probe 从零训练自己的小 IMU 塔，学的是"这批合成信号内部的可分性"；只要合成流自洽，
probe R@1 就非零，于是给出乐观排名。sim2pipe 的 IMU 塔是**冻结的预训练 DeSPITE**（在真实 IMU 统计上训得），
只训一个薄对齐头：
- TSTR 用合成流训头 + 用**合成源的归一化统计量**施加到真实测试流（诚实 TSTR 部署假设）。合成与真实
  的帧语义/尺度一旦不匹配，真实测试 IMU 经合成统计归一化后落到 DeSPITE 没见过的区域 → **低于随机**。
- 这不是 bug，是**真实 pipeline 层面的 sim-to-real gap 被如实测出**：当前三个生成器产出的信号，
  对一个在真实 IMU 上预训练的编码器都不可用。probe 因为自训编码器绕过了这道坎，才显得乐观。

## 健全性锚点（N1 门禁）

- **格式往返** ✅：导出 npz 经主项目 `slice` 无警告物化，37 序列 9207 窗口。
- **真实锚点** ✅：TRTR 0.25 ≫ 随机线，≫ 零样本地板。
- **打乱对照** ✅：`--shuffle_video_in_batch` 训练下 val_top1 被压在 ~0.10–0.11 不随 epoch 上升
  （正常训练 train_top1 持续爬升）——确认这个测试床的**有效地板是冻结预训练编码器的零样本对齐 ~0.11**，
  而非理论 1/64。

## 对主项目 G4 的直接含义

- **I4（数据增强）/ I5（预训练）用当前三个生成器都不会奏效**：pipe 层面 TSTR 崩、mix 不涨。
  在把合成数据接进 custom 单 IMU 目标之前，生成器必须先跨过"匹配真实 IMU 统计/帧语义"这道坎。
- **首要修复方向 = F2（globalpose 比力输出变体）**：DeSPITE 期望比力（含重力）语义；naive 输出自由加速度、
  humogen 缺重力对齐。F2 出来后**优先进 sim2pipe 复跑**，这是唯一能翻盘 TSTR 的已知杠杆。
- 次要：TSTR 归一化鲁棒性（用真实源统计而非合成源统计做消融，隔离"归一化域移"与"信号语义差"两个因素）。

## 局限

- 单一测试床（TotalCapture）、单窗口（w24）、单指标（batch 内 top1）。绝对数受 batch 大小影响。
- estskel 退化参数仍是文献默认值（继承 sim2real 局限），未用真实 AlphaPose 标定。
- P2（合成预训练 → custom 微调 → FrameAcc）本期未做，另行立项；但 P1 已强烈提示 P2 用当前生成器不会有正收益。

## 归一化消融：拆开"域移"与"信号语义"（2026-07-07）

**问题**：TSTR 崩到随机线以下，是因为 (a) 用合成源统计量归一化、真实测试 IMU 被推到 DeSPITE 分布外
（**归一化域移**），还是 (b) 合成信号的帧语义/时序动态本身和真实不同（**硬 sim-to-real gap**）？

先看一阶统计量（真实 S1–S3 vs 各生成器 S1–S3，48 维里加速度槽）：真实加速度均值 `[2.59, -0.70, 2.07]`
带明显重力 DC 分量；naive `[-6.89,…]`（重力反向）、globalpose_clean `[0,0,0]`（零均值/纯自由加速度、无重力）、
humogen `[-3.06,…]`。三者均值向量与真实的方向余弦全不为正（naive/humogen 甚至反向 -0.76/-0.64）——
一阶统计量严重错位。但这把两个因素纠缠在一起（重力语义差本身就表现为均值差）。

**消融**（`outputs/sim2pipe_normabl/`，9 格 TSTR × 3 seeds）：用 **oracle 真实统计量**（TRTR 的 S1–S3 真实
`imu_stats.json`）归一化 train+test，其余不变。测试侧真实 IMU 因此和 TRTR 完全同样处理——若合成训练的
对齐头能对齐真实 IMU 嵌入，就该得分。

| 流 | synth-stats（baseline） | real-stats（oracle） | Δ |
|---|---|---|---|
| naive | 0.0127 ± 0.0020 | 0.0098 ± 0.0020 | −0.0029 |
| humogen | 0.0128 ± 0.0016 | 0.0122 ± 0.0011 | −0.0006 |
| globalpose_clean | 0.0098 ± 0.0020 | 0.0105 ± 0.0024 | +0.0006 |

**结论：三个生成器用完美真实统计量归一化后 TSTR 全都没恢复**（Δ 全在噪声内，仍钉在随机线）。
对照 mix 在 real-stats 下仍 0.2455（真实统计量本身没破坏管线）。**所以崩溃的主因不是归一化域移，
是信号语义/时序动态的硬 gap**——冻结的 DeSPITE 吃合成 IMU（哪怕归一化到真实统计量）产出的嵌入，
其"对齐到 motion"的映射对真实 IMU 嵌入结构无效。统计量对齐是必要但远不充分，甚至不是瓶颈。

**强化 F2 建议**：修复必须在**信号生成层面**（比力/重力/时序语义），而非归一化。这正是 F2（globalpose
比力输出变体）要解决的。归一化鲁棒性不是杠杆。
