# 交接备忘：合成 IMU 在主项目 pipeline 下的评测结论（sim2pipe → G4）

> 收件人：justlanxuan（Rd-id-Project / G4 负责人）
> 发件人：hrli（ReID_imu_generation / 合成 IMU 生成）
> 日期：2026-07-07
> 依据：`ReID_imu_generation/docs/sim2pipe_{design,findings}_v1.md`、账本 `outputs/sim2pipe*/…/results.jsonl`

## 一句话

**我用你主项目自己的模型（IMUVideoMatcher）而不是我自建的 probe，评测了三条合成 IMU pipeline
（naive / globalpose / humogen）。结论：当前三个生成器产出的合成 IMU，在你的真实模型下下游迁移全部失败，
且这不是归一化能修的——是信号语义（比力/重力/时序）层面的硬 sim-to-real gap。因此 G4 的 I4（数据增强）
与 I5（合成预训练）用现有生成器接进来不会有正收益，先别投入。**

## 我做了什么（一句话可信性）

- 测试床 TotalCapture，冻结 subject 切分（S1–S3 训 / S4 验 / S5 测），传感器 R_LowArm、窗口 w24，
  motion 侧用**估计骨架**（模拟你真实 pipeline 的 `视频→AlphaPose→MotionBERT→H36M17`，不吃像素）。
- 把每条动作喂给各生成器得到"同一动作、真假多版 IMU"的平行语料，导出成你主项目吃的 48 维 npz 格式，
  **直接 subprocess 调你自己的 `src.data.slice.totalcapture` / `src.engine.train` / `src.engine.eval`**
  （在你的 autism_test 环境里跑，我这边零改动你仓库）。
- 指标 = 真实测试窗口的 batch 内检索 top1（就是你 `retrieval_top1` 那套，也是 sim2real R@1 的同族）。
  注意这是**检索 top1，不是 FrameAcc**；FrameAcc 要到 P2（见下）。

健全性都过了：TRTR（真训真测）0.2518 是健康上界（约 16× 随机线）；打乱配对对照被压在
"冻结编码器零样本地板"~0.11 不涨；格式往返无警告。

## 关键数字（test_top1，S5 真实，3 seeds mean±std）

| 协议 | 训练数据 | test_top1 | 读法 |
|---|---|---|---|
| TRTR | 真实 | **0.2518 ± 0.006** | 上界锚点，测试床成立 |
| TSTR | naive 合成 | 0.0127 ± 0.002 | 塌到随机线 |
| TSTR | globalpose 合成 | 0.0098 ± 0.002 | 塌 |
| TSTR | humogen 合成 | 0.0128 ± 0.002 | 塌 |
| mix | 真实 + naive | 0.2406 ± 0.014 | 相对 TRTR **无增益（略降）** |
| mix | 真实 + globalpose | 0.2601 ± 0.001 | 噪声内持平 |
| mix | 真实 + humogen | 0.2374 ± 0.010 | 略降 |

两点直接对你的假设：
- **I5（合成预训练→迁移）**：TSTR 三个生成器全塌到随机线（彼此差 0.003 < 随机线 0.0156，不可区分）。
  纯合成训练的模型在真实 IMU 上无检索能力。
- **I4（合成做增强）**：mix（真实+合成）≈ 纯真实 TRTR，合成没带来增益，个别还略降。

## 为什么 —— 我拆开了主因（归一化消融）

TSTR 崩到随机线以下，先验有两个嫌疑：合成源统计量归一化把真实测试推到分布外（**归一化域移**），
或合成信号帧语义本身不对（**硬 gap**）。我用 **oracle 真实统计量**（TRTR 的 S1–S3 真实 imu_stats）
重跑 9 格 TSTR，测试侧真实 IMU 因此和 TRTR 完全同处理：

| 流 | 合成统计量 | 真实统计量(oracle) | Δ |
|---|---|---|---|
| naive | 0.0127 | 0.0098 | −0.003 |
| globalpose | 0.0098 | 0.0105 | +0.001 |
| humogen | 0.0128 | 0.0122 | −0.001 |

**用完美真实统计量归一化也救不回来**（Δ 全在噪声内）。所以是信号语义/时序动态的硬 gap，不是归一化。
一阶统计量旁证：真实加速度均值 `[2.59, -0.70, 2.07]` 带重力 DC 分量，而 globalpose_clean 是 `[0,0,0]`
（纯自由加速度、无重力）、naive 重力反向——这些是**你的冻结 DeSPITE（在真实 IMU 上预训练）根本没见过的信号**。

## 建议

1. **现有生成器别接进 G4 的 I4/I5**——pipeline 层面已证无收益，会浪费 seed 预算。
2. **首要且唯一已知的杠杆是生成器信号语义修复**，即我这边的 **F2（globalpose 比力输出变体，
   让合成 IMU 输出含重力的比力而非自由加速度，匹配 DeSPITE 的预训练分布）**。F2 出来后我会优先复跑 sim2pipe，
   翻盘了再找你接 I4/I5。
3. **需要你配合的两件事（不急）**：
   - **estskel 标定**：我的估计骨架退化参数目前是文献默认值，未用真实 AlphaPose 数据标定。若你能给一份
     custom（或 TotalCapture）的 AlphaPose→MotionBERT 提取骨架样本，我能重标定 `estimated_default.json`，
     让 P1 的骨架侧更贴近你真实 pipeline。
   - **P2 流程对齐**：P2 = 用胜出生成器合成预训练 → 你的 `--init_alignment_ckpt` 接 custom per-video 7:3
     微调 → 出 FrameAcc（6 seeds）。这才是能直接进你 SOTA 表的数字，但依赖 custom 数据链路和你的实验纪律
     （SOTA 表更新规则、experiments 目录归属）。**本期我没做 P2**，等 F2 让 P1 先翻盘、再和你对 P2 流程。

## 可复现入口 & 局限

- 驱动：`ReID_imu_generation/scripts/sim2pipe/03_run_pipe_probe.py`（读 `configs/sim2pipe/matrix_pipe_v1.yaml`，
  账本续跑）；报告：`05_report.py`；机器路径在 `configs/sim2pipe/paths.yaml`（指向你的仓库、autism_test env、
  MotionBERT/despite ckpt）。
- 局限：单测试床（TotalCapture）、单窗口（w24）、指标是检索 top1 非 FrameAcc（绝对数受 batch 大小影响，
  排名/增益的定性结论稳健）；estskel 未真实标定。这些都不影响"现有生成器无迁移价值 + 主因是信号语义"这两个定性结论。
