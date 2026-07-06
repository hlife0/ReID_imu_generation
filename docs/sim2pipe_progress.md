# sim2pipe 施工进度

> 设计依据：`docs/sim2pipe_design.md`（v1，2026-07-06 定稿）。本文件按里程碑记录执行结果。

## N0：骨架 + convert/export 契约 + paths + 测试（✅ 2026-07-06）

- 主项目契约核实（R1 关闭）：quat **wxyz**（`Rd-id-Project/src/datasets/totalcapture.py::quat_to_rotmat`）；
  48 维布局 = 先 4 槽 9 维旋转矩阵（0:36）再 4 槽 3 维 acc（36:48），槽序
  [L_LowLeg, R_LowLeg, L_LowArm, R_LowArm]，单传感器复制 ×4；骨架归一化 =
  root-relative + 除以 ||joint8−joint0||；TotalCapture 路径 60Hz 1:1 消费**无重采样**
  （R3 只影响已移出范围的 P2）；合成消费口 = 复制 unified-schema npz。
  主项目 adapter 默认 subject 切分 S1-S3/S4/S5 与冻结切分 `totalcapture_subject_v1` 一致。
- 落地：`src/sim2pipe/{convert,export,overlay,bridge,ledger}.py`、
  `scripts/sim2pipe/{01_export_corpus,02_gate_export}.py`（03/05 为 CLI 契约 stub）、
  `configs/sim2pipe/{paths.yaml.example,matrix_pipe_v1.yaml,overlays/pipe_probe_tc.yaml}`。
- 测试：`test_sim2pipe_convert.py` + `test_sim2pipe_export_contract.py` 20 项全绿；全套 63 项无回归。
- 门禁依赖检查 PASS：主仓库、主项目环境（**autism_test**，即 environment.yml 的 test_reid；
  torch 2.1+cu118 CUDA ✅）、MotionBERT/despite ckpt（/home/fzliang 下均在本机可达）、语料。
- 冒烟：real 流全语料导出 37 序列 / 147,901 帧，导出 npz 在 autism_test 环境
  （numpy 1.26）加载正常（注意：本仓库导出用 data_generation venv 的 numpy 2.x
  写 object 数组，1.26 可读，但系统 python3 的 numpy 1.x 旧版读不了——无关紧要）。

## N1：P0 门禁三件套（✅ 2026-07-06）

- [x] 格式往返：导出物经主项目 `src.data.slice.totalcapture` 物化，37 序列 9207 窗口，无警告
- [x] 真实流锚点：TRTR real test_top1 0.2518（≫ 随机线 0.0156，≫ 零样本地板 ~0.11）
- [x] 打乱对照：`--shuffle_video_in_batch` 下 val_top1 压在 ~0.10–0.11 不随 epoch 上升
- [x] overlay 模板 `pipe_probe_tc.yaml` 验为实配
- 修正：eval.py 不接受 `--imu_ckpt`（从 checkpoint 加载），已从驱动 eval 分支删除
- 发现：本测试床有效地板 = 冻结预训练 DeSPITE+MotionBERT 的零样本对齐 ~0.11（非 1/64）

## N2：P1 pipe-probe 矩阵（✅ 2026-07-06）

- 21 格（7 流 × 3 seeds），三卡（cuda:1/2/3）并行，每格约 12 分钟 @40 epoch
- 驱动 `03_run_pipe_probe.py`：compose→slice→train→eval 全走主项目模块（subprocess，autism_test 环境）
- 账本续跑生效（key = protocol×stream×motion_source×seed）
- 结果见 `docs/sim2pipe_findings_v1.md` 与 `outputs/sim2pipe/pipe_probe_tc_v1/{results.jsonl,report.md}`

## N3：报告与交付（✅ 2026-07-06）

- `05_report.py` 自动聚合 + probe/pipe 排名一致性判定，产出 `report.md`
- **头版：probe 排名未幸存**——三生成器 TSTR 全塌到随机线附近（spread 0.003 < 随机线），
  mix 无增益；probe 的 naive 优势与 +141% mix 增益都是量具假象
- 交付含义写入 findings：I4/I5 用当前生成器不奏效，首要杠杆 = F2 比力变体（优先复跑 sim2pipe）
