# TotalCapture 当前主流程详解

这份文档详细说明仓库里当前维护中的 `TotalCapture` 主流程到底在做什么。

当前主流程只围绕一个样例工作：

- sequence: `S1_freestyle3`
- sensor: `R_LowArm`

并且强约束为：

- `raw` 必须是原始样例目录
- `processed` 必须是标准三元组
- 后续 `generation` 只允许读取 `processed` 里的 `SMPL-X`

## 一、当前目录约定

### 1. raw

目录：

- `data/raw/totalcapture/S1_freestyle3/`

这里保存的是“原始样例应该有的样子”，不是派生结果。

当前目录里包含：

- `TC_S1_freestyle3_cam1.mp4` 到 `cam8.mp4`
- `s1_freestyle3_Xsens.sensors`
- `s1_freestyle3_Xsens_AuxFields.sensors`
- `s1_freestyle3_calib_imu_bone.txt`
- `s1_freestyle3_calib_imu_ref.txt`
- `gt_skel_gbl_pos.txt`
- `gt_skel_gbl_ori.txt`

这些文件分别对应：

- 原始视频
- 原始 IMU 传感器文件
- IMU calibration 文件
- Vicon/GT 全局位置与全局姿态

### 2. processed

目录：

- `data/processed/totalcapture_test/S1_freestyle3/`

这里保存的是标准三元组，只保留后续真正需要的最小集合：

- `TC_S1_freestyle3_cam1.mp4`
- `s1_freestyle3_R_LowArm.csv`
- `s1_freestyle3_smplx.npz`

也就是：

- 原始视频
- 右手手腕完整版 IMU
- SMPL-X

### 3. outputs

目录：

- `outputs/totalcapture_test/GlobalPose_origin/`

这里保存运行结果：

- `csv/R_LowArm_raw.csv`
- `csv/R_LowArm_generated.csv`
- `metrics/R_LowArm_raw_vs_generated_metrics.json`
- `plots/R_LowArm_raw_vs_generated.png`
- `report.md`
- `run_manifest.json`

## 二、流程总览

当前主流程分成两个入口脚本：

### 入口 1：标准三元组预处理

脚本：

- `scripts/totalcapture_test/prepare_triplet.py`

作用：

1. 从外部原始数据源抽取并组装一个完整 raw 样例目录
2. 从 raw 样例中提炼出标准三元组

### 入口 2：generation + evaluation

脚本：

- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`

作用：

1. 读取标准三元组
2. 从 `processed` 里的 `SMPL-X` 生成 synthetic IMU
3. 将 `raw` 和 `generated` 做对比
4. 输出图、指标和报告

## 三、预处理流程在干什么

### Step 1：组装 raw 样例目录

脚本入口：

- `scripts/totalcapture_test/prepare_triplet.py`

核心函数：

- `src/globalpose_origin_adapter.py: stage_totalcapture_raw_sample(...)`

它做的事：

1. 从 `lxhong` 的原始压缩包里取视频
2. 从 `s1_Gyro_Mag.tar.gz` 里取 richer IMU 文件 `Xsens_AuxFields.sensors`
3. 从 `S1_imu.tar.gz` 里取：
   - `Xsens.sensors`
   - `calib_imu_bone.txt`
   - `calib_imu_ref.txt`
4. 从 `s1_vicon_pos_ori.tar.gz` 里取：
   - `gt_skel_gbl_pos.txt`
   - `gt_skel_gbl_ori.txt`

最终形成：

- `data/raw/totalcapture/S1_freestyle3/`

这个阶段的重点不是“算”，而是“收集原始样例”。

### Step 2：从 raw 提炼标准三元组

核心函数：

- `src/globalpose_origin_adapter.py: prepare_totalcapture_processed_triplet(...)`

它做的事：

1. 从 raw 目录复制 `cam1` 视频
2. 从 `Xsens_AuxFields.sensors` 里提取单传感器 `R_LowArm`
3. 生成 `s1_freestyle3_R_LowArm.csv`
4. 复制 `SMPL-X npz`

最终形成：

- `data/processed/totalcapture_test/S1_freestyle3/`

此后主流程只认这个三元组。

## 四、generation 流程在干什么

这里只讲 generation，不再讲 preprocess。

入口脚本：

- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`

真正生成逻辑在：

- `scripts/totalcapture_test/GlobalPose_origin/_run_pipeline_impl.py`

### Step 1：读取 processed 里的原始 IMU

从这里读取：

- `data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_R_LowArm.csv`

这个文件不是用来“生成”，而是用来作为后续对比基准。

在 helper 里它被读成：

- `raw_data`

包含：

- quaternion
- acceleration
- gyroscope
- magnetic field

### Step 2：读取 processed 里的 SMPL-X

从这里读取：

- `data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_smplx.npz`

这个文件是 generation 的真正输入。

它包含：

- `trans`
- `root_orient`
- `pose_body`
- `pose_hand`
- `pose_jaw`
- `pose_eye`
- `betas`
- `num_betas`
- `mocap_frame_rate`

### Step 3：做 SMPL-X 前向

helper 里直接用官方 `smplx`：

- `smplx.create(...)`

然后把 `SMPL-X` 参数送进去前向，得到：

- 每一帧的 `output.joints`

也就是说，这一步是在做：

- `SMPL-X 参数 -> 人体关节轨迹`

### Step 4：把 joints 转成当前 pipeline 约定

helper 里做了两层转换：

1. `SMPLXRunner.convert_smplx_to_pipeline_world(...)`
2. `SMPLXRunner._project_real_joints_to_pipeline_layout(...)`

作用是：

- 把 `SMPL-X` 的 joint 输出变成当前仓库统一的 joint 布局和坐标约定

### Step 5：提取右手手腕传感器轨迹

函数：

- `compute_sensor_trajectory(...)`

当前只取：

- `wrist_right`

输出：

- 手腕位置轨迹 `positions`
- 手腕姿态轨迹 `quaternions`

也就是把人体动作变成：

- 传感器的 6DoF 轨迹

### Step 6：用 GlobalPose 官方 synthesis 核心生成 IMU

当前 helper 内置了一份当前主流程自己的 GlobalPose synthesis 核心实现。

它不是 `legacy` 脚本，也不是再去调用 `legacy` 子进程。

它遵循的思路来自：

- `third-party/GlobalPose/imu_synthesis.py`

这一段做的事情包括：

1. 给传感器加安装误差 `RBS`
2. 给位置加 random walk `dp`
3. 给姿态加 random walk `dw`
4. 用 `IMUSimulator` 从 6DoF 轨迹生成：
   - acceleration
   - angular velocity
   - magnetic field
5. 再叠加噪声
6. 再从 noisy 角速度积分出姿态估计
7. 最后得到 synthetic IMU 输出

这个阶段的本质是：

- `手腕 6DoF 轨迹 -> synthetic IMU`

### Step 7：对 quaternion 做连续性修正

因为四元数存在 `q` 和 `-q` 表示同一姿态的问题，所以 helper 会做一层连续性修正：

- 若相邻帧四元数点积为负
- 则把当前帧整体乘 `-1`

这个修正只影响表示，不改变姿态本身。

### Step 8：写出 generated CSV

最终输出：

- `outputs/totalcapture_test/GlobalPose_origin/csv/R_LowArm_generated.csv`

这个文件就是当前 synthetic IMU 结果。

## 五、评测流程在干什么

脚本：

- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`

评测核心：

- `src/evaluation/imu_csv.py`

当前只做一组对比：

- `raw vs generated`

评测输入：

- `outputs/totalcapture_test/GlobalPose_origin/csv/R_LowArm_raw.csv`
- `outputs/totalcapture_test/GlobalPose_origin/csv/R_LowArm_generated.csv`

评测输出：

- `outputs/totalcapture_test/GlobalPose_origin/metrics/R_LowArm_raw_vs_generated_metrics.json`

### 指标层次

当前评测包含：

1. `motion_intensity`
   - acc magnitude RMSE
   - gyro magnitude RMSE

2. `temporal_consistency`
   - acc magnitude correlation
   - gyro magnitude correlation

3. `event_consistency`
   - peak timing error

4. `frequency_structure`
   - PSD distance

5. `window_statistics`
   - window feature distance

此外还会给：

- `real_vs_synthetic`

这里的命名沿用了评测模块原有结构，但在当前主流程里应理解为：

- `raw` vs `generated`

## 六、画图流程在干什么

脚本：

- `scripts/totalcapture_test/plot_imu_comparison.py`

helper：

- `scripts/totalcapture_test/_plot_imu_comparison.py`

当前画的是：

- `raw`
- `generated`

虽然参数名仍然是 `real` / `synthetic`，但在当前主流程里实际传进去的是：

- `real-csv = raw.csv`
- `synthetic-csv = generated.csv`

输出：

- `outputs/totalcapture_test/GlobalPose_origin/plots/R_LowArm_raw_vs_generated.png`

## 七、report 在干什么

脚本仍然是：

- `scripts/totalcapture_test/GlobalPose_origin/run_pipeline.py`

它会把：

- raw 样例目录
- processed 三元组目录
- SMPL-X 输入
- 指标摘要

整理成：

- `outputs/totalcapture_test/GlobalPose_origin/report.md`

## 八、当前主流程的边界

当前主流程的关键边界是：

### 允许依赖

- `data/raw/totalcapture/S1_freestyle3/`
- `data/processed/totalcapture_test/S1_freestyle3/`
- `third-party/GlobalPose/` 中的官方算法参考与 `IMUSimulator`

### 不允许依赖

- `legacy` 目录中的脚本实现
- `data/interim`
- 已删除的 `.pt benchmark` 包
- 其它临时样例目录

### generation 唯一合法输入

- `data/processed/totalcapture_test/S1_freestyle3/s1_freestyle3_smplx.npz`

## 九、当前主流程的最短理解

可以把整个主流程压缩成下面这条链：

1. 外部原始数据 -> 组装 raw 样例目录
2. raw 样例目录 -> 提炼标准三元组
3. 标准三元组里的 `SMPL-X` -> synthetic IMU
4. 标准三元组里的 `R_LowArm IMU` -> raw 基准
5. raw vs generated -> plot + metrics + report

如果只记一句话：

**当前主流程是：`raw -> processed triplet -> generated`，其中 generation 只读 `processed` 里的 `SMPL-X`。**
