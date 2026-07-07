# sim2real 估计骨架变体 —— 发现与对比(2026-07-06)

> 数据出处:`outputs/sim2real/tc_rlowarm_w24_estskel_v1/`(`results.jsonl` 为唯一事实源)。
> 与主基准 `tc_rlowarm_w24_v1` 的**唯一区别**:motion 侧从 21 关节 SMPL-X 真值骨架
> 换成 17 关节 H36M **估计骨架**(`motion_estimated`,退化模块 `src/sim2real/gen_estimated.py`,
> 参数 `configs/sim2real/generators/estimated_default.json`)。IMU 流、窗口、切分、门禁全部一致。

## 为什么做这个变体

主项目真实 ReID pipeline 是 `视频 → ByteTrack+AlphaPose → MotionBERT → 17关节 H36M 三维骨架`——
它匹配的是**估计骨架**,不是像素、也不是干净真值。主基准用的干净 21 关节真值骨架
可能高估了合成 IMU 的价值。本变体把 motion 侧退化成"像真实姿态估计那样"
(重定向 21→17 + 逐关节抖动 + 时序平滑 + 遮挡丢帧,文献默认参数),回答:

> **当骨架侧从完美真值降级为带估计噪声的 17 关节时,"naive 独赢 TSTR、real+naive mix 大幅增益"这两个结论是否仍然成立?**

## 一句话结论

**成立,而且更强。** 估计骨架退化后,naive 仍以约 9 倍优势独占 TSTR 榜首,
real+naive 混合训练相对 TRTR 的增益从 +37% 升到 **+44%**;打乱对照仍等于随机线,
退化没有引入泄漏。**对主项目而言这是好消息:合成 IMU(naive)的价值不是"骨架太干净"
制造的假象,在贴近真实 pipeline 的估计骨架下依然稳固。**

## 并排对比(R@1,3 seeds,mean±std)

| 指标 | GT 骨架(21关节) | 估计骨架(17关节) | 变化 |
|---|---|---|---|
| TRTR(上界) | 0.0395 ± 0.0066 | 0.0362 ± 0.0066 | 轻微下降(骨架更噪,预期) |
| **TSTR naive** | **0.0226 ± 0.0033** | **0.0172 ± 0.0039** | 仍第一,仍约 9× 弱生成器 |
| TSTR humogen | 0.0023 | 0.0020 | ≈随机线 |
| TSTR globalpose(clean, 42c7b0f2) | 0.0013 | 0.0013 | ≈随机线 |
| TSTR globalpose(noise_full, 8026be9d) | 0.0013 | 0.0010 | ≈随机线 |
| **mix real+naive** | **0.0541 ± 0.0045** | **0.0521 ± 0.0047** | **+44% 相对 TRTR(GT 为 +37%)** |
| mix real+globalpose | 0.0385(无增益) | 0.0458(+27%) | **转为正增益(见下)** |
| 打乱配对对照 | 0.0010 | 0.0010 | =随机线,无泄漏 ✅ |

> ⚠️ 更正(2026-07-07 第二轮审查):上表 globalpose 两行的 clean/noise 标签原先写反。
> 用 `config_hash` 复算的真值是 **42c7b0f2=clean、8026be9d=noise_full**(与
> `configs/sim2pipe/matrix_pipe_v1.yaml` 一致;`configs/sim2real/matrix_v1.yaml` 与
> `matrix_estskel_v1.yaml` 的旧头注释是错误来源,已同步更正)。两行数值均≈随机线,
> E1/E2 及"clean≈noise_full ⇒ 噪声无关"的结论不受影响;E3 的 mix real+globalpose
> 用的是 clean 流。

## 三个发现

**E1:generator 排名对骨架退化稳健。** naive ≫ humogen ≈ globalpose 的格局完全保持,
naive 与弱生成器之间仍有约一个数量级的差距,远大于 seed 方差。用下游任务筛生成器的
结论不依赖"骨架是否完美"。

**E2:mix 增益不降反升(相对)。** real+naive 的绝对分从 0.0541 微降到 0.0521,
但因为 TRTR 上界也降了(0.0395→0.0362),**相对增益反而从 +37% 升到 +44%**。
解释:骨架侧变噪后,真实训练更容易过拟合到估计噪声,而掺入 naive 合成 IMU 的
增强/正则作用相对更值钱。这进一步支持主项目的 I4(增强式)优先策略。

**E3:估计骨架下 globalpose 增强开始有正增益(新现象)。** GT 骨架时 real+globalpose
无增益(0.0385≈TRTR),估计骨架时升到 0.0458(+27%)。此处 globalpose 指 **clean 流
(42c7b0f2,全部真实感开关关闭)**,非 noise_full。可能是骨架噪声"抹平"了
globalpose 自由加速度语义与真实比力之间的部分差异,使其作为增强样本不再纯是矛盾信号。
**这是一个值得在 F2(globalpose 比力输出变体)里进一步验证的线索**,但注意其方差较大
(±0.0122),需更多 seeds 确认。

## ⚠️ 强化（2026-07-07 数据翻倍对照）：估计骨架下 mix 增益是真信号，非数据翻倍

原文 E2 把 real+naive 相对 TRTR 的 +44% 归给"naive 合成 IMU 的增强/正则价值"。为隔离
"仅仅数据翻倍（mix 有 2N 窗口 vs TRTR 的 N）"这一混杂，补一个**同尺寸零新信息**对照
（`mix --train-sources real real`，真实数据复制一份 → 14706 窗，3 seeds）：

| 训练组成 | 窗口数 | R@1 | 相对 TRTR(0.0362) |
|---|---|---|---|
| **mix real+REAL（翻倍、无新信息）** | 14706 | **0.0305 ± 0.0061** | **−16%（翻倍反而伤）** |
| mix real+naive | 14706 | 0.0521 ± 0.0047 | +44% |
| mix real+globalpose | 14706 | 0.0458 ± 0.0122 | +27% |

**结论：估计骨架下复制真实数据不但不涨、反而降 16%**（真实训练更易过拟合估计噪声，
重复样本放大之），**而掺入 naive/globalpose 合成 IMU 却带来 +44%/+27%——所以这些增益
是合成信号的真实贡献，不是数据翻倍的副产品。** 这与 v1（干净真值骨架）基准相反：那里
real+real 给 +40%、naive 无额外贡献（详见 `sim2real_findings_v1.md` 更正节）。**故 estskel
才是可信的 mix 增益基准；E2 的 +44% 结论幸存并被此对照坐实。**

**稳健性与诚实注脚**：naive 的优势是逐-seed 的、非单点——real+naive 三 seed（0.0587/
0.0498/0.0478）全部高于 TRTR（≤0.0408）与 real+real（≤0.0378），区间不重叠。另需排除一层
混杂：real+real 是**逐字复制**，InfoNCE 里同一窗口与其副本同批会产生相同嵌入的"负样本"
冲突惩罚，real+naive/real+globalpose 无此惩罚——若 naive 优势仅来自"避开冲突"，则同样无
冲突的 real+globalpose 应有同等优势；但 globalpose 未复制（0.031–0.061，与 real+real 重叠），
故"避开冲突"不足以解释，naive 确实加了信号。**局限**：更干净的同尺寸控制应是 real+抖动real
（无新信息但无精确重复），尚未跑；n=3 偏薄，作为 G4 的 I4 依据定稿前建议加到 6 seeds 复核。

## 已知局限

- **退化参数是文献默认值,未用真实 AlphaPose 数据标定**——结论的绝对数不可尽信,
  但"排名/增益是否幸存"这一定性问题对参数不敏感(E1/E2 的差距远大于噪声)。
  真实 custom AlphaPose 提取骨架到位后应重新标定 `estimated_default.json` 并复跑。
- 21→17 重定向丢弃了 hand/foot 关节(与真实 pipeline 一致),thorax 以 spine2 近似。
- sim-boost(预训练)在估计骨架下仍不稳/偏负,与 GT 骨架结论一致,无新信息。

## 复现

```
python3 scripts/sim2real/01b_build_estimated_skeleton.py --overwrite      # 语料层退化
python3 scripts/sim2real/03_build_windows.py --benchmark configs/sim2real/benchmarks/tc_rlowarm_w24_estskel_v1.json
python3 scripts/sim2real/06_run_matrix.py --matrix configs/sim2real/matrix_estskel_v1.yaml --gpus 0,1,2,3,5,7
python3 scripts/sim2real/07_report.py --benchmark-id tc_rlowarm_w24_estskel_v1
```

## ⚠️ 更正（2026-07-07 第二轮审查）：逐序列对齐修复后的复跑对照（核心结论幸存）

同 v1，lag 口径重建 windows 后全矩阵 + real+real 对照复跑
（`outputs/sim2real_lagfix/tc_rlowarm_w24_estskel_v1/`，3 seeds，样本标准差）：

| 格 | 修复前 | 修复后 |
|---|---|---|
| TRTR | 0.0362±0.0080 | 0.0414±0.0136 |
| TSTR naive | 0.0172±0.0047 | 0.0156±0.0015（仍 ≫ 其余≈随机线） |
| mix real+naive | 0.0521±0.0058 | **0.0547±0.0017（+32% vs TRTR）** |
| mix real+globalpose(clean) | 0.0458±0.0150 | 0.0488±0.0010（+18%，E3 强化） |
| mix real+real（对照） | 0.0305±0.0075 | 0.0375±0.0042 |
| 打乱对照 | 0.0010 | 0.0000 |

**E1（naive 排名第一）、E2/强化节（naive mix 是真信号：real+naive > TRTR > real+real）
全部幸存**；相对增益从 +44% 收敛到 +32%（TRTR 上修所致），E3 的 globalpose(clean)
正增益也复现且方差大幅收窄。绝对数以本节为准；定稿仍待 6 seeds + real+抖动real 控制（计划 C）。
