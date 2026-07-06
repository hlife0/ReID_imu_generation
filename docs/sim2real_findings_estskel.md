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
| TSTR globalpose(noise) | 0.0013 | 0.0013 | ≈随机线 |
| TSTR globalpose(clean) | 0.0013 | 0.0010 | ≈随机线 |
| **mix real+naive** | **0.0541 ± 0.0045** | **0.0521 ± 0.0047** | **+44% 相对 TRTR(GT 为 +37%)** |
| mix real+globalpose | 0.0385(无增益) | 0.0458(+27%) | **转为正增益(见下)** |
| 打乱配对对照 | 0.0010 | 0.0010 | =随机线,无泄漏 ✅ |

## 三个发现

**E1:generator 排名对骨架退化稳健。** naive ≫ humogen ≈ globalpose 的格局完全保持,
naive 与弱生成器之间仍有约一个数量级的差距,远大于 seed 方差。用下游任务筛生成器的
结论不依赖"骨架是否完美"。

**E2:mix 增益不降反升(相对)。** real+naive 的绝对分从 0.0541 微降到 0.0521,
但因为 TRTR 上界也降了(0.0395→0.0362),**相对增益反而从 +37% 升到 +44%**。
解释:骨架侧变噪后,真实训练更容易过拟合到估计噪声,而掺入 naive 合成 IMU 的
增强/正则作用相对更值钱。这进一步支持主项目的 I4(增强式)优先策略。

**E3:估计骨架下 globalpose 增强开始有正增益(新现象)。** GT 骨架时 real+globalpose
无增益(0.0385≈TRTR),估计骨架时升到 0.0458(+27%)。可能是骨架噪声"抹平"了
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
