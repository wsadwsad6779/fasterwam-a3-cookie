# 旧"12k 入盒"与当前 F0/F2 的可比性核查（2026-11-26）

## 结论先行
- 我在本机记录里**没有找到**"当前 FasterWAM F0/F2 在 12k 入盒 3 块 / 6 块"的可验证证据。
- 记录里最接近"入盒"的证据来自**另一条旧管线**：`A3 LoRA r4/r5`（`adapter.pt step_004000 / step_040000`），**pm1 二值夹爪**、**旧场景**；评估输出 `in_target_final: 8 / 2`，但 **`success: false`**，原因 `"expert_failed: released Cookies not upright, settled and contained"` —— 即**饼干进了目标区但没立正（tip）**。
- 唯一 `"successes": 1 / success_rate 0.3333` 的记录是一个**函数片段（seeds=[11,12,13]）**，**不是 checkpoint 结果**（见 `p_items/T7_GRASP_ROOT_CAUSE_15K.json` 的 `in_repo_evidence`）。
- 依据 `T7_GRASP_ROOT_CAUSE_15K.json`：**旧无加力 12k checkpoint 在本实例不存在**（此前盘点只有 40k 合并权重、base、12 步 rehearsal）。→ **"旧 12k 成功 vs 现在失败"目前 NOT_VERIFIED，不能据此判"加力导致退化"或"训练退化"**。

## 协议差异表（关键：不可直接比）
| 维度 | 旧"入盒"记录 | 当前 F0/F2 |
|---|---|---|
| 模型 | A3 LoRA r4/r5（adapter） | FasterWAM F0(无加力)/F2(力) |
| 夹爪参数化 | **pm1 二值 开/合** | **V3 连续 0.28夹/0.49放/1.0开** |
| 场景 | 旧目标盒/垫几何 | V3.1 薄垫 0.0005、target y≈0.100 |
| 成功判据 | 严格"upright+settled+contained"（当时已判 false） | strict `cookies_in_target` + exact_fill(src==70) |
| 评估链 | 旧 eval/adapter | eval 5edaad7f（37376d2c+fail-closed）、sim 62f6c863 |
| seeds | 11–13（snippet） | 7500–7509 / 7200–7209 等 |
| "3块/6块"具体 | **未找到原文**；可核的是 in_target_final 8/2 且 success=false | F0 15k: strict 0/relaxed 0 (10/10) |

→ 旧"入盒"是**不同模型+pm1+旧场景下的"倒着进框"（strict 反而不算成功）**；与当前 V3 连续夹爪模型**不是同一任务表征**，数字不可横比。

## 要拿到"真正同条件对照"缺什么
1. 旧 checkpoint（A3 LoRA 的 `adapter.pt`，或任一"旧 12k 无加力"）——本实例缺失；需从 GitHub/HF 归档找回。
2. 旧场景清单 + 旧评估适配器（pm1 动作参数化）——才能在同一场景评估旧模型。
3. 若无法找回 → 该对比标 **NOT_VERIFIED**，不以"旧 12k 成功"论证退化。

## 已做的证据/工具改进
- `_t7_eval_diag_v2.py`（pod sha `4c3e0d1d`）：每次 rollout 存 `trajectory.npz` = 逐帧**每块饼干 xpos+xmat(朝向) + 目标盒 pos+mat + COOKIE_HALF_SIZE/TARGET_INNER_HALF_SIZE/floor_top/wall_height/max_tilt + 力/开度/EEF**。→ 之后 strict+relaxed **离线秒算**（含"一条边贴盒底"的朝向判定），无需再跑 6.26B 闭环。
  - 说明：上一轮 F0 15k relaxed 是用**旧脚本**跑的（只存 cookie_pos 位置），故花了 ~30 分钟重放；后续用 v2。

## 待办（不需立即重训）
1. 找回旧 checkpoint/旧评估适配器（GitHub/HF 归档）→ 同场景同 seed 评估。
2. 用 v2 对 F0 9k/12k/15k 同 seed 重跑，产出 trajectory.npz，离线统计"离开/接触未闭合/夹住滑脱/提起未送达/短暂入盒又离开 + 严格/relaxed/回合中曾入盒"。
3. 复核 Flow-Matching 目标：确认第 7 维损失作用在"流匹配速度场"而非"最终开度均值"，据此检验"均值坍缩"假设（而非直接断言）。

## 附：说"只改了夹爪标签"是不准确的——至少 5 处同时不同（有证据）
1. **夹爪标签**：V2 是 pm1/二值（os 记录：carry 期 opening≈0.45 被塌成 +1 关闭）；V3 改为连续 0.28/0.49/1.0（`GRIPPER_LABEL_V3_BUILD.json`：gripper_min0.28/max1.0）。**这一处确实改了。**
2. **场景几何变了**（`EVAL_SCENE_FREEZE_DIFF_V3_TO_V31.json`）：旧 freeze `99194298` 厚垫 0.003175、target bin y≈**-0.011**；新 freeze `cef07926` 薄垫 0.0005、target y≈**+0.100**；**80/80 场景 hash 全不同**（目标盒移动约 11cm）。
3. **模型谱系变了**：旧"入盒 8/2"是 **A3 LoRA r4/r5**（adapter, step_040000，40k 合并）；现在是 **FasterWAM 官方 base step_021700 + action expert**（F0/F2），架构/初始化都不是同一个。
4. **数据变了**：800→845 集，V2 split 589/85/171，V3 labels sha `c4090dd2`、8D proprio、H32、30/20/50 采样器。
5. **评估链/映射变了**：`g=2a-1 / trans×2 / rot×5` 修复；eval `37376d2c`→`5edaad7f`(fail-closed)；C-01 曾"restore pipeline"（说明旧跑可能用的是不同/有缺陷的管线）。
→ 因此"旧 12k 入盒 vs 现在 0"被**≥5 个因素**同时混淆，**不能归因于单一"夹爪标签"**，也不能据此判"退化/加力有害"。

## 关于"均值坍缩"更准确的表述 + 一个便宜的免训练检验
- `ACTION_SUPERVISION_AUDIT_15K.json`：第 7 维损失是 **Flow-Matching 速度场 MSE**（`fastwam.py:850`，7 维等权、无 per-dim、无 pm1/clip/平滑），归一化 0.28→-1 / 0.49→-0.417 / 1.0→+1。→ "把最终开度回归到均值"是**假设**，需用 teacher-forced 分布 + 去噪轨迹验证，不能当成已证。
- 该审计还指出：**仅 5 步去噪可能"解不干净"多模态**。→ 一个**免训练**的检验：同一 F2 15k 用 `DENOISE=10/20` 重跑 small seed 集，看执行 <0.30 比例与入盒是否变化（若变好，说明是去噪步数不足，而非"标签/损失均值化"）。
