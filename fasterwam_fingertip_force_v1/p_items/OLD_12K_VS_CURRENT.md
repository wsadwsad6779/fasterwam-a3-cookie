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
