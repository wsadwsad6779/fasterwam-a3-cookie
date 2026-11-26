# 旧"12k 入盒"溯源账本（P3, 2026-11-26）

目标：找用户所说"旧 12k 入盒 3 块 / 6 块"的可核原始证据（checkpoint/seed/场景/in_target 定义/结果 JSON）。仅找到不同版本 → 不得声称找回 3/6。

## 证据条目（来源 → 归属）
| 条目 | 来源 | 内容 | 归属 |
|---|---|---|---|
| E1 | `2026-11-07/ea751058/_scan_out1.txt` | variant `E_rows_lift0012_f045`：`in_target_final: 8`，`success:false`，stop `expert_failed: released Cookies not upright, settled and contained`；variant `F_rows_nocap_f045`：`in_target_final: 2`，`success:false` | **FOUND_UNATTRIBUTED**：这是 **A3 LoRA r4/r5**（adapter.pt step_004000/040000，40k 合并谱系）+ 旧场景的**策略**结果；入盒是"倒着在框里"(not upright)→ 当时已判失败；**不是 3/6，也不是当前 F0/F2** |
| E2 | `2026-11-07/ea751058/_scan_succ.txt` | `"successes": 1, "success_rate": 0.3333, seeds [11,12,13]` | **FOUND_UNATTRIBUTED**：一段**函数/调用片段**（seeds 11–13，1/3），**非 checkpoint 结果**；这就是"旧 ~1/3 成功"的来源，不是"3/6 块" |
| E3 | `p_items/T7_GRASP_ROOT_CAUSE_15K.json` | "旧无加力 12k checkpoint 在本实例不存在（仅 40k 合并、base、12 步 rehearsal）" | **NOT_FOUND**：旧无加力 12k 权重缺失 |
| E4 | 当前 `_t7_eval_diag` 对 F0 15k（7500–7509）strict+relaxed | 均 0.0/10 | 现模型组：0 |
| E5 | 用户口述"12k 入盒 3/6 块" | 无对应 checkpoint/seed/scene/结果 JSON | **NOT_VERIFIED**（未找到原文） |

## 谱系区分（时间线）
- **旧 A3 LoRA r4/r5**（adapter, 40k 合并）：E1/E2 出自此；**pm1 二值夹爪**；旧场景/厚垫；**禁止作为当前模型初始化**。
- **旧无加力 12k**（用户所指）：**本实例缺失（NOT_FOUND）**。
- **当前官方 FasterWAM F0/F2**：base step_021700 + action expert；V3 连续标签；V3.1 薄垫场景。与上面两条均不同谱系。

## 结论
- **"旧 12k 入盒 3/6 块"= NOT_VERIFIED（未找到原文）**；可核的最近证据是 **A3 LoRA/pm1 的 `in_target_final 8/2` 且 success:false（倒着进框）** 与 **1/3(snippet, seeds11–13)**，均 **FOUND_UNATTRIBUTED**。
- 旧模型缺失 → **不做伪配对、不拿 40k 初始化**。若要真正同条件复评：先从 GitHub/HF 归档找回旧 checkpoint + 旧评估适配器，先出一页迁移协议，再评估。
