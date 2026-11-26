# Task7 · FasterWAM + 指尖力 · 本周归档（2026-11-24 ~ 2026-11-30）

> **说明（重要）**：本 README **只描述本周（new）**。
> 更早的历史归档、完整旧代码与**旧的根 README 原文**都保留在旧分支里（见下方"分支索引"），此处不再堆叠旧记录，仅在"历史"一节做指引。

## 1. 本周做了什么（new）
- 训练保持暂停：F0 在 15k **冻结**（`step_015000.pt` sha `8ef07241…`），F2 在 15800 **冻结**（`fa9d87e3`），F1 未启动。
- 评估：F0 3k–15k、F2 3k–15k（官方 r10）strict in-box 0/10、exact_fill 0/10；F0 15k relaxed 也 0/10。
- r 扫描（F2 15k）：r=10/16/32 → strict 0/0.2/0.3、relaxed 0/0.5/0.7、exact 全 0。
- **F0 r 扫描（本周新增，同 diag 评估器，7500–7509）**：r=10/16/32 → strict 0/0/0、relaxed 0/0/0、**曾入盒 0/0/0** → **r32 不救 F0**；r32 入盒是 F2(力)+少数 seed 特有。
- H 扫描（F2 15k 零训练）：H32 relaxed0.7/strict0.3 → **H48/64 = 0**；不 +16、不细扫（H>32 未训练=OOD）。
- 去噪 K 扫描（F2 15k 冻结观测）：**K 越大闭合召回越低**（CARRY 0.708→0.438→0.438）→ **否证"5 步去噪太少"**。
- 夹爪动作语义（源码级）：env 目标开度=`(g+1)/2`，eval `g=2a−1` ⇒ **目标开度=模型输出开度 a（恒等，无解码 bug）**；官方 FasterWAM 评估用 `sign` 二值；mid-band 0.43 ⇒ 连续抓不住 / sign 放不开（两者 exact 都 0）。0.28 非通用判据（用力+结果）。
- 旧"12k 入盒 3/6" = **NOT_VERIFIED**；最接近是 A3 LoRA/pm1 `in_target_final 8/2 success:false`（倒着进框）；旧无加力 12k ckpt 缺失。

## 2. 目录（本分支只放精炼产物）
- `history/2026-11-23_to_25_baseline_rH_B/` — F0/F2 里程碑、r/H 扫描、B2–B5 诊断（含 README.md + 有用证据/图）
- `history/2026-11-26_denoise_gripper_semantics/` — 去噪 K 审计、夹爪语义审计、旧实验溯源、F0 r-sweep（含 README.md）
- `code/` — 本周实际用到的脚本（eval_diag_v2、P0/P1 探针、r-sweep 链与收集、视频打包、归档脚本等）

## 3. 与"历史（旧）"的区分
- **本分支不包含旧的全量工作区**；旧的全量代码、早周归档、**旧根 README 原文**都在下列旧分支中（未改动、仅改名）。
- 需要旧内容 → 去对应旧分支；需要本周内容 → 看本分支 `history/` 与 `code/`。

## 4. 分支索引（GitHub origin，按时间命名，2026-11-26 重命名）
| 分支 | 文件数 | 内容/用途 |
|---|---|---|
| `2026-11-01_archive` | 1221 | 早期归档/快照 |
| `2026-11-01_backup-r6-audit` | 288 | r6 审计备份 |
| `2026-11-08_a3-cookie-r27` | 239 | A3 cookie r27；含 `bridge/action_adapter(_v2/_v3).py` 等适配器 |
| `2026-11-17_rescue-40k` | 680 | 40k 恢复代码/身份账本 |
| `2026-11-19_main` | 1082 | 主开发线（**旧根 README 原文在此**）|
| `2026-11-24_fingertip-force` | 1375 | Task7 工作区（p_items 全量/训练评估脚本）|
| `history-2026-11-24_to_11-30` | 本周 | **本分支/默认分支**：仅本周精炼归档+README |

## 5. 镜像与产物
- HF 数据集（经 hf-mirror 上传）：https://huggingface.co/datasets/wsadwsad6779/fasterwam-a3-cookie-task7
- 视频包（pod，`/workspace` 持久保留）：`/workspace/fwv1/EVAL_VIDEOS_named_20261126.zip`（314 mp4，1.32GB，sha256 `3a68853d…`，内含 `00_MANIFEST.csv` 按实验/seed 命名）。

## 6. 安全
- HF token 仅在命令环境临时使用，未写入任何文件/提交；因已在对话出现，建议尽快在 HF 后台 rotate。
