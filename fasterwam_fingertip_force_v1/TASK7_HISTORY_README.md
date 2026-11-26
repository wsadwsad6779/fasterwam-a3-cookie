# Task7 · FasterWAM + 指尖力 · 打包总述（按时间归档）

本仓 `fasterwam-a3-cookie` 的正式评估/训练证据按时间归档在 `history/` 下；HF 同步镜像：
https://huggingface.co/datasets/wsadwsad6779/fasterwam-a3-cookie-task7

## 目录结构（本分支）
- `history/2026-11-23_to_25_baseline_rH_B/` — F0/F2 里程碑、r/H 扫描、B2–B5 诊断（含 README.md）
- `history/2026-11-26_denoise_gripper_semantics/` — 去噪步数审计、夹爪动作语义审计、旧实验溯源、F0 r-sweep（含 README.md）
- 工作区（未归档的完整证据/脚本/中间件）仍在 `p_items/`、以及仓库根的历史文件。

## 分支索引（按时间命名，2026-11-26 重命名；origin 实际）
- `2026-11-01_archive`（1221 文件）
- `2026-11-01_backup-r6-audit`（288）
- `2026-11-08_a3-cookie-r27`（239；含 bridge/action_adapter_v3.py 等适配器）
- `2026-11-17_rescue-40k`（680）
- `2026-11-19_main`（1082）
- `2026-11-24_fingertip-force`（1375）
- `history-2026-11-24_to_11-30`（本周归档，**默认分支**，本 README 所在）
> 原名 main/a3-cookie-r27/archive-2026-11-01/backup-2026-11-01-r6-audit/rescue-40k-restore-20261117/fingertip-force-v1 已按时间前缀重命名（内容/提交未改）。

## 冻结状态（勿改）
- F0 15k 冻结：`step_015000.pt` sha256 `8ef07241…`（strict/relaxed 0/10）。
- F2 暂停 15800：`fa9d87e3`（只读）。F0 9k `7ca235ba`、F0 12k `d3f2a3a8`、F2 15k `83aacf23`。
- 训练全停（F0 15k 冻结、F2 15800 冻结、F1 未启）。

## checkpoint 说明
大文件 checkpoint（各 ~3.35GB）在 pod `/workspace`（**销毁后 /workspace 持久保留**），未上传（体积+带宽）；sha 已写入各证据 json。

## 安全
HF token 仅在命令环境临时使用，**未写入任何文件/提交**；因已在对话中出现，建议尽快在 HuggingFace 后台 **rotate**。


## 2026-11-26 追加（F0 r-sweep + 视频包 + 上传状态）
- **F0 15k r=10/16/32 ×7500–7509（同 diag 评估器）**：strict 0/0/0、relaxed 0/0/0、**曾入盒 0/0/0**、exact 0；grasps 0/0.9/1.1。→ **r32 不救 F0（连曾入盒都0）**；r32 入盒是 F2(力) 特有（r16 strict0.2/relaxed0.5，r32 strict0.3/relaxed0.7）。证据见 history/2026-11-26_denoise_gripper_semantics/F0_R_SWEEP_RESULTS.json。
- **全视频打包（pod）**：`/workspace/fwv1/EVAL_VIDEOS_named_20261126.zip`（314 个 mp4，1.32GB，sha256 `3a68853d4fbb2007c9f5c969546be429189e3c3d1f39d25b96ebdcfe14c12c3d`；内含 `00_MANIFEST.csv`，按实验/seed 命名）。/workspace 销毁后持久保留，可从 JupyterLab 文件浏览器下载。
- **HF 上传**：本机到 huggingface.co 上传端点连接超时（WinError 10060/10061），pod 直连 HF 亦不可用 → **HF 本轮未完成**（仓 `wsadwsad6779/fasterwam-a3-cookie-task7` 已建、空）。需在能连 HF 的网络执行：`huggingface-cli upload --repo-type dataset wsadwsad6779/fasterwam-a3-cookie-task7 ./history history`。
