# Task7 · FasterWAM + 指尖力 · 打包总述（按时间归档）

本仓 `fasterwam-a3-cookie` 的正式评估/训练证据按时间归档在 `history/` 下；HF 同步镜像：
https://huggingface.co/datasets/wsadwsad6779/fasterwam-a3-cookie-task7

## 目录结构（本分支）
- `history/2026-11-23_to_25_baseline_rH_B/` — F0/F2 里程碑、r/H 扫描、B2–B5 诊断（含 README.md）
- `history/2026-11-26_denoise_gripper_semantics/` — 去噪步数审计、夹爪动作语义审计、旧实验溯源、F0 r-sweep（含 README.md）
- 工作区（未归档的完整证据/脚本/中间件）仍在 `p_items/`、以及仓库根的历史文件。

## 分支索引（origin 实际状态，git ls-remote 核实）
- `fingertip-force-v1`：**唯一有内容的分支**（293 文件：p_items 全套证据、训练/评估脚本、本归档）。
- `main` / `a3-cookie-r27` / `archive-2026-11-01` / `backup-2026-11-01-r6-audit` / `rescue-40k-restore-20261117`：**origin 上为空树（0 文件）**。
- 本分支（`history-2026-11-24_to_11-30`）由 fingertip-force-v1 派生，仅**新增** history/ 与本文档，未改历史文件；设为默认分支。

> 注：若你认为"还有很多分支"，那可能在另一台机器/另一个仓库或官方 hustvl/FasterWAM；本仓 origin 经 ls-remote 只有以上 6 个。

## 冻结状态（勿改）
- F0 15k 冻结：`step_015000.pt` sha256 `8ef07241…`（strict/relaxed 0/10）。
- F2 暂停 15800：`fa9d87e3`（只读）。F0 9k `7ca235ba`、F0 12k `d3f2a3a8`、F2 15k `83aacf23`。
- 训练全停（F0 15k 冻结、F2 15800 冻结、F1 未启）。

## checkpoint 说明
大文件 checkpoint（各 ~3.35GB）在 pod `/workspace`（**销毁后 /workspace 持久保留**），未上传（体积+带宽）；sha 已写入各证据 json。

## 安全
HF token 仅在命令环境临时使用，**未写入任何文件/提交**；因已在对话中出现，建议尽快在 HuggingFace 后台 **rotate**。
