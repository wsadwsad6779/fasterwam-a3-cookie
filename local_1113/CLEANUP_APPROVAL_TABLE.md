# pod 磁盘清理审批表（只提交，不执行）

生成时间 2026-11-01，pod `rgapi-one-click-24316-57b6b870`。
**本轮没有删除任何东西**：所有数字来自已回传到本机的只读扫描（`_r6_stage/out/backup_filelist.json` + `backup_hashes.jsonl` → `backup_status_abc.json`），"是否仍被代码引用"一列来自 pod 上的只读 grep 回读（`_r6_stage/out/refscan.log`），不是第一遍清单里那个布尔值。

## 0 判定规则

1. **只有 A 档（pod 外存在逐字节 sha256 相同的副本）才允许进入"建议清理"**；
2. C 档（pod 唯一副本）无论看起来多没用，一律标"先备份再谈清理"；
3. pod 内部同 sha256 的第二份不算备份——它们在同一块盘上；
4. "无引用"是 grep 全 `/workspace` 文本文件的结果，只能证明没有脚本按路径读它，不证明语义上没用。

全 pod 逐文件档位（3888 个文件，来自内容哈希实扫）：A 档 81 个 10.36 GiB / A0 档 26 个 0.00 GiB / B 档 987 个 39.59 GiB / C 档 2794 个 42.13 GiB

## 1 明确禁止删除

| 绝对路径 | 容量 | 是否仍被代码引用 | 外部备份路径 | SHA256 是否匹配 | 是否唯一副本 | 建议 | 预计释放 |
|---|---|---|---|---|---|---|---|
| `a3_train_runs/a3_lora_r3/step_003000_merged.pt` | 11,117,977,659 B（10.35 GiB，1 个） | 现役引用：`_r4_stage/run_r5.sh` 的 `FW_BASE_CKPT`（缺文件直接 ABORT）、`_r4_stage/run_r4.sh`、`_r5_stage/r5_eval.sh` 的 `--ckpt`、`chk3_run.sh`、`bridge_next/test_light_state.py` | 无 | 1 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **禁止删除** | 0 |
| `a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/` | 185,090,773 B（0.17 GiB，5 个） | 现役引用：`a3_eval/r4_b0_c0_paired/manifest_lora.json` 的 `adapter.path` | 无 | 5 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **禁止删除** | 0 |
| `a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/` | 185,090,775 B（0.17 GiB，5 个） | 现役引用：`a3_eval/r5_b0_c0_paired/manifest_lora.json` 的 `adapter.path` | 无 | 5 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **禁止删除** | 0 |
| `data/a3_front_close_left_100` | 4,306,574,611 B（4.01 GiB，60 个） | 现役引用：`a3_data/a3_fw/summary.json` 与全部训练/评估 config 的 `dataset` 字段 | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eb | 60 个只声明"已上传/可重下"，未做读回 sha256 核验（B） | 是（pod 外那份未读回核验，按唯一副本对待） | **禁止删除** | 0 |
| `a3_data/a3_fw_dagger/` | 652,802,564 B（0.61 GiB，4 个） | 现役引用：`_r4_stage` 之后的训练 config | 无 | 4 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **禁止删除** | 0 |
| `a3_data/a3_fw_dagger_near/` | 917,773,169 B（0.85 GiB，4 个） | 现役引用：同上 | 无 | 4 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **禁止删除** | 0 |
| `a3_data/a3_fw/` | 12,851,445,040 B（11.97 GiB，4 个） | 现役引用：`a3_eval/*/manifest_lora.json` 的 `stats.path` 指向 `a3_fw/dataset_stats.json` | pod 外零逐字节副本；另有 1 个声称已上传但未做读回哈希核验（B 档） | 1 个只声明"已上传/可重下"，未做读回 sha256 核验（B）；3 个 pod 外零副本，无哈希可比（C） | 混合：1 个只在 pod 外声称有副本（未读回核验，不能当作已有副本），3 个是 pod 唯一副本 | **禁止删除** | 0 |
| `a3_eval/` | 523,227,691 B（0.49 GiB，375 个） | 评估结果本体 | 只有 15 个（.mp4 .txt）在 pod 外有逐字节相同的副本（Windows 视频合集），其余 360 个（.json .log .npz .pt .tsv .txt）pod 外零副本；A0 是 1 个 0 字节占位，哈希相同不代表备份 | 14 个与 pod 外副本 sha256 逐个相等（A）；1 个是 0 字节占位，哈希相等无意义（A0）；360 个 pod 外零副本，无哈希可比（C） | 混合：15 个 pod 外有逐字节相同的副本，360 个是 pod 唯一副本 | **禁止删除** | 0 |
| `fasterwam_a3_bridge/bridge/outputs/` | 5,074,359,908 B（4.73 GiB，212 个） | 评估结果本体 | 只有 27 个（.mp4）在 pod 外有逐字节相同的副本（Windows 视频合集），其余 185 个（.csv .dat .json .jsonl .npz）pod 外零副本 | 27 个与 pod 外副本 sha256 逐个相等（A）；185 个 pod 外零副本，无哈希可比（C） | 混合：27 个 pod 外有逐字节相同的副本，185 个是 pod 唯一副本 | **禁止删除** | 0 |
| `a3_train_runs/a3_lora_r1/step_002000_merged.pt` | 11,117,977,659 B（10.35 GiB，1 个） | 现役引用：`activate_fasterwam_a3.sh` 的 `FASTERWAM_A3_CKPT` 默认指针、`fasterwam-a3-cookie/bridge/_next/common.py` | HuggingFace（本轮已逐字节回读核验） | 1 个与 pod 外副本 sha256 逐个相等（A） | 否（pod 外已有逐字节相同的副本） | **禁止删除** | 0 |
| `a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt` | 11,179,613,340 B（10.41 GiB，1 个） | 仅历史引用：`watch_r3.sh`、训练日志、本轮审计报表 | 无 | 1 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **禁止删除** | 0 |

## 2 用户点名的四类候选，逐项结论

| 绝对路径 | 容量 | 是否仍被代码引用 | 外部备份路径 | SHA256 是否匹配 | 是否唯一副本 | 建议 | 预计释放 |
|---|---|---|---|---|---|---|---|
| `a3_train_runs/a3_lora_r2/step_003000_merged.pt` | 11,117,977,659 B（10.35 GiB，1 个） | 仅历史引用：`watch_r2.sh`、`run_r3.sh`（当年造 r3 的脚本）、`fasterwam_a3_bridge/bridge/outputs/r2/C0/manifest_lora.json`（历史记录）、`_r4_stage` 日志、本轮 `_r6_stage/out` 审计文件 | 无 | 1 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 10.35 GiB |
| `a3_train_runs/chk3_A/` | 185,095,469 B（0.17 GiB，6 个） | 无引用（grep 未命中任何脚本/配置） | 无 | 6 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.17 GiB |
| `a3_train_runs/chk3_A2/` | 185,095,470 B（0.17 GiB，6 个） | 无引用 | 无 | 6 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.17 GiB |
| `a3_train_runs/chk3_B/` | 185,095,453 B（0.17 GiB，6 个） | 无引用 | 无 | 6 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.17 GiB |
| `a3_train_runs/chk3_B2/` | 185,095,473 B（0.17 GiB，6 个） | 无引用 | 无 | 6 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.17 GiB |
| `a3_train_runs/chk3_B3/` | 185,095,473 B（0.17 GiB，6 个） | 无引用 | 无 | 6 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.17 GiB |
| `a3_train_runs/chk4_N/` | 185,095,464 B（0.17 GiB，6 个） | 无引用 | 无 | 6 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.17 GiB |
| `a3_train_runs/chk4_S/` | 370,186,035 B（0.34 GiB，11 个） | 无引用 | 无 | 11 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.34 GiB |
| `a3_train_runs/a3_lora_r4_crashed_step500/` | 185,140,482 B（0.17 GiB，9 个） | 无引用 | 无 | 9 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档，pod 外零副本） | 0.17 GiB |
| `FasterWAM/checkpoints/Wan-AI/` | 14,202,213,669 B（13.23 GiB，6 个） | 现役引用：`DIFFSYNTH_MODEL_BASE_PATH` 指向 `/workspace/FasterWAM/checkpoints`，缺文件会触发向 hf-mirror 重下 5B 模型 | ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable) | 6 个只声明"已上传/可重下"，未做读回 sha256 核验（B） | 是（pod 外那份未读回核验，按唯一副本对待） | **谨慎：可重下但未核验** | 13.23 GiB |
| `FasterWAM/checkpoints/fasterwam_release/` | 11,117,720,881 B（10.35 GiB，2 个） | 仅历史/预训练指针 | HuggingFace/Wan-AI upstream release step_021700.pt (re-downloadable) | 2 个只声明"已上传/可重下"，未做读回 sha256 核验（B） | 是（pod 外那份未读回核验，按唯一副本对待） | **谨慎：可重下但未核验** | 10.35 GiB |
| `a3_train_runs/a3_lora_r4/{5,10,15,20,25,30,35}00 与 a3_lora_r5/{45,50,55,60,65,70,75}00` | 2,591,270,853 B（2.41 GiB，70 个） | 无现役引用（评估只读 step_004000 / step_008000） | 无 | 70 个 pod 外零副本，无哈希可比（C） | 是（pod 唯一副本） | **先备份再谈清理**（C 档） | 2.41 GiB |

## 3 中间 step 逐文件明细

- `a3_train_runs/a3_lora_r4/checkpoints/light/step_000500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_000500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_000500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_000500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_000500/trainer_state.json` 638 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001000/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001000/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001000/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001000/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001000/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_001500/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002000/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002000/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002000/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002000/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002000/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_002500/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003000/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003000/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003000/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003000/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003000/trainer_state.json` 641 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r4/checkpoints/light/step_003500/trainer_state.json` 641 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/trainer_state.json` 639 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005000/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005000/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005000/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005000/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005000/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_005500/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006000/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006000/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006000/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006000/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006000/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_006500/trainer_state.json` 642 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007000/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007000/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007000/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007000/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007000/trainer_state.json` 641 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007500/optimizer.pt` 123,478,413 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007500/adapter.pt` 61,595,489 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007500/rng.pt` 14,557 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007500/scheduler.pt` 1,675 B  档 C  pod 唯一副本
- `a3_train_runs/a3_lora_r5/checkpoints/light/step_007500/trainer_state.json` 641 B  档 C  pod 唯一副本

## 4 空间算术

| 组 | 释放量 | 前提 |
|---|---|---|
| chk3_A/A2/B/B2/B3 + chk4_N/S + r4_crashed（8 个测试目录） | 1.38 GiB | C 档：要么先逐个传出去并核验 sha256，要么由你明确豁免"测试产物不必备份" |
| r2 merged | 10.35 GiB | C 档，11 GB，pod 外零副本 |
| 14 个中间 step | 2.41 GiB | C 档 |
| Wan-AI + fasterwam_release | 23.58 GiB | B 档，公开源可重下，代价是 25 GB 重下 与 `io.py:49` 相对路径写满盘的风险 |
| **合计上限** | **37.90 GiB** | 但除最后一行外全部卡在"没有 pod 外 sha256 副本"这一条 |

当前 `/workspace`（本轮 `df -B1` 实测，2026-11-01）：总 105,089,261,568 B / 已用 96,303,927,296 B / **空闲 3,399,847,936 B（3.17 GiB，97%）**。
带图重采需要 9.88 GB（301,056 B/帧 × 32,818 帧），所以现在这点空闲放不下；2–3 条 smoke 只要 0.41–0.61 GB，还能塞进去。

## 5 需要批准的只有两件事

1. 给一个 pod 外可写的持久存储（见第 6 节的最小权限），把 C 档里你认定要留的大件传出去并读回核验；在此之前 r2 merged、中间 step、chk3/chk4 测试目录都只能挂着。
2. 或者明确豁免：同意"light_state 恢复测试目录与 r4_crashed 残骸属于一次性诊断产物，不需要 pod 外副本，可以直接删"，那样立刻回收 1.38 GiB。

## 6 任务五：外部大文件备份需要你的哪一条权限

pod 侧凭据现状（全部只读实测，没有任何一次写操作）：

- 环境变量 `HF_TOKEN`、`HUGGING_FACE_HUB_TOKEN` 都不存在；`~/.cache/huggingface/token` 不存在；`huggingface_hub.whoami()` 直接抛 `LocalTokenNotFoundError`。**所以 pod 现在对任何 HF 仓库都只有匿名读，没有写权限**（HF 上传必须带 token，公开仓库的匿名写会被 401 拒掉）。
- `ssh -T git@github.com` 在 pod 上 22 端口 connection timed out，所以 pod 也 push 不了 Git。**GitHub 那条路只能由本机走**（本轮 `wsadwsad6779/fasterwam-a3-cookie` 的 `backup-2026-11-01-r6-audit` 分支就是从本机推的；推之前对整棵树做过 secret 扫描，`hf_`/`ghp_`/`AKIA`/`PRIVATE KEY` 模式零命中，`backup/2026-11-01/repro/cookie_stash.txt` 实测 0 字节，没有密钥外泄）。
- 因此本轮备份只做到"代码 + 清单 + 影像"，**权重与训练态大件全部仍是 C 档**；在完成读回哈希核验之前，不宣布"全量备份完成"。

最小权限（按你想要的量级三选一，不要更多）：

| 方案 | 需要的最小权限 | 能带走 | 覆盖不了 |
|---|---|---|---|
| A：HF 写 token | 只对 `wsadwsad6779/fasterwam-a3-cookie` 一个仓库的 `models.write`（fine-grained token 里勾这一个 repo 的 write，不给 other repos、不给 write:repositories 全量范围） | r2/r3 merged 各 10.35 GiB、r3 weights 10.41 GiB、r4/r5 末态 light 目录 0.34 GiB、中间 step 2.41 GiB | HF 单仓库软上限，25 GB 级别要 LFS 配额；评估 npz 42 GiB 里的大头会顶到配额墙 |
| B：指定别的持久存储 | 一个 S3/R2/网盘的 **单 bucket 前缀**写权限 + 可读回即可（不需要账号级权限） | 任意大小，包括 42 GiB 的评估结果 | 需要我按你的命名规则建目录，读回核验同样逐文件 sha256 |
| C：Runpod Network Volume | 在你账户下建一个 volume：只为解锁本表的清理候选， 14 GiB 就够（r2 + 8 个测试目录 + 14 个中间 step）；要把全部 C 档搬走则是 42 GiB | 独立于这块 98 G 盘、pod 删除后仍在 | 严格讲仍在同一云厂商、同一账户；若 pod 和 volume 一起删就同时没了——按你定的"pod 外"判据，我不把它单独算作备份，除非你确认它算 |

凭据交接方式（保证 token 不进代码/日志/commit/聊天正文）：

1. 不要把 token 发在对话里。
2. 由你在 pod 的 JupyterLab 终端里自己敲 `hf auth login --token`（写进容器内 `~/.cache/huggingface/token`，路径在 `/workspace` 之外，不会被我后续的任何目录扫描或清单抓到）；或只给我一个"环境变量注入"的位置，我调用时不 echo、不落盘、不写进任何 .sh。
3. 我的上传脚本只从 `HF_TOKEN` 环境变量或 `hf auth` 已登录态读取，日志里只打印仓库名、文件路径、字节数与 sha256；任何一次上传后我都会重新读回该远端路径做逐文件 sha256 对拍，只有对拍通过的文件才从 C 档改记为 A 档。

在这一条落实之前，本表第 2 节所有"先备份再谈清理"的行都不会变成"可清理"。
