# BACKUP_MANIFEST v2（pod 全量对账版）

- 生成：2026-11-01，pod `rgapi-one-click-24316-57b6b870`，根目录 `/workspace`。
- v1（`BACKUP_MANIFEST.md`，963 个 / 47.88 GiB）只枚举了当时想到的目录。本版用一次全量 `find`（4,933 个文件 / 92.08 GiB）与清单做差集，补入 2,925 个此前未记录的文件，跳过 1,046 个 shell 碎片/缓存/字节码。
- 数据链路：`backup_scan.py` → `scan_extra.py` → `scan_checkpoints.py`（枚举）→ `backup_hash.py`（流式 sha256）→ `manifest_rows.py`（合并成 TSV）→ 本脚本。合并表回传 8 个分片后本地 sha256 = `62afef81caed177f`，与 pod 端记录一致，逐字节相同。

## 1 汇总

| 指标 | 数值 |
|---|---|
| 文件数 | 3888 |
| 总字节 | 98,876,695,253 B（92.09 GiB） |
| 已算出 sha256 | 3888 个 / 98,876,695,253 B（100.0%） |
| 无 sha256 | 0 个 |
| pod 内同 sha 重复组 | 164 组 |
| 仅存在于 pod、尚无外部副本 | 2594 个 / 44,003,211,865 B（40.98 GiB） |
| 被现役配置/脚本按绝对路径引用 | 109 个 |

每个条目带 8 个字段：绝对路径、字节数、pod 端 mtime(UTC)、完整 sha256、分组、类别、是否被现役引用、副本状态（pod 内副本 / 外部副本 / 校验强度）。逐条目原始记录在 `backup_manifest_v2.json` 的 `entries`（3888 条）与 `manifest_rows.tsv`。

## 2 类别覆盖

| 类别 | 文件数 | 字节 | 有 sha256 | 仅 pod 字节 |
|---|---|---|---|---|
| 最终模型 | 3 | 33,353,932,977 | 3 | 22,235,955,318 |
| 上游基础模型（可从外部源重建） | 8 | 25,319,934,550 | 8 | 0 |
| 训练恢复状态 | 81 | 14,141,065,741 | 81 | 14,079,428,257 |
| 科研数据（数据集与派生数组） | 10 | 12,856,767,916 | 10 | 7,330,485 |
| 科研数据（新轨迹语料，未转换原始帧） | 87 | 5,064,960,376 | 87 | 5,064,960,376 |
| 科研数据（原始 100 条专家示范） | 56 | 4,306,511,889 | 56 | 0 |
| 临时测试产物（诊断小跑 checkpoint） | 45 | 1,665,815,333 | 45 | 493,746,884 |
| 科研数据（新轨迹语料） | 236 | 1,587,533,083 | 236 | 1,586,945,134 |
| 临时测试产物 | 20 | 504,996,502 | 20 | 504,992,766 |
| 复现材料（代码） | 1226 | 35,565,592 | 1226 | 488,306 |
| 科研数据（评估结果） | 154 | 20,238,093 | 154 | 11,417,603 |
| 复现材料（审计脚本与中间产物） | 1095 | 11,044,820 | 1095 | 10,816,762 |
| 复现材料（配置与日志） | 57 | 3,642,434 | 57 | 3,635,442 |
| 复现材料（上游代码） | 784 | 3,504,251 | 784 | 3,494,532 |
| 复现材料（环境、commit 与工作区 diff） | 26 | 1,181,696 | 26 | 0 |

> **历史限制（必须随备份一起引用）**：r4/r5 训练末态的 proprio_encoder 权重（2 张量 / 36,864 元素 / 147,456 B 张量字节）在 pod 上没有任何副本，本次备份不能恢复它；备份覆盖的是 LoRA adapter、优化器/调度器/RNG 状态与 r3 基座，因此历史评估组合可复现，训练末态模型不可复现。

## 3 目录级汇总（前 3 层路径）

| 目录 | 文件数 | 字节 | 其中仅 pod | 类别 |
|---|---|---|---|---|
| `FasterWAM/checkpoints/Wan-AI` | 6 | 14,202,213,669 | 0 | 上游基础模型（可从外部源重建） |
| `a3_data/a3_fw/frames.dat` | 1 | 12,849,371,136 | 0 | 科研数据（数据集与派生数组） |
| `a3_train_runs/a3_lora_r3/checkpoints` | 1 | 11,179,613,340 | 11,179,613,340 | 训练恢复状态 |
| `a3_train_runs/a3_lora_r1/step_002000_merged.pt` | 1 | 11,117,977,659 | 0 | 最终模型 |
| `a3_train_runs/a3_lora_r2/step_003000_merged.pt` | 1 | 11,117,977,659 | 11,117,977,659 | 最终模型 |
| `a3_train_runs/a3_lora_r3/step_003000_merged.pt` | 1 | 11,117,977,659 | 11,117,977,659 | 最终模型 |
| `FasterWAM/checkpoints/fasterwam_release` | 2 | 11,117,720,881 | 0 | 上游基础模型（可从外部源重建） |
| `fasterwam_a3_bridge/bridge/outputs` | 212 | 5,074,359,908 | 5,071,867,229 | 复现材料（代码）；科研数据（新轨迹语料，未转换原始帧）；科研数据（评估结果） |
| `data/a3_front_close_left_100/data` | 50 | 4,305,930,976 | 0 | 科研数据（原始 100 条专家示范） |
| `a3_train_runs/a3_lora_r5/checkpoints` | 40 | 1,480,726,202 | 1,480,712,802 | 训练恢复状态 |
| `a3_train_runs/a3_lora_r4/checkpoints` | 40 | 1,480,726,199 | 1,419,102,115 | 训练恢复状态 |
| `a3_data/a3_fw_dagger_near/frames.dat` | 1 | 917,618,688 | 917,618,688 | 科研数据（新轨迹语料） |
| `a3_data/a3_fw_dagger/frames.dat` | 1 | 652,689,408 | 652,689,408 | 科研数据（新轨迹语料） |
| `a3_train_runs/chk4_S/checkpoints` | 10 | 370,181,141 | 185,105,128 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_train_runs/a3_lora_r4_crashed_step500/checkpoints` | 5 | 185,090,772 | 123,478,413 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_train_runs/chk3_A/checkpoints` | 5 | 185,090,573 | 14,557 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_train_runs/chk3_A2/checkpoints` | 5 | 185,090,573 | 14,557 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_train_runs/chk3_B2/checkpoints` | 5 | 185,090,573 | 14,557 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_train_runs/chk3_B3/checkpoints` | 5 | 185,090,573 | 14,557 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_train_runs/chk4_N/checkpoints` | 5 | 185,090,570 | 14,557 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_train_runs/chk3_B/checkpoints` | 5 | 185,090,558 | 185,090,558 | 临时测试产物（诊断小跑 checkpoint） |
| `a3_eval/light_state_test/light` | 5 | 184,941,311 | 184,939,443 | 临时测试产物 |
| `a3_eval/light_state_test2/light` | 5 | 184,941,311 | 184,939,443 | 临时测试产物 |
| `a3_eval/light_state_test/adapter.pt` | 1 | 61,595,425 | 61,595,425 | 临时测试产物 |
| `a3_eval/light_state_test2/adapter.pt` | 1 | 61,595,425 | 61,595,425 | 临时测试产物 |
| `a3_dual_arm_sim/assets/a3` | 24 | 28,633,557 | 0 | 复现材料（代码） |
| `a3.tar.gz` | 1 | 9,608,636 | 9,608,636 | 临时测试产物 |
| `_r4_stage/expert_force.npz` | 1 | 5,293,712 | 5,293,712 | 复现材料（审计脚本与中间产物） |
| `a3_data/a3_fw_force/arrays.npz` | 1 | 5,243,625 | 5,243,625 | 科研数据（数据集与派生数组） |
| `r2.log` | 1 | 2,160,470 | 2,160,470 | 复现材料（配置与日志） |
| `a3_data/a3_fw/arrays.npz` | 1 | 2,070,331 | 2,070,331 | 科研数据（数据集与派生数组） |
| `a3_eval/r3_c0_diag_600/rep0` | 6 | 2,025,388 | 541,035 | 科研数据（评估结果） |
| `fasterwam-a3-cookie/finetune/logs` | 16 | 1,570,060 | 0 | 复现材料（代码） |
| `repo_a3c.tgz` | 1 | 1,357,984 | 1,357,984 | 临时测试产物 |
| `FasterWAM/third_party/RoboTwin` | 732 | 1,352,896 | 1,352,896 | 复现材料（上游代码） |
| `_r6_stage/out/repro` | 26 | 1,181,696 | 0 | 复现材料（环境、commit 与工作区 diff） |
| `a3_data/text_embeds_cache/b6f1657eeb7e8f5a4e4310074fab14ad8d79ed3ef798f677be2c137817d0f7b6.t5_len128.wan22ti2v5b.pt` | 1 | 1,051,869 | 1,051,869 | 复现材料（审计脚本与中间产物） |
| `fw.tar.gz` | 1 | 945,358 | 945,358 | 临时测试产物 |
| `FasterWAM/train.log` | 1 | 943,101 | 943,101 | 复现材料（上游代码） |
| `fasterwam-a3-cookie/.git/objects` | 384 | 939,958 | 0 | 复现材料（代码） |
| `smoke_r2.log` | 1 | 705,608 | 705,608 | 复现材料（配置与日志） |
| `a3_eval/demos_wrongclosure_v6b/expert_wrongclosure_7042.npz` | 1 | 683,248 | 683,248 | 科研数据（新轨迹语料） |
| `a3_eval/demos_wrongclosure_v6b/expert_wrongclosure_7038.npz` | 1 | 655,279 | 655,279 | 科研数据（新轨迹语料） |
| `FasterWAM/src/fasterwam` | 55 | 637,417 | 0 | 复现材料（上游代码）；复现材料（代码） |
| `a3_dual_arm_sim/src/a3_dual_arm_sim` | 87 | 631,807 | 0 | 复现材料（代码） |
| `data/a3_front_close_left_100/meta` | 5 | 578,409 | 0 | 科研数据（原始 100 条专家示范） |
| `a3_eval/r3_c0_entry_check/lora_C0_11.npz` | 1 | 536,224 | 0 | 科研数据（评估结果） |
| `a3_eval/pilot_single_E_rows_lift0012_f045/expert_single_6001.npz` | 1 | 513,836 | 513,836 | 科研数据（新轨迹语料） |
| `a3_eval/demos_wrongclosure_v6b/expert_wrongclosure_7041.npz` | 1 | 507,195 | 507,195 | 科研数据（新轨迹语料） |
| `a3_eval/demos_wrongclosure_v6b/expert_wrongclosure_7037.npz` | 1 | 474,439 | 474,439 | 科研数据（新轨迹语料） |
| `a3_eval/r4_b0_c0_paired/lora_C0_22.npz` | 1 | 437,106 | 0 | 科研数据（评估结果） |
| `a3_eval/r4_c0_frozen/lora_C0_22.npz` | 1 | 437,106 | 0 | 科研数据（评估结果） |
| `a3_eval/r4_c0_seed22_video/lora_C0_22.npz` | 1 | 437,106 | 0 | 科研数据（评估结果） |
| `a3_eval/r4_b0_c0_paired/lora_B0_22.npz` | 1 | 421,793 | 421,793 | 科研数据（评估结果） |
| `a3_eval/demos_correct_c1/expert_correct_7211.npz` | 1 | 408,064 | 408,064 | 科研数据（新轨迹语料） |
| `a3_eval/r5_b0_c0_paired/lora_C0_11.npz` | 1 | 380,929 | 380,929 | 科研数据（评估结果） |
| `a3_eval/r5_b0_c0_paired/lora_B0_11.npz` | 1 | 373,211 | 373,211 | 科研数据（评估结果） |
| `a3_eval/demos_wrongclosure_v6b/expert_wrongclosure_7039.npz` | 1 | 372,199 | 372,199 | 科研数据（新轨迹语料） |
| `a3_eval/r4_b0_c0_paired/lora_B0_11.npz` | 1 | 370,730 | 0 | 科研数据（评估结果） |
| `a3_eval/r4_b0_c0_paired/lora_C0_11.npz` | 1 | 370,730 | 0 | 科研数据（评估结果） |

## 4 尚无外部副本的大文件（≥50 MiB，必须先传到 pod 外才算备份完成）

| 路径 | 字节 | mtime(UTC) | SHA256（pod 端） | 类别 |
|---|---|---|---|---|
| `/workspace/a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt` | 11,179,613,340 | 2026-10-30T19:19:11Z | `e45110fd6742a72a998aa853…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r2/step_003000_merged.pt` | 11,117,977,659 | 2026-10-30T16:44:30Z | `baafffcf9aacfc7a8a1e9276…` | 最终模型 |
| `/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt` | 11,117,977,659 | 2026-10-30T19:20:36Z | `1ac089bf0b4ec41441be274d…` | 最终模型 |
| `/workspace/a3_data/a3_fw_dagger_near/frames.dat` | 917,618,688 | 2026-10-31T03:38:27Z | `165c985f546f8b59f1b12860…` | 科研数据（新轨迹语料） |
| `/workspace/a3_data/a3_fw_dagger/frames.dat` | 652,689,408 | 2026-10-31T00:15:18Z | `6e1d053f213336224a2f0d1b…` | 科研数据（新轨迹语料） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s70/frames.dat` | 270,950,400 | 2026-10-31T03:28:40Z | `558bbba0f2c4abcb4aad7bc9…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s80/frames.dat` | 270,950,400 | 2026-10-31T03:30:00Z | `0bca02c173a2e009f5c17d4a…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s85/frames.dat` | 270,950,400 | 2026-10-31T03:31:22Z | `73cee12c52d00fca1604f9c1…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed11_s506/frames.dat` | 270,950,400 | 2026-10-31T03:13:39Z | `d0408cc593271deaee18dc66…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed11_s530/frames.dat` | 270,950,400 | 2026-10-31T03:15:34Z | `3e7f216608a45758caf0ad72…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed21_s190/frames.dat` | 270,950,400 | 2026-10-31T03:18:01Z | `a47f51d3a3219119489a65e3…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed21_s226/frames.dat` | 270,950,400 | 2026-10-31T03:18:01Z | `9c07101fe7adc75df2481863…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger/seed11_s90/frames.dat` | 210,739,200 | 2026-10-31T00:13:43Z | `7a95412231b4df220071939f…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger/seed21_s76/frames.dat` | 210,739,200 | 2026-10-31T00:12:08Z | `7c90a5caced2cb146e126845…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s40/frames.dat` | 210,739,200 | 2026-10-31T00:06:45Z | `f1907ae8c54ba9956c4110a7…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s60/frames.dat` | 210,739,200 | 2026-10-31T00:08:12Z | `a6576957633a22180368f0dd…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s70/frames.dat` | 210,739,200 | 2026-10-31T00:10:32Z | `293d24285faf63422e7a1077…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s150/frames.dat` | 210,739,200 | 2026-10-31T03:19:06Z | `4462b6e0b13f82f50f12ecfc…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s160/frames.dat` | 210,739,200 | 2026-10-31T03:20:08Z | `2739f1bf6cf5aef5b4e4f708…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s170/frames.dat` | 210,739,200 | 2026-10-31T03:22:32Z | `54688a013bbe41df8d2ee321…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s570/frames.dat` | 210,739,200 | 2026-10-31T00:43:12Z | `25e59a646b88269c53e4681b…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s579/frames.dat` | 210,739,200 | 2026-10-31T01:00:58Z | `7c1a9950cc14df71b622bac5…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed21_s214/frames.dat` | 210,739,200 | 2026-10-31T00:37:49Z | `014fa1902cbe6e4a62ee7d14…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed21_s223/frames.dat` | 210,739,200 | 2026-10-31T01:03:17Z | `925e0f3ef2cd56eba8385d56…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed22_s56/frames.dat` | 210,739,200 | 2026-10-31T01:04:53Z | `d727f2c8ef438bf874b47710…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed22_s68/frames.dat` | 210,739,200 | 2026-10-31T01:06:14Z | `31b5a2c6659e072ab6e6fee9…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near/seed22_s72/frames.dat` | 210,739,200 | 2026-10-31T01:08:49Z | `a25da23806ba5ef37066454b…` | 科研数据（新轨迹语料，未转换原始帧） |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/optimizer.pt` | 123,478,413 | 2026-10-31T13:52:11Z | `e8e9d350c412019daca8feec…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/optimizer.pt` | 123,478,413 | 2026-10-31T20:31:13Z | `6e517a87cbb3baa634f743cb…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_000500/optimizer.pt` | 123,478,413 | 2026-10-31T11:33:58Z | `dc5df84bc67b17dbcffcc5d1…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_001000/optimizer.pt` | 123,478,413 | 2026-10-31T11:53:31Z | `45feaa328249942c05468ae4…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_001500/optimizer.pt` | 123,478,413 | 2026-10-31T12:13:17Z | `96786acbad044b7ab3bf2fc2…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_002000/optimizer.pt` | 123,478,413 | 2026-10-31T12:32:51Z | `877f047c6ac7254b3bd0fa88…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_002500/optimizer.pt` | 123,478,413 | 2026-10-31T12:52:15Z | `36ab1246e48056008c40d507…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_003000/optimizer.pt` | 123,478,413 | 2026-10-31T13:11:48Z | `1f55386d623368fdb121d0e0…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_003500/optimizer.pt` | 123,478,413 | 2026-10-31T13:31:22Z | `32d4fd428c231d6dcc2dcea1…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4_crashed_step500/checkpoints/light/step_000500/optimizer.pt` | 123,478,413 | 2026-10-31T10:57:19Z | `61cbf4d3043a728fdb463a86…` | 临时测试产物（诊断小跑 checkpoint） |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/optimizer.pt` | 123,478,413 | 2026-10-31T18:09:32Z | `7f3ad5644e73d3d438ffc2a3…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_005000/optimizer.pt` | 123,478,413 | 2026-10-31T18:29:18Z | `e7c5780d1f6dd3889a0a9f63…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_005500/optimizer.pt` | 123,478,413 | 2026-10-31T18:49:06Z | `1b9838f01847d06db4c42997…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_006000/optimizer.pt` | 123,478,413 | 2026-10-31T19:09:30Z | `7452e2cc06e296d578e0c5b8…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_006500/optimizer.pt` | 123,478,413 | 2026-10-31T19:29:27Z | `87c153fdcc4b3a8b8e01c01e…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_007000/optimizer.pt` | 123,478,413 | 2026-10-31T19:49:42Z | `5caca47bf9e6bf0d3e7d5b63…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_007500/optimizer.pt` | 123,478,413 | 2026-10-31T20:10:59Z | `e9395de59259396543529fac…` | 训练恢复状态 |
| `/workspace/a3_train_runs/chk3_B/checkpoints/light/step_000005/optimizer.pt` | 123,478,413 | 2026-10-31T10:16:05Z | `a89462ab29e85064cf3ebd50…` | 临时测试产物（诊断小跑 checkpoint） |
| `/workspace/a3_train_runs/chk4_S/checkpoints/light/step_000003/optimizer.pt` | 123,478,413 | 2026-10-31T11:11:09Z | `68aa667010d1a87acbd00ba6…` | 临时测试产物（诊断小跑 checkpoint） |
| `/workspace/a3_eval/light_state_test/light/optimizer.pt` | 123,329,333 | 2026-10-31T07:11:38Z | `d417dc67eafc4f06e145707b…` | 临时测试产物 |
| `/workspace/a3_eval/light_state_test2/light/optimizer.pt` | 123,329,333 | 2026-10-31T07:26:37Z | `3463f8a97bb557b9547f9d9c…` | 临时测试产物 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/adapter.pt` | 61,595,489 | 2026-10-31T13:52:10Z | `e8372c333eb2e454dd91c79f…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/adapter.pt` | 61,595,489 | 2026-10-31T20:31:13Z | `af6bba85d1bd99c14d2547c8…` | 训练恢复状态 |
| `/workspace/a3_eval/light_state_test/light/adapter.pt` | 61,595,489 | 2026-10-31T07:11:37Z | `a02a5a45bbcd09b663cadb8b…` | 临时测试产物 |
| `/workspace/a3_eval/light_state_test2/light/adapter.pt` | 61,595,489 | 2026-10-31T07:26:36Z | `7c2955e43a281296509dc43e…` | 临时测试产物 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_001000/adapter.pt` | 61,595,489 | 2026-10-31T11:53:30Z | `12e311b092ea98a05d4b88d5…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_001500/adapter.pt` | 61,595,489 | 2026-10-31T12:13:16Z | `654716208595d0a23af0b72a…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_002000/adapter.pt` | 61,595,489 | 2026-10-31T12:32:51Z | `8fdf3f801dafb0aa0140fc6e…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_002500/adapter.pt` | 61,595,489 | 2026-10-31T12:52:15Z | `46220840ab80fa6c977ada68…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_003000/adapter.pt` | 61,595,489 | 2026-10-31T13:11:48Z | `4d950bfcc0ef06a77deaff4c…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_003500/adapter.pt` | 61,595,489 | 2026-10-31T13:31:22Z | `45e8c5093178a0a9722cbd0e…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/adapter.pt` | 61,595,489 | 2026-10-31T18:09:31Z | `84287f759eea6cabafdc64b2…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_005000/adapter.pt` | 61,595,489 | 2026-10-31T18:29:18Z | `674c4ec8f9117a8e49c4db2b…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_005500/adapter.pt` | 61,595,489 | 2026-10-31T18:49:05Z | `23413858cf99d967bf27800f…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_006000/adapter.pt` | 61,595,489 | 2026-10-31T19:09:30Z | `efddc50151dfd63783541453…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_006500/adapter.pt` | 61,595,489 | 2026-10-31T19:29:27Z | `ea477baf2fee3f2ceac9adcf…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_007000/adapter.pt` | 61,595,489 | 2026-10-31T19:49:42Z | `92f9fa9803b0e0f67f24b391…` | 训练恢复状态 |
| `/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_007500/adapter.pt` | 61,595,489 | 2026-10-31T20:10:59Z | `0f4785491993269d2313c174…` | 训练恢复状态 |
| `/workspace/a3_train_runs/chk3_B/checkpoints/light/step_000005/adapter.pt` | 61,595,489 | 2026-10-31T10:16:05Z | `c0b99bba28ad8516760cec8c…` | 临时测试产物（诊断小跑 checkpoint） |
| `/workspace/a3_train_runs/chk4_S/checkpoints/light/step_000003/adapter.pt` | 61,595,489 | 2026-10-31T11:11:08Z | `6a82b9efa3c9953ab4736feb…` | 临时测试产物（诊断小跑 checkpoint） |
| `/workspace/a3_eval/light_state_test/adapter.pt` | 61,595,425 | 2026-10-31T07:10:14Z | `2a67ce172c0cae4aed52a00e…` | 临时测试产物 |
| `/workspace/a3_eval/light_state_test2/adapter.pt` | 61,595,425 | 2026-10-31T07:25:10Z | `8b98552461fb50452cd627e2…` | 临时测试产物 |

共 69 个，43,929,871,208 B（40.91 GiB）。

## 5 与 v1 的差异

- v1 的 963 条全部保留；新增 2,925 条。
- 新增中最大的一块是上游基础模型 `FasterWAM/checkpoints/`（25.32 GB，8 个文件）——它不属于科研成果，可外部重建，不需要占用外存预算，但已入清单并给出 sha256 以便重建后比对。
- 新发现 `a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt`（11,179,613,340 B）是 r3 的完整训练状态（模型+优化器），v1 漏掉了它；这是 r4/r5 的训练基座，目前只有 pod 一份，属于最高优先级外存对象。
- `a3_train_runs/*/checkpoints/state` 全部为空目录（0 个文件）：pod 上根本不存在独立的 sampler/dataloader 状态文件，任何"可断点续训"的说法都不包含采样器状态。
- r4/r5 的 `checkpoints/light` 每个 run 有 8 个中间 step（40 个文件），v1 只记录了最后一步；诊断小跑 chk3_A/A2/B/B2/B3、chk4_N/S、r4_crashed_step500 此前完全未入清单。

## 6 pod 内部同 sha256 重复组

同内容多副本意味着"pod 内复制"不等于"pod 外备份"。共 164 组，完整列表见 `backup_manifest_v2.json` 的 `duplicate_groups`。前 25 组：

- `e3b0c44298fc1c14`（26 份）: `FasterWAM/src/fasterwam/datasets/__init__.py`，`FasterWAM/src/fasterwam/models/wan22/__init__.py`，`_r6_stage/out/repro/cookie_stash.txt`，`FasterWAM/ds_check.log`，`FasterWAM/third_party/RoboTwin/policy/__init__.py`，`FasterWAM/third_party/RoboTwin/script/__init__.py`
- `71b37148bfd83bd8`（10 份）: `a3_eval/demos_single_n0/expert_single_7101.npz`，`a3_eval/demos_single_n0/expert_single_7102.npz`，`a3_eval/demos_single_n0/expert_single_7103.npz`，`a3_eval/demos_single_n0/expert_single_7104.npz`，`a3_eval/demos_single_n0/expert_single_7105.npz`，`a3_eval/demos_single_n0/expert_single_7106.npz`
- `3307c9bb122f7709`（5 份）: `_r6_stage/out/repro/fasterwam-a3-cookie_branch.txt`，`_r6_stage/out/repro/fasterwam-a3-cookie_head.txt`，`_r6_stage/out/repro/fasterwam-a3-cookie_lastcommit.txt`，`_r6_stage/out/repro/fasterwam-a3-cookie_remote.txt`，`_r6_stage/out/repro/fasterwam-a3-cookie_status.txt`
- `ed833168725b0b63`（4 份）: `a3_data/a3_fw/dataset_stats.json`，`a3_data/a3_fw_dagger/dataset_stats.json`，`a3_data/a3_fw_dagger_near/dataset_stats.json`，`fasterwam-a3-cookie/data/dataset_stats.json`
- `7a41a41354ab8049`（4 份）: `FasterWAM/.python-version`，`FasterWAM/environments/libero/.python-version`，`FasterWAM/environments/libero-plus/.python-version`，`FasterWAM/environments/robotwin/.python-version`
- `048af8fe6f801dc6`（4 份）: `a3_train_runs/chk3_A/checkpoints/light/step_000010/scheduler.pt`，`a3_train_runs/chk3_A2/checkpoints/light/step_000010/scheduler.pt`，`a3_train_runs/chk3_B2/checkpoints/light/step_000010/scheduler.pt`，`a3_train_runs/chk3_B3/checkpoints/light/step_000010/scheduler.pt`
- `360ff1c41e7fb8a7`（4 份）: `a3_train_runs/chk3_A/checkpoints/light/step_000010/trainer_state.json`，`a3_train_runs/chk3_A2/checkpoints/light/step_000010/trainer_state.json`，`a3_train_runs/chk3_B2/checkpoints/light/step_000010/trainer_state.json`，`a3_train_runs/chk3_B3/checkpoints/light/step_000010/trainer_state.json`
- `37c4520f1d952e36`（3 份）: `a3_eval/r3_c0_diag_600/rep0/lora_C0_11.npz`，`a3_eval/r3_c0_entry_check/lora_C0_11.npz`，`fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_11.npz`
- `b9f4e160b17d4be3`（3 份）: `a3_eval/r4_b0_c0_paired/lora_B0_11.npz`，`a3_eval/r4_b0_c0_paired/lora_C0_11.npz`，`a3_eval/r4_c0_frozen/lora_C0_11.npz`
- `c550d261d5d00e18`（3 份）: `a3_eval/r4_b0_c0_paired/lora_C0_21.npz`，`a3_eval/r4_c0_frozen/lora_C0_21.npz`，`a3_eval/r4_c0_seed21_video/lora_C0_21.npz`
- `afac1ef1df3375eb`（3 份）: `a3_eval/r4_b0_c0_paired/lora_C0_22.npz`，`a3_eval/r4_c0_frozen/lora_C0_22.npz`，`a3_eval/r4_c0_seed22_video/lora_C0_22.npz`
- `a18b207c137fd455`（3 份）: `FasterWAM/src/fasterwam/light_state.py`，`_r4_stage/light_state.new.py`，`fasterwam-a3-cookie/finetune/light_state.py`
- `3b9f206bf9d8b983`（3 份）: `_r6_stage/out/repro/FasterWAM_head.txt`，`_r6_stage/out/repro/a3_dual_arm_sim_head.txt`，`_r6_stage/out/repro/fasterwam_a3_bridge_head.txt`
- `0ea7cf1d885ad497`（3 份）: `a3_eval/disk_job.log`，`dagger_logs/near_runner.log`，`dagger_logs/resume_runner.log`
- `e9289d43b25fd898`（3 份）: `a3_train_runs/a3_lora_r4/checkpoints/light/step_000500/scheduler.pt`，`a3_train_runs/a3_lora_r4_crashed_step500/checkpoints/light/step_000500/scheduler.pt`，`a3_train_runs/a3_lora_r5/checkpoints/light/step_004500/scheduler.pt`
- `a1cc42189446991c`（3 份）: `expert_seed11.npz`，`expert_seed21.npz`，`expert_seed22.npz`
- `f470a84f04d9cf0e`（2 份）: `a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/scheduler.pt`，`a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/scheduler.pt`
- `434173eff9a8152d`（2 份）: `a3_train_runs/a3_lora_r4/config.yaml`，`a3_train_runs/a3_lora_r4_crashed_step500/config.yaml`
- `760f334c5d4d3992`（2 份）: `a3_eval/r3_c0_diag_600/force.json`，`a3_eval/r4_c0_frozen/cmp/r3_force_recheck.json`
- `999ecaaa3d309797`（2 份）: `a3_eval/r3_c0_diag_600/lift.json`，`a3_eval/r4_c0_frozen/cmp/r3_lift_recheck.json`
- `9019656b4999418a`（2 份）: `a3_eval/r3_c0_diag_600/rep0/lora_C0_21.npz`，`fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_21.npz`
- `6436ee92514716fd`（2 份）: `a3_eval/r3_c0_diag_600/rep0/lora_C0_22.npz`，`fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_22.npz`
- `4641e6963285d401`（2 份）: `a3_eval/r3_c0_diag_det/rep0/lora_C0_21.npz`，`a3_eval/r3_c0_diag_det/rep1/lora_C0_21.npz`
- `18cd7a50462ed8da`（2 份）: `a3_eval/r3_c0_diag_smoke/rep0/lora_C0_11.npz`，`_r4_stage/parity_entry/lora_C0_11.npz`
- `a8d2f7174d0b2e1a`（2 份）: `a3_eval/r4_b0_c0_paired/lora_B0_11_front.mp4`，`a3_eval/r4_b0_c0_paired/lora_C0_11_front.mp4`