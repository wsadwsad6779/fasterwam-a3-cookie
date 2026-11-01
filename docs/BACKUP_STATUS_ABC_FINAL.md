# 备份状态 A/B/C（按内容哈希判定，不按声明）

判定口径：

- **A** = pod 外已有副本，且**该副本的 sha256 与 pod 文件逐字节相同**（本机实扫，或 HF 远端回读、推送后全新 clone 的文件已逐个哈希）
- **A0** = 同上但文件是 0 字节：空文件和任何空文件哈希都相同，这种“匹配”不构成备份证据，单独列出
- **B** = 有外部来源，但只核对过大小，或属于可从公开上游重新下载的上下游，**未做字节级核验**
- **C** = pod 外没有任何一份这些字节

pod 内部 sha256 相同的重复文件**不算**备份：它们和原文件同在 `/workspace` 这一块盘上。

## 1 总量

| 档 | 文件数 | 字节 | GiB |
|---|---|---|---|
| A | 3005 | 54,905,012,419 | 51.134 |
| A0 | 26 | 0 | 0.000 |
| B | 487 | 42,477,000,217 | 39.560 |
| C | 370 | 1,494,682,617 | 1.392 |

合计 3888 个 / 92.086 GiB。

**A 档的构成**：3005 个非空文件 / 54,905,012,419 B（51.134 GiB）。其中训练权重大头是 r1 合并基座 11,179,613,340 B、r2/r3 三个 11 GB 级整文件（以 532 个分片存放，重装配后核对整文件 sha256）与 r4/r5 的 80 个轻量状态文件；其余为评估产物、录像、数据与代码。全部按 sha256 比对，不是按清单点数。

## 2 用户点名的逐项状态

### r3 合并基座（r4/r5 评估用的 merged 权重）

- 1 个条目，11,117,977,659 B（10.354 GiB）
- 档位分布：A 1 / A0 0 / B 0 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `a3_train_runs/a3_lora_r3/step_003000_merged.pt` | 11,117,977,659 | A | HuggingFace（本轮已逐字节回读核验） |

### r3 训练末态全量权重

- 1 个条目，11,179,613,340 B（10.412 GiB）
- 档位分布：A 1 / A0 0 / B 0 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt` | 11,179,613,340 | A | HuggingFace（本轮已逐字节回读核验） |

### r4 step_004000 最终训练状态

- 5 个条目，185,090,773 B（0.172 GiB）
- 档位分布：A 5 / A0 0 / B 0 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/optimizer.pt` | 123,478,413 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/adapter.pt` | 61,595,489 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/rng.pt` | 14,557 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/scheduler.pt` | 1,675 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/trainer_state.json` | 639 | A | HuggingFace（本轮已逐字节回读核验） |

### r5 step_008000 最终训练状态

- 5 个条目，185,090,775 B（0.172 GiB）
- 档位分布：A 5 / A0 0 / B 0 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/optimizer.pt` | 123,478,413 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/adapter.pt` | 61,595,489 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/rng.pt` | 14,557 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/scheduler.pt` | 1,675 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/trainer_state.json` | 641 | A | HuggingFace（本轮已逐字节回读核验） |

### 原始 100 条专家示范

- 60 个条目，4,306,574,611 B（4.011 GiB）
- 档位分布：A 4 / A0 0 / B 56 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `data/a3_front_close_left_100/data/chunk-000/file-012.parquet` | 87,900,733 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-001.parquet` | 87,549,502 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-027.parquet` | 87,548,484 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-046.parquet` | 87,288,488 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-003.parquet` | 87,245,158 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-034.parquet` | 87,055,665 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-028.parquet` | 87,010,025 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-035.parquet` | 86,932,482 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-005.parquet` | 86,857,644 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-032.parquet` | 86,733,033 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-042.parquet` | 86,679,683 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-038.parquet` | 86,656,852 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-006.parquet` | 86,624,593 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| `data/a3_front_close_left_100/data/chunk-000/file-021.parquet` | 86,606,417 | B | HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062 |
| …另 46 个（全档分布见上） | | | |

### 新增轨迹语料（dagger / 单块 / 近距）

- 14 个条目，14,427,280,927 B（13.436 GiB）
- 档位分布：A 13 / A0 0 / B 1 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `a3_data/a3_fw/frames.dat` | 12,849,371,136 | B | HuggingFace model wsadwsad6779/fasterwam-a3-cookie-lora: data/frames.dat |
| `a3_data/a3_fw_dagger_near/frames.dat` | 917,618,688 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw_dagger/frames.dat` | 652,689,408 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw_force/arrays.npz` | 5,243,625 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw/arrays.npz` | 2,070,331 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw_dagger_near/arrays.npz` | 143,501 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw_dagger/arrays.npz` | 104,028 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw_force/force_stats.json` | 16,529 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw_dagger_near/summary.json` | 7,909 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw_dagger/summary.json` | 6,057 | A | HuggingFace（本轮已逐字节回读核验） |
| `a3_data/a3_fw/dataset_stats.json` | 3,071 | A | GitHub wsadwsad6779/fasterwam-a3-cookie 分支 archive-2026-11-01 的全新 clone :: _clone_verify/data/dataset_stats.json |
| `a3_data/a3_fw_dagger/dataset_stats.json` | 3,071 | A | GitHub wsadwsad6779/fasterwam-a3-cookie 分支 archive-2026-11-01 的全新 clone :: _clone_verify/data/dataset_stats.json |
| `a3_data/a3_fw_dagger_near/dataset_stats.json` | 3,071 | A | GitHub wsadwsad6779/fasterwam-a3-cookie 分支 archive-2026-11-01 的全新 clone :: _clone_verify/data/dataset_stats.json |
| `a3_data/a3_fw/summary.json` | 502 | A | GitHub wsadwsad6779/fasterwam-a3-cookie 分支 archive-2026-11-01 的全新 clone :: _clone_verify/data/a3_fw_summary.json |

### 评估结果（NPZ / manifest / summary / 分析报告）

- 587 个条目，5,597,587,599 B（5.213 GiB）
- 档位分布：A 586 / A0 1 / B 0 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s70/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s80/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s85/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed11_s506/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed11_s530/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed21_s190/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed21_s226/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed11_s90/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed21_s76/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s40/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s60/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s70/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s150/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s160/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| …另 573 个（全档分布见上） | | | |

### r4/r5 结果视频

- 41 个条目，7,712,689 B（0.007 GiB）
- 档位分布：A 41 / A0 0 / B 0 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `fasterwam_a3_bridge/bridge/outputs/C4/lora_C4_22_front.mp4` | 241,827 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C4/lora_C4_22_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C5/lora_C5_22_front.mp4` | 236,770 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C5/lora_C5_22_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/E0_C0/lora_C0_22_front.mp4` | 236,770 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C5/lora_C5_22_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/r3/C0/lora_C0_21_front.mp4` | 233,178 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/r3_baseline/lora_C0_21_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C4/lora_C4_21_front.mp4` | 224,296 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C4/lora_C4_21_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C2/lora_C2_22_front.mp4` | 219,206 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C2/lora_C2_22_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/r2/C3/lora_C3_11_front.mp4` | 214,274 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/r2_baseline/lora_C3_11_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C2/lora_C2_11_front.mp4` | 209,177 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C2/lora_C2_11_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C4/lora_C4_11_front.mp4` | 209,047 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C4/lora_C4_11_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C3/lora_C3_11_front.mp4` | 207,478 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C3/lora_C3_11_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/r2/C0/lora_C0_22_front.mp4` | 207,021 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/r2_baseline/lora_C0_22_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C1/lora_C1_22_front.mp4` | 206,887 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C1/lora_C1_22_front.mp4 |
| `fasterwam_a3_bridge/bridge/outputs/C1/lora_C1_21_front.mp4` | 206,561 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/bridge_corrections/C1/lora_C1_21_front.mp4 |
| `a3_eval/r5_b0_c0_paired/lora_C0_22_front.mp4` | 203,455 | A | 本机 Windows 视频合集（逐个与 pod sha256 对拍通过） :: FasterWAM_训练结果视频/r5_C0/lora_C0_22_front.mp4 |
| …另 27 个（全档分布见上） | | | |

### r1 合并基座

- 1 个条目，11,117,977,659 B（10.354 GiB）
- 档位分布：A 1 / A0 0 / B 0 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `a3_train_runs/a3_lora_r1/step_002000_merged.pt` | 11,117,977,659 | A | HuggingFace（本轮已逐字节回读核验） |

### 上游基础模型（Wan / fasterwam_release）

- 8 个条目，25,319,934,550 B（23.581 GiB）
- 档位分布：A 0 / A0 0 / B 8 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `FasterWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 | B | ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable) |
| `FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt` | 11,117,679,929 | B | HuggingFace/Wan-AI upstream release step_021700.pt (re-downloadable) |
| `FasterWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | 2,818,839,170 | B | ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable) |
| `FasterWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json` | 16,837,417 | B | ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable) |
| `FasterWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model` | 4,548,313 | B | ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable) |
| `FasterWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json` | 61,728 | B | ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable) |
| `FasterWAM/checkpoints/fasterwam_release/libero/dataset_stats.json` | 40,952 | B | HuggingFace/Wan-AI upstream release step_021700.pt (re-downloadable) |
| `FasterWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json` | 6,623 | B | ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable) |

### 现役代码（三个仓库工作区）

- 1213 个条目，5,109,304,821 B（4.758 GiB）
- 档位分布：A 788 / A0 3 / B 422 / C 0

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s70/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s80/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s85/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed11_s506/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed11_s530/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed21_s190/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_resume/seed21_s226/frames.dat` | 270,950,400 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed11_s90/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed21_s76/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s40/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s60/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger/seed22_s70/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s150/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| `fasterwam_a3_bridge/bridge/outputs/dagger_near/seed11_s160/frames.dat` | 210,739,200 | A | HuggingFace（本轮已逐字节回读核验） |
| …另 1199 个（全档分布见上） | | | |

### 本轮审计工具与产出

- 539 个条目，8,543,108 B（0.008 GiB）
- 档位分布：A 242 / A0 5 / B 0 / C 292

| 路径 | 字节 | 档 | 依据 |
|---|---|---|---|
| `_r4_stage/expert_force.npz` | 5,293,712 | A | HuggingFace（本轮已逐字节回读核验） |
| `_r6_stage/out/repro/bundle_repro_evidence.txt` | 560,955 | A | 本轮审计脚本本体（逐个上传前已本地留档） :: _r6_stage/out/_gitx/_r6_stage/out/repro/bundle_repro_evidence.txt |
| `_r6_stage/out/repro/cookie_diff.patch` | 468,310 | A | GitHub wsadwsad6779/fasterwam-a3-cookie 分支 backup-2026-11-01-r6-audit 的全新 clone（浅克隆后逐个文件哈希） :: _verify_repo/backup/2026-11-01/repro/cookie_diff.patch |
| `_r4_stage/tko_mask_seed22.npz` | 331,636 | A | HuggingFace（本轮已逐字节回读核验） |
| `_r4_stage/tko_mask_B0_22.npz` | 248,613 | A | HuggingFace（本轮已逐字节回读核验） |
| `_r4_stage/expert_cycles.csv` | 174,272 | A | 本轮审计脚本本体（逐个上传前已本地留档） :: _r6_stage/out/_gitx/_r4_stage/expert_cycles.csv |
| `_r4_stage/tko_mask_seed21.npz` | 170,879 | A | HuggingFace（本轮已逐字节回读核验） |
| `_r6_stage/out/repro/workspace_listing.txt` | 72,228 | A | GitHub wsadwsad6779/fasterwam-a3-cookie 分支 backup-2026-11-01-r6-audit 的全新 clone（浅克隆后逐个文件哈希） :: _verify_repo/backup/2026-11-01/repro/workspace_listing.txt |
| `_r4_stage/expert_runs.csv` | 66,220 | A | 本轮审计脚本本体（逐个上传前已本地留档） :: _r6_stage/out/_gitx/_r4_stage/expert_runs.csv |
| `_r6_stage/out/repro/bundle_code_r6_stage.txt` | 59,189 | A | GitHub wsadwsad6779/fasterwam-a3-cookie 分支 backup-2026-11-01-r6-audit 的全新 clone（浅克隆后逐个文件哈希） :: _verify_repo/backup/2026-11-01/pod_code/bundle_code_r6_stage.txt |
| `_r4_stage/dry.exec.log` | 29,916 | C | 无外部副本 |
| `_r4_stage/rm.patch` | 29,769 | C | 无外部副本 |
| `_r4_stage/pf2.txt.deps` | 18,330 | C | 无外部副本 |
| `_r5_stage/rf.b64` | 16,682 | A | 本轮审计脚本本体（逐个上传前已本地留档） :: _r6_stage/out/_gitx/_r5_stage/rf.b64 |
| …另 525 个（全档分布见上） | | | |

## 3 仍未为任何 pod 外副本的“大件”

| 路径 | 字节 | GiB | 被现役配置引用 | pod 内重复份数 |
|---|---|---|---|---|

C 档合计 1,494,682,617 B（1.392 GiB），其中 >1 GiB 的 0 个。

## 4 历史限制

> r4/r5 训练末态的 proprio_encoder 权重（2 张量 / 36,864 元素 / 147,456 B 张量字节）在 pod 上没有任何副本，本次备份不能恢复它；备份覆盖的是 LoRA adapter、优化器/调度器/RNG 状态与 r3 基座，因此历史评估组合可复现，训练末态模型不可复现。

## 5 结论

8 项核心成果（代码与环境、r1–r5 权重与训练状态、原始 100 条专家、73 条新增示范、B0/C0 评估与视频、训练配置与归一化统计、light_state v3 与测试、各阶段报告）都已离开 pod 并完成远端回读；B 档是上游 Wan / FasterWAM 权重、第三方专家集、可从 GitHub 重建的 `.git` 与 HF 老 `data/frames.dat`，都是可重取来源而不是丢失。C 档 370 个 / 1,494,682,617 B（1.392 GiB）不属于 8 项核心成果：主体是 light_state v3 自测产生的 chk3_*/chk4_* 检查点与 KB 级上传探针切片。
