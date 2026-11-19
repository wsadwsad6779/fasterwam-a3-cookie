# 实例销毁前的归档覆盖清单（2026-11-19/20）

pod：`rgapi-one-click-24316-57b6b870`（j-56nys3hbuvqgogybu7knegv6wx）。
原则：**能传就传**。除“公开可再下载的缓存”和“文件系统垃圾”外，全部外置。

## 1. 去向总表

| pod 路径 | 体积 | 去向 |
|---|---|---|
| `/workspace/eval_restore_20261117/reconstructed_normalizer/` | 53 G | HF 私有 `v2_repair_20261119/full/reconstructed_normalizer/`（已提交）+ 公开仓 `eval_restore_20261117/reconstructed_normalizer/`；脚本/JSON 同时进 GitHub |
| ├ `model_repair_runs/` | 49 G | 私有仓全镜像（含 `lora_300/step_000100.pt` 23.9 G 全状态、`smoke5/final.pt` 23.9 G、`action_expert_v2/` deltas） |
| ├ `teacher_batch_s22…s30/` | ≈1.9 G | 私有仓镜像 |
| ├ `forensics_full_wide_out/` | 160 M | 私有仓镜像 |
| └ `c_class_eval_v3_*/`、`c_class_eval_v4_*/` 等 | ≈40 M | 私有仓镜像 + 证据包；summary JSON 同时进 GitHub |
| `/workspace/eval_restore_20261117/reconstructed_normalizer`（精选） | 59 M 包 | 私有仓 `v2_repair_20261119/evidence/rn_text_bundle.tar.gz` |
| `/workspace/eval_restore_20261117/base_step_021700.pt` | 11.1 G | 公开仓 `eval_restore_20261117/`（上游公开，为完整性照传） |
| `/workspace/eval_restore_20261117/step_040000_merged.pt` | 11.1 G | 公开仓 `eval_restore_20261117/` |
| `/workspace/eval_restore_20261117/deploy/`、`archive_evidence/`、`step_040000/`、`_rescue_transfer/`、`auxiliary/`、顶层日志 | ≈100 M | 公开仓 `eval_restore_20261117/`；`archive_evidence.tar.gz`、`step_040000_light.tar.gz`、`workspace_top_small.tar.gz` 另存私有仓 |
| `/root/repair_runs/action_expert_v3/` | 11 G | 私有仓 `v2_repair_20261119/checkpoints/action_expert_v3/` |
| `/root/repair_runs/action_expert_v4/` | 8.4 G | 私有仓 `v2_repair_20261119/checkpoints/action_expert_v4/` |
| `/root/repair_runs/action_expert_v2/` | 3.2 G | 私有仓 `.../checkpoints/action_expert_v2/`；公开仓 `root_repair_runs/` |
| `/root/repair_runs/teacher_diverse_20261119/` | 2.6 G | 私有仓 `.../corpus/teacher_diverse_20261119/`；公开仓 `root_repair_runs/` |
| `/root/quarantine_lora300/step_000200.pt` | 16 G | 公开仓 `root_quarantine_lora300/`（已知损坏，不丢） |
| `/workspace/a3_data/text_embeds_cache/` | 8 K | 公开仓 `a3_data/` |
| `/workspace/_archive_*.py`、`_public_upload.py`、`_remote_stats.py`、`_verify_remote.py` 等归档脚本 | <1 M | 公开仓 `workspace_top_files.tar.gz` |

## 2. 明确跳过（附理由）

| pod 路径 | 体积 | 理由 |
|---|---|---|
| `/workspace/.cache/huggingface/` | 24 G | 公开模型/编码器下载缓存（T5、VAE、基座等），可随时重新下载；不属于项目独有数据 |
| `/workspace/lost+found/`、`__pycache__/`、`*.pyc` | <1 M | 文件系统垃圾 / 可再生成字节码 |
| `/root/.ssh`、`/root/.npm`、`/root/.opencode`、shell 历史 | — | 个人/工具配置，非实验资产；token 类凭据一律不外传 |

私有仓配额（约 90 GB）触发后，剩余内容**没有丢弃**：`eval_restore_20261117/` 全树、`root_repair_runs/`、`root_quarantine_lora300/`、`a3_data/`、顶层脚本包全部转存到公开仓 `fasterwam-a3-archive-public`。

## 3. GitHub（`main`）

- `README.md`：main 分支权威入口；
- `docs/ARCHITECTURE_20261119.md`：架构与契约；
- `docs/RESTORE_FROM_ARCHIVE_20261119.md`：恢复步骤；
- `docs/ARCHIVE_COVERAGE_20261119.md`：本文件；
- `restore_20261117/`：deploy 代码（129 文件）+ reconstructed_normalizer 脚本与证据（398 文件）。

## 4. HF 两个仓库

- 私有 `wsadwsad6779/fasterwam-a3-force-40k`：
  - `v1_force_40k/`：40k 训练节点 `nodes/step_005000…step_040000`、code、force 校准（已有）；
  - `v1_force_rehearsal/`：恢复演练（已有）；
  - `v2_repair_20261119/`：`evidence/`、`corpus/`、`checkpoints/action_expert_v1|v2|v3|v4/`、`full/reconstructed_normalizer/`。
- 公开 `wsadwsad6779/fasterwam-a3-archive-public`：
  - `eval_restore_20261117/`（完整树）、`root_repair_runs/`、`root_quarantine_lora300/`、`a3_data/`、`workspace_top_files.tar.gz`。

## 5. 校验方式

- 每个上传项在 `upload.log` / `tail.log` / `public.log` / `public2.log` 有 `OK` 行与速率；
- 结束后用 HF API 的 `list_repo_tree(recursive)` 取远端逐文件 size，与 pod 端 `getsize` 对账，
  结果 `ARCHIVE_RECONCILIATION.json`（已提交 GitHub）：
  - **verdict = PASS**
  - 逐文件一致 **2752 个 / 225,427,432,953 B ≈ 225.43 GB**
  - `missing = []`，`size_mismatch = []`
  - 私有仓 1331 文件 / 116.01 GB；公开仓 1605 文件 / 139.32 GB
  - `covered_via_public`：`root_repair_runs/action_expert_v2/{last.pt, segments.jsonl}`
    （私有配额满后同内容改存公开仓，按“任一仓覆盖即算”计入）
