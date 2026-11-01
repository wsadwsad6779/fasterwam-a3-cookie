# 清理候选清单（第二阶段，只提交不执行）

pod `/workspace`，数据源是 `backup_manifest_v2.json`（3,888 条，逐条带 pod 端 sha256 与 `backup_status`）。
快照时间 2026-11-01；`/dev/loop3` 可用 **3,400,036,352 B**，已用 97%。

本轮**没有删除、移动或覆盖 pod 上任何文件**。下面每一项都只是候选，等你逐项批准。
判定依据都写成可复核的数字与路径，不用「看起来没用」这类理由。

## A 档：绝对保留（无论空间多紧张）

| 对象 | 字节 | 保留理由 |
|---|---|---|
| `a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt` | 11,179,613,340 | r3 完整训练状态（模型+优化器），r4/r5 的训练基座；pod 唯一副本 |
| `a3_train_runs/a3_lora_r2/step_003000_merged.pt`、`a3_lora_r3/step_003000_merged.pt` | 各 11,117,977,659 | 只有 pod 一份；重新生成需要 11 GB 中间文件，**当前 3.4 GB 空间根本放不下**，删了就是永久丢失 |
| `a3_train_runs/a3_lora_r4/checkpoints/light/**`、`a3_lora_r5/checkpoints/light/**` | 2,961,452,401（80 个文件） | 8 个中间 step × 2 run；回答「哪一步开始退化」的唯一证据，也是 v3 接线 smoke 的加载对象 |
| `data/a3_front_close_left_100/**` | 4,306,511,889 | 原始 100 条专家示范，仿真重采集不可逐比特复现 |
| `a3_data/a3_fw/frames.dat` + `arrays.npz` + `dataset_stats.json` | 12,851,445,040 | 主训练语料。它是派生产物，但重转换要占 GPU 且依赖基础模型到位；在 HF 副本完成**逐字节**回读前按原件对待 |
| `a3_data/a3_fw_dagger/**`、`a3_data/a3_fw_dagger_near/**` | 1,570,575,733 | dagger 语料，V1 力感知改造的直接输入（12 条入选 take） |
| `a3_data/a3_fw_force/arrays.npz` | 5,243,625 | 18 维力通道的抽样数组，任务二/三的结论来源 |
| `a3_eval/**`（除 `light_state_test*`） | ~30,140,000 | 评估结果与逐回合报告；体积只占 0.03%，没有任何清理价值 |
| `a3_dual_arm_sim/`、`fasterwam_a3_bridge/`（代码部分）、`fasterwam-a3-cookie/`、`_r6_stage/`、`_r4_stage/`、`_r5_stage/` | ~65,000,000 | 现役代码与审计脚本；GitHub 上只有其中一部分（`_r6_stage` 16 个脚本 + cookie diff），其余仍是 pod 唯一副本 |
| `FasterWAM/src`、`FasterWAM/scripts`、`FasterWAM/configs` 等上游代码 | 3,504,251 | 上游 commit `83667817d` 已锚定，但本地改动（symlink/patch）只在 pod；体积小，不动 |

## B 档：已有外部副本或可外部重建 → 建议清理（每条都有前置条件）

| # | 对象 | 释放字节 | 前置条件（先做完再删） |
|---|---|---|---|
| B1 | `FasterWAM/checkpoints/Wan-AI/**`（Wan2.2 VAE 2,818,839,170 + umt5-xxl t5 11,361,920,418 + 4 个 tokenizer 文件） | 14,202,213,669 | 上游是 ModelScope `Wan-AI/Wan2.2-TI2V-5B` 公开权重。先用**只读 HEAD 请求**确认仍可下载且大小与清单一致，记录 URL+sha256（清单里已有 sha256，重建后比对即可）。代价：重建需重新下载 ~14 GB，按 1.7 MB/s 约 2.3 小时 |
| B2 | `FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt`（11,117,679,929）+ `dataset_stats.json` | 11,117,720,881 | 同上，来源是 fasterwam release。**这一项风险高于 B1**：它是 LoRA 初始化基座，若上游撤版就没有第二条路。建议只在确认 HF 上文件仍存在后再清 |
| B3 | `a3_train_runs/chk3_A/A2/B/B2/B3/checkpoints/light/**`、`chk4_N/chk4_S/**`、`a3_lora_r4_crashed_step500/**` 里的 `adapter.pt`/`optimizer.pt` | ~1,665,000,000 | 这些是 r4/r5 之前用来验证接线的诊断小跑。依赖已查明：`_r4_stage/chk3_abc.py:17` 直接引用 `chk3_A/checkpoints/light/step_000010/adapter.pt`。所以前置条件是：先把每个 step 的 `trainer_state.json`、`config.yaml`、`preflight.txt`、`train.log`（合计 < 200 KB）**逐文件推到 GitHub 备份分支**，再删 `.pt`，并保留 `chk3_abc.py` 里那句引用对应的说明，避免以后有人拿脚本去跑一个已被清掉的路径 |
| B4 | `a3_eval/light_state_test/**`、`a3_eval/light_state_test2/**` 里的 `adapter.pt`/`light/step_*/`（17 个文件，493,079,849 B） | ~492,000,000 | v2 缺陷测试的产物，结论已经写进 `docs_light_state_param_sets_audit.md` 并进了 GitHub。前置条件：把 `report.json`、`report_unreseeded_failure.json`、`trainer_state.json`、`light_state_test.log` 四个小文件（合计约 50 KB）一起推到备份分支；引用它们的是 `_r4_stage/tls_new.py`、`bridge/_next/test_light_state.py`（一次性测试脚本，清完需在文档里标注「产物已释放，脚本不可直接重跑」） |
| B5 | `a3.tar.gz`（9,608,636）、`fw.tar.gz`（945,358）、`repo_a3c.tgz`（1,357,984）、`_r4_stage/r5up.tgz`、`_r4_stage/x.tgz` | 11,920,112 | 内容就是已经发布到 GitHub 的同一批代码的打包件。前置条件：逐个 `tar tzf` 出清单、与 GitHub 分支里的文件表比对，确认没有「只存在于 tar 包里」的文件 |
| B6 | `.Trash-0/**`（234 个文件，43,430 B） | 43,430 | 回收站本身。释放量微不足道，**我不建议动它**，列出来只为完整 |

B 档全部执行后合计释放约 **27.5 GB**，可用空间从 3.4 GB 升到约 30.9 GB。

**但顺序不对就不该做**：B1/B2 之后腾出的空间，第一个用途应该是「给 P0 大文件外存留出可分片暂存/校验的余量」还是「直接用于 v3 接线 smoke」，取决于你给不给 HF 写 token。拿到 token 就优先外存；拿不到则先清 B3–B5（约 2.2 GB，不动基础模型），把 smoke 跑完，再谈 B1/B2。

## C 档：依赖不明或信息未被覆盖 → 本轮禁止清理

| 对象 | 字节 | 为什么不能算「已备份可清」 |
|---|---|---|
| `fasterwam_a3_bridge/bridge/outputs/dagger/**`、`dagger_near/**`、`dagger_resume/**` | 5,064,960,376（31 个 take 目录） | **原始帧不是转换语料的无损上游**。已核对：`a3_data/a3_fw_dagger/summary.json` 记录 `episodes: 5`、`transitions: 2168`，`source_takes` 只列 5 条；`a3_fw_dagger_near` 记 `episodes: 7`。也就是 31 个原始 take 里只有 **12 个**进了语料，剩下 **19 个 take 只存在于这些目录**（每个含 `frames.dat` 270,950,400/210,739,200 B、`report.json`、`recover.npz`、`state_takeover.npz`）。删掉就等于丢掉 19 条可用示范——而任务#24 正要用这批轨迹做审计 |
| `a3_train_runs/a3_lora_r1/**`（除已核验的 `step_002000_merged.pt`） | ~2 GB 级 | HF `trainer/step_002000.pt` 只做过**大小**比对（11,179,613,340 B 相同），sha256 未回读；而且它与 pod 上 r3 的 `weights/step_003000.pt` 字节数恰好相同、内容不同，不能靠大小互相顶替。HF 侧未逐字节核验前，pod 这份仍是唯一副本 |
| `data/a3_front_close_left_100/**` | 4,306,511,889 | HF dataset `Eter0109/a3-front-close-left-100` 不是我们的仓库，且镜像 API 不暴露 LFS sha256；「大小一致 + commit 锚点」达不到你定的验证标准 |
| `a3_eval/replay_*`、`a3_eval/pilot_*`、`_selftest/**` | 数 MB | 未逐个确认是否被 `run_matrix.py` 的某档重新读取；体积小，不值得冒险 |
| `FasterWAM/**` 其余文件（`train.log`、`train_light*.log`、`export_lora_merged.log`、`assets`、`environments`、`third_party/RoboTwin` 732 个文件） | ~1.8 MB | 上游 commit 锚点覆盖不到本地跑出来的日志与被改过的 `third_party`；日志本身是「配置与日志」类的必需内容 |
| `a3.tar.gz` 等之外的顶层 `.npy`/`.npz`/`.csv`（`r3ev_*.npy`、`expert_seed22.npz`、`readme_head.md`） | ~170 KB | 未确认下游消费者；体积可忽略，直接归入保留 |

## 空间算术（把话说完）

- 碎片类清干净能换多少？`_cmd_*.sh` 236 个 43,770 B + `*.out` 639 个 1,301,432 B + `__pycache__` 150 个 1,785,658 B + `.Trash-0` 43,430 B + `.ipynb_checkpoints` 4,172 B = **约 3.1 MB**。
- 也就是说：**清碎片对 3.4 GB 的缺口毫无帮助**，还会制造「删过东西」的既成事实。所以我没有把碎片列入 B 档建议执行项，只列在 C 档旁边作为对照。
- 真正有效的只有 B1/B2（25.3 GB），而它们恰恰是唯一「删了要花时间重新下载」的两项。这就是我不建议在拿到外存通道之前动它们的原因。
