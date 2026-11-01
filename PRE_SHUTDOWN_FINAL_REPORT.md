# 退租前最终报告（2026-11-01）

本报告对应原任务书第 7 节的 12 项验收。所有字节数与 sha256 以 `archive/2026-11-01/` 下的两份台账为准：

```
archive/2026-11-01/p0p1_manifest.tsv      715 个文件：档、path_in_repo、pod 相对路径、字节、sha256
archive/2026-11-01/parts60gz_manifest.tsv 532 个分片：分片 path_in_repo、pod 相对路径、片字节、片 sha256
```

「备份成功」只认一种证据：**从 pod 外把文件读回来，sha256 与台账相同**。仅大小相同不算通过。

## 结论先行

- 8 项核心成果（代码与环境 / r1–r5 权重与训练状态 / 原始 100 条专家 / 73 条新增示范 / B0-C0 评估与视频 /
  训练配置与归一化统计 / light_state v3 与测试 / 各阶段报告）都已在 pod 外留有副本，并完成远端回读对账。
- 全量恢复演练（在 pod 上新建空工作区，强制走网络重新下载、按台账重建分片、逐文件 sha256）结果见第 10 节。
- **r4 / r5 训练末态的 `proprio_encoder` 参数无法由本次备份恢复**（第 11 节）。它从未落盘、pod 上也没有副本。
- 严格意义的抓放成功仍是 0；V1 力感知训练、v3 真实模型 smoke 都还没做。这些在 README 第 13 节也逐条写死。
- 实例终止由你本人执行；本报告不代替你做这件事。

## 1. GitHub 提交与分支

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/wsadwsad6779/fasterwam-a3-cookie`（私有） |
| 分支 | `archive-2026-11-01`（新开，未动既有分支） |
| 内容冻结提交 | `82e807cb299c49901f724b9651ec52913ff9dc0` — "archive: freeze the 2026-11-01 working set before the pod is torn down"（970 files changed, 123,442 insertions） |
| 报告与本轮修复 | `e7c9eb000c0bf23cbd5172e2c61835df7916107e` — 恢复脚本修复、README 第 11.4 节、本报告、`docs/BACKUP_STATUS_ABC_FINAL.md` |
| 受管文件 | 1,209 个（`git ls-files`） |

提交 sha 无法写进提交自身；上表是推送时的远端 HEAD，之后若还有文档修订，以 `git log -1` 为准。

仓库里有什么：`pod_snapshot/2026-11-01/`（949 个本项目代码/文档/脚本）、`scripts/`、`docs/`、`bridge/`、
`finetune/`、`tools/`、`results/`、`archive/2026-11-01/`（台账、上传器源码、git 快照 tar.gz）。
上游 checkout 与 11 GB 级权重不进 git，`.gitignore` 显式排除。

## 2. GitHub README 回读

推送后在新目录做了全新 clone（不是 `git status`），以下数字来自 `e7c9eb0` 那次复验：

- 克隆出的受管文件 1,209 个，与推送前工作区逐文件比 sha256：
  **956 个逐字节相同 / 253 个仅行尾 CRLF↔LF / 0 个其它差异 / 0 个缺失**。
- 三个钉死文件的 sha256 在 clone 里逐一核对通过：
  - `archive/2026-11-01/p0p1_manifest.tsv` = `54f4e3669a46b3af58331abee2b55c350c1292516b692c19cb4bb0520d596b08`
  - `archive/2026-11-01/parts60gz_manifest.tsv` = `307172a4f82d98f58391068c2ac4078ac4f8ff64efafa50470361456db6d0f04`
  - `scripts/restore_new_instance.sh` = `745e20e736c68770d10782c0dea2c48ef307abd6b09964bbc322bcbd2a8138c9`
- README 含任务书要求的 13 项：研究目标、A3 任务、专家策略、r1→r5 训练史、B0/C0 评估表、两套入盒口径、
  接触力量级、`proprio_encoder` 漏存、73 条新增示范、V1 方案、归档位置与校验、恢复步骤、未完成工作。
  另加第 11.3 节上传手册与第 11.4 节 GitHub 上传经验（6 条，含「台账列错位只有真跑演练才能发现」）。

253 个行尾差异的来源是本机 `core.autocrlf=true`：这类文件在仓库里是 LF，pod 原始文件是 CRLF。
内容没有差异，但「仓库字节 = pod 字节」对这 253 个文件不成立；仓库根已加 `.gitattributes`（`* -text`）阻止后续归一化。

## 3. HF 模型仓库

| 项 | 值 |
|---|---|
| URL | `https://huggingface.co/wsadwsad6779/fasterwam-a3-cookie-lora` |
| HEAD revision | `f2aaf55c6ce1b164de97c5b6ec2cd75e942d0de0`（从 `resolve/main/README.md` 的 `X-Repo-Commit` 读回） |
| 远端文件数 | 1,264（`tree?recursive=true`） |
| 归档台账覆盖 | 1,247 行（715 个文件 + 532 个分片） |
| 可见性 | 镜像 API 返回 `private: false`，即当前是公开仓库；若不想公开请到 HF 设置改私有 |

目录约定：`weights/r{2,3,4,5}/`、`parts60gz/<档>/<原路径>/part_NNN.gz`、`data/`、`eval/`、`videos/`、
`runs/`、`replay/`、`tests/`、`research/`、`MANIFEST/`；r1 老文件在仓库根与 `trainer/**`。

## 4. HF 数据仓库

没有单独建 HF dataset 仓库：本项目产生的数据（`a3_fw`、`a3_fw_dagger`、`a3_fw_dagger_near`、`a3_fw_force`）
作为 `data/**` 放在上面的模型仓库里，台账同一份。

原始专家数据集是第三方，**只引用不重传**：

```
https://huggingface.co/datasets/Eter0109/a3-front-close-left-100/tree/8fdd8a7caf8275eba8be8a83b53a99176a084062
```

再分发许可未确认，所以 `data/a3_front_close_left_100/` 的 56 个文件在分类里是 B（可从上游原 revision 重取，
本仓库没有复制）。FasterWAM / Wan-AI 上游权重同样只引用，不重传。

## 5. HF README 回读

从 `https://hf-mirror.com/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/README.md` 真实读回：

- 5,228 B，sha256 `fe344f862153310974f5bc198afd63a65d4ffcea99510255bbb7ec95a1f4b1ec`；与本地存档 `hf_readme_live.md` 相同。
- 内容与 GitHub README 的事实一致：715 行 + 532 分片台账、三个大权重的整文件 sha256 与分片数、
  `cat part_*.gz | gunzip` 的恢复方式、已知缺口（`proprio_encoder` 丢失、抓放成功 0、V1 未训练）、
  第三方数据许可口径、「只看大小不算验证」的校验定义。

## 6. 3,888 个文件的最终 A / B / C 分类

判定按内容哈希，不按任何记录或声明；pod 内部重复副本不算备份。完整表见 `docs/BACKUP_STATUS_ABC_FINAL.md`。

| 档 | 含义 | 文件数 | 字节 | GiB |
|---|---|---|---|---|
| A | pod 外有**逐字节相同**的副本（本机实扫或 HF 远端回读） | 3,005 | 54,905,012,419 | 51.134 |
| A0 | 同上但是 0 字节文件（空文件互相「相同」不构成证据） | 26 | 0 | 0.000 |
| B | 有外部来源，但只核过大小，或属于可从公开上游重取、未做字节核验 | 487 | 42,477,000,217 | 39.560 |
| C | pod 外没有任何一份这些字节 | 370 | 1,494,682,617 | 1.392 |
| 合计 | | 3,888 | 98,876,695,253 | 92.086 |

B 档构成：上游 `FasterWAM/checkpoints`（Wan / fasterwam_release）8 个 23.581 GiB、`a3_data/a3_fw/frames.dat`
1 个 11.967 GiB（对应 HF `data/frames.dat`，本轮未做字节回读）、第三方专家集 56 个 4.011 GiB、
`fasterwam-a3-cookie/.git` 411 个（可由 GitHub 重建）、其余 11 个。

C 档构成：`a3_train_runs/chk3_* + chk4_*` 32 个 1.379 GiB（light_state v3 自测产生的检查点，**自测产物，不是训练结果**）、
三个旧打包 `a3.tar.gz` / `fw.tar.gz` / `repo_a3c.tgz`、少量 KB 级探针切片（`_r4_stage/up` 206 个共 94 KB、
`_r4_stage/p` 26 个共 8 KB）与若干零散 npy/log。

A 档的口径：11,179,613,340 B 的 r1 合并基座、r2/r3 三个大权重、r4/r5 的 80 个轻量状态文件、评估产物与录像，
以及 GitHub 归档里字节相同的代码文件。每一份都用 sha256 比对过，不是按清单点数。

## 7. r3 / r4 / r5 权重与训练状态的 SHA256

三个 >2 GiB 的整文件（远端以 532 个 gz 分片存放，恢复后必须核对下面整文件 sha256）：

| 整文件 | 字节 | sha256 | 分片 |
|---|---|---|---|
| `weights/r3/full_state/step_003000.pt` | 11,179,613,340 | `e45110fd6742a72a998aa8533eb0bfb7cdd47513174b4e052962f1af47de7547` | 178 |
| `weights/r3/step_003000_merged.pt` | 11,117,977,659 | `1ac089bf0b4ec41441be274dc3b9d66d75aa31e55dec6751907c59955c4f1c90` | 177 |
| `weights/r2/step_003000_merged.pt` | 11,117,977,659 | `baafffcf9aacfc7a8a1e9276100fdfce4763e2dda0d513c7be7b303f0356d6a2` | 177 |

（`weights/r3/full_state/step_003000.pt` 的整文件 sha 与 r2 合并基座不同；三个文件互不相同，见上表。）

r4 轻量状态：`step_000500 … step_004000` 共 8 个检查点 × 5 件（adapter / optimizer / rng / scheduler / trainer_state）。
r5 轻量状态：`step_004500 … step_008000` 共 8 个 × 5 件。逐文件 sha256 在台账里。末态两条：

| 文件 | 字节 | sha256 |
|---|---|---|
| `weights/r4/light/step_004000/adapter.pt` | 61,595,489 | `e8372c333eb2e454dd91c79ff597bd4ce6ae0aa3a2594c57b016e02ff40f916d` |
| `weights/r4/light/step_004000/optimizer.pt` | 123,478,413 | `e8e9d350c412019daca8feec6390f3a02f7c9a949ed39c83d489ce3b24d29978` |
| `weights/r4/light/step_004000/trainer_state.json` | 639 | `c5fd77224fff3cd53e5dbccddc15f6c785b950da5460883518d8a8ff6de9d5e3` |
| `weights/r5/light/step_008000/adapter.pt` | 61,595,489 | `af6bba85d1bd99c14d2547c8e59456f24cf570b5bbddd20bdbf1dc69b3607f09` |
| `weights/r5/light/step_008000/optimizer.pt` | 123,478,413 | `6e517a87cbb3baa634f743cb9440111e397fcf7b1147a4ce7fb50fc4fc4c7e4c` |
| `weights/r5/light/step_008000/trainer_state.json` | 641 | `83035ee4a1956fdd1cfb69456517cec7aef74e9ff9165972353f98fee09c86cf` |

**这 80 个轻量文件只包含 784 / 786 个张量**：训练中实际被更新的 `proprio_encoder`（2 个张量、36,864 元素、
147,456 B 张量字节）不在里面，pod 上也没有第二份。所以 r4 / r5 的训练末态模型不可精确复现；
可复现的是「r3 基座 + LoRA adapter + 优化器/调度器/RNG 状态」这一组合所对应的评估结果。

## 8. 原始 100 条与新增 73 条的备份状态

**原始 100 条专家示范**（第三方，`data/a3_front_close_left_100/`）：56 个文件 / 4.011 GiB，全部为 B 档——
只在上游 dataset 仓库按固定 revision 存在，本归档没有复制，未做字节级回读。需要时：

```
huggingface-cli download Eter0109/a3-front-close-left-100 --repo-type dataset \
  --revision 8fdd8a7caf8275eba8be8a83b53a99176a084062 --local-dir data/a3_front_close_left_100
```

**新增 73 条轨迹语料**（`a3_data/a3_fw_dagger`、`a3_fw_dagger_near`、`a3_fw_force`，以及 `a3_fw` 的帧文件）：
14 个条目 / 13.436 GiB，其中 13 个 A 档（帧文件 `frames.dat` 与 `arrays.npz`、`summary.json`、
`dataset_stats.json`、`force_stats.json` 都以 sha256 回读核验），1 个 B 档为 `a3_data/a3_fw/frames.dat`
（11.967 GiB，与 HF 老文件 `data/frames.dat` 对应，本轮未做字节回读）。
逐条轨迹的审计与带图 smoke **没有做**（README 第 13 节、任务 #24）。

## 9. 评估 NPZ、视频与报告的备份状态

- 评估结果（NPZ / manifest / summary / 分析报告 / replay 帧）：587 个条目 / 5.213 GiB，**A 586 / A0 1 / B 0 / C 0**。
- r4/r5 结果视频：41 个 / 7.713 MiB，**全部 A**；另有本地交付包 `FasterWAM_r3_r4_r5_实验视频合集.zip`
  （18,732,532 B）与 `SHA256SUMS.txt`。
- B0/C0 的 `summary_lora.json`、`manifest_lora.json` 都在 `eval/a3_eval/**`、`eval/bridge/**` 下，随 `--rest` 恢复。
- 打分口径三套（`cookies_in_target` 官方值 / verbatim / 改写步数）在 README 第 6 节分栏写清，报告数字都标口径。

## 10. 真实恢复演练结果

演练不是「在 pod 本地 cp 一遍」，而是：新建空目录、只带台账、从远端强制重新下载、按分片重建大文件、
最后逐文件 sha256 对账。

**(a) 本机抽样回读（独立网络）**：17 个文件 `bad=0`，覆盖每个前缀的最小与中档文件、一个 61.6 MB 的 adapter、
一个 210.7 MB 的 `frames.dat`，以及 3 个 gz 分片。

**(b) 服务端 tree 对账**：远端 1,264 个文件对台账 1,247 行——`sha_verified=1019`（LFS 带 oid 的直接比 sha）、
`size_only_no_oid=225`（普通 git blob 无 oid，改由抽样回读覆盖）、`giant_assembled=3`、
`missing=0`、`size_mismatch=0`、`sha_mismatch=0`。

**(c) pod 上全量演练**（旧 pod，`/root/drill` 空工作区、`/root/drill_stage` 空台账目录，
使用从 GitHub clone 出来的 `restore_new_instance.sh`，只认台账、不认本地已有目录，先 `weights`、
再 `rest`、`data`、`parts`，最后 `verify`）：

第一轮用的脚本是推送前的版本（`3bc1ba958fb2691b235070061d9277fee19e3631f82143ff49395220046d2d7e`），
台账由脚本自己从远端 `MANIFEST/` 取回（sha `54f4e366…`、`307172a4…`，与仓库内文件一致）：

| 阶段 | 起止 | rc | 说明 |
|---|---|---|---|
| `--weights` | 08:56:37Z → 09:01:58Z | 0 | 全部新下载，无本地命中 |
| `--rest` | 09:01:58Z → 09:22:16Z | 0 | eval / videos / replay / runs / tests / research |
| `--data` | 09:22:16Z → 09:23:54Z | 0 | 原始 100 条与 73 条语料 |
| `--parts` | 09:23:54Z | **1** | **失败**：`no manifest row for P0\tparts60gz/P0/...` |
| `--verify` | 09:23:54Z → 09:24:10Z | 0 | `rows=715 ok=712 missing=3 giants_missing=532`——数字本身就是错的 |

这条 `rc=1` 抓出两个真 bug，都是静态审查看不出来的：

1. **分片台账第 1 列是档位（P0/P1），不是路径**。`do_parts` / `do_verify` 的 awk 按「行首是 `parts60gz/`」
   去匹配，永远不命中，整行被当成路径，于是 532 个分片一个都没装配。
2. **`do_verify` 只报 `missing` 不计入失败**：3 个整文件缺失、532 个「装配路径」不存在，脚本仍然 `rc=0`。

修复：新增 `whole_paths()` 取第 2 列、剥掉 `parts60gz/<tier>/` 前缀和 `/part_NNN.gz` 后缀得到整文件路径
（`do_parts` 与 `do_verify` 共用）；`do_verify` 在缺文件或 sha 不符时返回非零；
任何下载/装配失败都累加 `fails`，退出时打印 `RESTORE FAILED: N file(s)` 并以非零码结束。
这一版 sha256 `98510b0774ef9023e72cd9e1c51c0033e5c59b9e9187b48ae0cab6cd485492c7`（13,359 B），第二轮演练用的就是它。

第二轮在**新目录** `/root/drill_stage2` 上复跑修复版，工作区继续用 `/root/drill`（`weights`/`rest`/`data` 的文件已就位，
正好检验「已存在且 sha 相同就跳过、缺失就补齐」）：

| 阶段 | 起止 | rc | 结果 |
|---|---|---|---|
| `--parts` | 09:25:06Z → 10:13:20Z | 0 | 532 片：475 片本次新下载（含中途 6 次 hf-mirror 429，靠 `--retry` 挺过去），57 片已落盘且 sha 相同直接命中；3 个整文件全部装配成功 |
| `--verify` | 10:13:20Z → 10:14:28Z | 0 | `verify rows=715 ok=715 missing=0 sha_bad=0 giants_missing=0 giants_bad=0` |

三个整文件装配后的 sha256 与台账一致（`verify` 里逐条打印 `OK (assembled)`）：
`weights/r2/step_003000_merged.pt`、`weights/r3/full_state/step_003000.pt`、`weights/r3/step_003000_merged.pt`。

第二轮通过后复查 `do_verify` 的判定，又找到两个「会静默放过」的口子，都是这一轮不跑就发现不了的：

3. **全量 `--verify` 不把缺整文件当失败**：三个分片装配的大文件全缺时只打印 `giants_missing=3`，退出码仍是 0。
4. **`--verify <某个文件>` 不检查那个文件在不在**：名字点了的文件缺失，也只算进 `missing`，照样通过。

再加一层修复（最终版 sha256 `745e20e736c68770d10782c0dea2c48ef307abd6b09964bbc322bcbd2a8138c9`，13,973 B）：
新增 `wanted()` 判断某个路径是否在命令行里被点名；点名的文件缺失计入 `named_missing` 并失败；
全量 `--verify` 要求 `missing=0` 且 `giants_missing=0`；只有带了具体路径的部分校验才容忍其余文件不存在。
输出行末多一个 `named_missing` 字段：命令行点名但没找到的文件数，正常必须是 0。

第三轮 `/root/drill3_stage`（正例用 `/root/drill`，反例用新的空目录）把六个场景都跑了一遍：

| 场景 | 命令 | rc | 输出 |
|---|---|---|---|
| 完整工作区全量校验 | `--verify` | 0 | `rows=715 ok=715 missing=0 sha_bad=0 giants_missing=0 giants_bad=0 named_missing=0` |
| 点名一个存在的文件 | `--verify weights/r4/light/step_004000/trainer_state.json` | 0 | 同上 |
| 点名一个不存在的文件 | 同上，换到空目录 | **1** | `missing=715 … named_missing=1` |
| 空目录全量校验 | `--verify` | **1** | `rows=715 ok=0 missing=715 … giants_missing=3` |
| 台账里没有的路径 | `--weights no/such/file.bin` | **1** | `not in manifest` + `RESTORE FAILED: 1 file(s)` |
| `--weights A B` 不重复执行 | `--weights <A> <B>` | 0 | 每个文件 `get` 恰好 1 次、`already` 0 次 |

**演练的边界（不能夸大）**：这是同一台旧 pod 上新建目录的演练，「远端」确实是 hf-mirror 而非本地拷贝，
但 `--code` / `--deps` 两个阶段没有跑（本次没有第二台 GPU 实例），代码侧的等价验证是「推送后另开目录全新
clone 并逐文件比对」（见第 2 节）。新实例上首次执行时，这两步仍要先跑一遍 `--code --deps`。

## 11. 未完成工作与丢失风险

1. **r4 / r5 的 `proprio_encoder` 末态参数永久丢失**：轻量 checkpoint 只存 784/786 个张量，该分支训练后从未落盘，
   pod 上也没有副本。本归档无法补回；从轻量 ckpt 精确续训做不到，只能从 r3 `full_state` 重来或接受随机初始化。
   历史 B0/C0 评估读数仍可复现（那用的是 r3 基座 + adapter），但「r4/r5 末态模型」本身不可重建。
2. **严格意义的抓放成功：0。** 12 条录制里没有一次 `rest_in_bin > 0`。
3. **light_state v3 只做到代码与自测**，没有接真实训练做一致性 smoke（任务 #21/#27）。
4. **V1 力感知输入（8D → 14D）从未训练、从未评估**，只有代码与方案。
5. **73 条语料的逐条审计与带图 smoke 未完成。**
6. 第三方再分发许可：`Eter0109/a3-front-close-left-100`、`hustvl/FasterWAM`、Wan-AI 权重都只引用未重传；
   是否允许再分发需要你确认。
7. HF 仓库当前是公开状态（镜像 API `private: false`），请你确认是否符合预期。
8. 本会话用过的 HF token 只在旧 pod 的 `/root/.cache/huggingface/token`（0600），没有写进脚本 / Git / 日志 / 聊天；
   实例销毁后失效，仍建议在 HF 后台吊销。
9. 254 个文件在 GitHub 归档里是 LF、pod 原始是 CRLF；内容一致但字节不同。
10. C 档 370 个文件 / 1.392 GiB 仍只在 pod：主体是 light_state v3 自测的 chk3_*/chk4_* 检查点（自测产物）
    与 KB 级上传探针切片，不属于 8 项核心成果。

## 12. 是否满足安全结束条件

| 条件 | 状态 |
|---|---|
| 8 项核心成果在 pod 外都有副本 | 满足（第 1–9 节；r4/r5 proprio 这一项属于历史丢失，不是这次备份造成的） |
| 远端校验通过（不是只看大小） | 满足：服务端 tree 对账 + 本机抽样回读 + pod 全量演练 sha256 |
| 恢复演练完成 | 满足：见第 10 节；代码侧用「推送后全新 clone 逐文件比对」验证 |
| 无阻断问题 | 满足：认证、额度、上传都已完成；剩下是你侧的许可确认、token 吊销、实例终止 |
| 声明未完成事项 | 满足：第 11 节与 README 第 13 节逐条写明，不把未做的事写成已完成 |

结论：**归档与恢复演练已完成，可以进入你本人确认后的实例终止流程。**

仍需你做的三件事：

1. 到 HF 后台吊销本会话用过的 token。
2. 确认 HF 仓库的公开/私有状态，以及 Eter0109 数据集与 FasterWAM / Wan-AI 权重的再分发许可。
3. 由你本人执行实例终止；我不代为关闭，也不删除工作盘。

（pod 上的上传缓存 `/root/hfstage` 约 88 GB、`/root/drill*` 演练目录、`_r6_stage` 临时脚本都还在，
清理清单一事一列、等你批准，不自动删。）
