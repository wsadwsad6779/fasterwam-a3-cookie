# 退租前最终报告（2026-11-01，终版）

本报告对应「最后的退租验收阶段」五项：全盘备份覆盖、GitHub 与 README、恢复验证边界、公开许可与 Token、
最终提交。上一版报告（A/B/C 旧口径、本轮终审前）完整保留在
`archive/2026-11-01/PRE_SHUTDOWN_FINAL_REPORT_v1.md`，其中的恢复演练细节（第 10 节）仍是有效证据。

判定方法：**按内容哈希，不按上传记录**。3,888 个文件的输入是 pod 全盘扫描 `backup_status_abc_final.json`
（路径 / 字节 / sha256），对照四类外部副本实时扫描：HF 远端文件树（1,264 个文件，`tree?recursive=true`
分页取全）、GitHub 远端 clone（含 git 历史 blob）、Windows 本机副本、固定上游 revision。
复算脚本：`_r6_stage/final_reconcile2.py`；提交前输出 `final_reconcile2.json`，提交后对全新 clone
（`_clone_verify6` @ `f40ab89`）复算输出 `final_reconcile2_postcommit.json`，两者逐文件 diff 只差下面 9 行。

## 结论

**满足退租条件。** 3,888 个文件中没有任何科研数据落在 F 档；F 档只剩 394 个 `fasterwam-a3-cookie/.git`
内部对象与元数据（972,633 B）；这个仓库的工作区文件已由 A/B/C 三档覆盖，丢的只是本地 reflog / index 这类元数据。
另有 3 项**你侧**待办（吊销 token、确认再分发许可、决定默认分支 README 与 HF 仓库可见性），都不是数据恢复阻断项。

| 档 | 含义 | 文件数 | 字节 | GiB |
|---|---|---|---|---|
| A | HF 已备份，SHA256 匹配 | 721 | 67,711,734,824 | 63.061 |
| B | GitHub 已备份，远端 clone 验证 | 1,224 | 9,018,778 | 0.008 |
| C | Windows 已备份，SHA256 匹配 | 1,083 | 33,723,595 | 0.032 |
| D | 固定上游 revision，可重新取得 | 69 | 29,626,865,464 | 27.592 |
| E | 可再生 / 自测产物 / 空文件 | 397 | 1,494,379,959 | 1.392 |
| F | 无可靠副本 | 394 | 972,633 | 0.001 |
| 合计 | | 3,888 | 98,876,695,253 | 92.086 |

（B/C 两行是**本次提交之后**的数字：提交把 9 个原先只有 Windows 副本的 C 转成了 B——`deploy/rcc_start.py`、
`deploy/wait_and_launch.sh`、light_state v2/v3 源码与测试（4 个）、`repro_env.sh`、`pip_freeze.txt`，
外加 `_r6_stage/out/repro/cookie_diff.patch`（与归档里的 worktree diff 逐字节相同）。
提交前是 B 1,215 / C 1,092；F 两轮都是 394，不受这次提交影响。A/D/E 与总字节数两轮完全一致。）

## 1. 全盘备份覆盖（逐重点项）

| 重点项 | 文件数 | 档 | 实际位置 |
|---|---|---|---|
| r3 merged + full_state | 2 / 20.766 GiB（另有 r2 merged 1 个 10.354 GiB 同属 A） | A | HF `parts60gz/P0/weights/r3/**`（178 + 177 片，装配后整文件 sha256 已核对） |
| r4 light `step_004000` | 5（r4 全部 40 个 ckpt 文件均 A） | A | HF `weights/r4/light/step_004000/**` |
| r5 light `step_008000` | 5（r5 全部 40 个 ckpt 文件均 A） | A | HF `weights/r5/light/step_008000/**` |
| 原始 100 条示范 | 60 / 4.011 GiB | D | `Eter0109/a3-front-close-left-100` @ `8fdd8a7caf8275eba8be8a83b53a99176a084062`（第三方，只引用不重传） |
| 73 条新增示范 | 14 / 13.436 GiB | A | HF `data/a3_fw*`（`frames.dat`、`arrays.npz`、统计、`corpus_manifest_v1.json`） |
| 原始 dagger take | 118 / 4.717 GiB | A | HF `eval/bridge/dagger*`（`frames.dat` 与 npz） |
| 评估 NPZ / manifest / 视频 | 375 | A 374 + E 1（空文件） | HF `eval/a3_eval/**`、`eval/bridge/**`、`videos/**`；41 个 mp4 全部 A |
| dataset_stats / force_stats | 6 | A 5 + D 1 | HF `data/**/dataset_stats.json`、`data/a3_fw_force/force_stats.json`；FasterWAM libero 那份属上游 |
| checkpoint 修复代码与测试 | 4 | B（本次提交后） | 仓库 `pod_snapshot/2026-11-01/_r6_stage/{light_state_v2,light_state_v3,test_light_state_v2,test_light_state_v3}.py` |
| 恢复脚本与环境配置 | 4 | B（本次提交后） | 仓库 `scripts/restore_new_instance.sh`、`scripts/restore_drill.sh`、`pod_snapshot/.../_r6_stage/repro_env.sh`、`.../out/repro/pip_freeze.txt` |

F 档逐条（394 个）：`fasterwam-a3-cookie/.git/objects/**` 384 个（loose object）+ `index` / `HEAD` / `config` /
`COMMIT_EDITMSG` / `FETCH_HEAD` / `refs/remotes/origin/a3-cookie-r27` 6 个 + `logs/**` 4 个。不含任何工作区文件。
说明：该仓库的 HEAD `8c2ef40d…`（分支 `a3-cookie-r27`，2026-10-31 03:50:10Z，"bridge: near-failure DAgger
takeovers and resume-held expert"）**没有推送到 GitHub**（远端同名分支是 `66f841cc…`）；它的工作区内容已被
A/B/C 完整覆盖，未提交的工作区改动另存为 `archive/2026-11-01/fasterwam-a3-cookie_worktree.diff`（468,310 B 文本）。
丢掉的是本地 reflog / index / dangling object 这类元数据，不是科研结果。

E 档构成（397 个 / 1.392 GiB）：`chk3_*` / `chk4_*` 自测检查点 32 个（1.379 GiB，light_state v3 自测产物，
不是训练结果）、上传探针切片与 scratch 日志 321 个（KB 级）、空文件 26 个、打包中间件 13 个、
启动日志 2 个、代码打包 1 个、已归档 npz 的派生切片 1 个、上传缓存 1 个。

重点项之外仍需注意的：`fasterwam-a3-cookie/README.md` 的 pod 版本（22,630 B）只存在于
`fasterwam-a3-cookie_worktree.diff` 的补丁文本里（C 档，补丁本身现已进仓库），当前仓库 README 是它的升级版。

## 2. GitHub 与 README 远端验收

以下都是对远端做只读查询的结果，不是本地 `git status`：

| 检查 | 结果 |
|---|---|
| `archive-2026-11-01` 远端 tip | `f40ab89cd5347750c5a1809a859d95d509d463fc`（`git ls-remote` 读回）——把 9 个文件转成 B 的那次提交，父提交是要求核对的 `b82854f`；本报告的修订提交在其之上 |
| 分支列表 | `a3-cookie-r27` `66f841cc…`（默认分支，未动）、`archive-2026-11-01` `f40ab89…`、`backup-2026-11-01-r6-audit` `1fb28037…` |
| 远端 README 是否最新 | 是。推送后在全新 clone `_clone_verify6` 里读到第 11.3 节（上传手册）与第 11.4 节（GitHub 上传经验）都在；本次又修正了三处失真见下 |
| **默认分支 README** | **未更新**。GitHub 默认分支是 `a3-cookie-r27`（`git ls-remote --symref` → `ref: refs/heads/a3-cookie-r27`），其 README 26,051 B、最后改动 `66f841c`（2026-10-30），**没有第 11 节**。按你的要求没有改写默认分支历史，是否合并由你决定 |
| 恢复脚本可执行且正确 | `scripts/restore_new_instance.sh` 远端 blob 13,973 B / `745e20e7…`（与 README 第 11 节钉的值一致） |
| HF Model Card | 已提交且可远端读取：`resolve/main/README.md` 200，5,228 B，含 YAML front matter 与「已知缺口 / 数据与许可 / 校验口径」 |
| HF Dataset Card | **不存在独立 dataset 仓库**（`/api/datasets/...` 401）。数据以 `data/**` 放在模型仓库内，并在 Model Card 的「数据与许可」里说明 |
| HF revision | `refs` 与 `tree?expand=true` 一致：`main = f2aaf55c6ce1b164de97c5b6ec2cd75e942d0de0`（2026-11-01T08:48:11Z，"Archive M batch 1/1"）。注意 `GET /api/models/...` 会返回 2026-10-26 的陈旧缓存体（sha `dc10a5d4…`、9 siblings），不能当依据 |
| README 链接真实性 | 逐目录实测：`weights/r4`、`weights/r5`、`parts60gz`、`data`、`eval/a3_eval`、`eval/bridge`、`videos/*`、`runs`、`replay/bridge`、`tests`、`research`、`MANIFEST` 全部 200；**`weights/r2`、`weights/r3` 是 404**——台账里这两行是目标路径，远端只存了分片。README 本次已改 |
| 台账逐行对远端 | 715 行台账，0 行缺失：712 行按 path 命中，3 行由 532 个分片装配（178 + 177 + 177，索引连续 0..N） |

提交 sha 无法写进提交自身：上表 `f40ab89` 是推送后立即读回的 tip；本报告的修订提交是它的子提交，
分支最终 tip 以 `git ls-remote origin refs/heads/archive-2026-11-01` 或 `git log -1` 为准
（本次终审推送后读回的是 `f40ab89` 的子提交，见对话里报出的最终 commit）。

README 本次修正的三处失真（都是这次对着远端查出来的）：

1. `weights/r2/`、`weights/r3/` 那行原写「r2、r3 的 11.1 GB 合并权重与 full_state」，实际远端没有这两个目录。
   已改为「台账目标路径，远端只有分片，必须装配」。
2. `tests/light_state_test*` 原写「light_state v3 修复代码与测试」，实际是 v3 产出的
   `adapter/optimizer/rng/scheduler` 与自测目录，不是源码。源码位置已改指 `pod_snapshot/2026-11-01/_r6_stage/`。
3. 第 11.1 节原来用的 A/B/C（A = pod 外有逐字节副本、B = 只核大小、C = pod 唯一副本）与本次验收的 A–F 口径冲突，
   已替换为 A–F 表，并注明旧口径作废。

## 3. 恢复验证的边界（必须如实读）

已经做到的，是**文件级**验证：

- 服务端对账：远端 1,264 个文件 ↔ 台账 1,247 行，`sha_verified=1019`、`size_only_no_oid=225`、
  `giant_assembled=3`、`missing=0`、`size_mismatch=0`、`sha_mismatch=0`。
- 本机独立网络抽样回读 17 个文件 `bad=0`（含 61.6 MB adapter、210.7 MB `frames.dat`、3 个 gz 分片）。
- 旧 pod 空目录全量演练（`/root/drill`、`/root/drill_stage`、`/root/drill_stage2`、`/root/drill3_stage`，
  只带台账、强制走 hf-mirror 重新下载、`cat part_*.gz | gunzip` 装配分片）：
  第三轮正例 `verify rows=715 ok=715 missing=0 sha_bad=0 giants_missing=0 giants_bad=0 named_missing=0`；
  3 个反例（空目录全量、点名一个不存在的文件、台账外路径）都按预期返回非零，2 个附加正例
  （点名一个存在的文件、`--weights A B` 不重复下载）返回 0。
- 三个 >2 GiB 权重装配后整文件 sha256 与台账一致：`e45110fd…`（r3 full_state）、`1ac089bf…`（r3 merged）、
  `baafffcf…`（r2 merged）。

**没有做到的，不能写成做到了：**

- 演练跑在同一台旧 pod 的新建目录里，不是另一台全新 GPU 实例。
- `--code`、`--deps` 两个阶段没有跑过；代码侧的等价证据是「推送后另开目录全新 clone 并逐文件比对」。
- **模型推理 smoke 从未在全新实例上跑过**（v3 一致性 smoke、V1 训练、严格抓放成功也都仍未做，见第 13 节 README）。
- 因此本归档的准确表述是：**文件级可恢复，已用远端重下载验证；端到端「新机器一键复现」尚未验证**。

## 4. 公开许可与 Token

| 对象 | 处置 | 依据 |
|---|---|---|
| `hustvl/FasterWAM` | 代码进 `git_snapshot_20261101.tar.gz`，Apache-2.0，LICENSE 随包 | 仓库 `THIRD_PARTY_NOTICES.md` |
| `FasterWAM/third_party/RoboTwin` | 同上，MIT © 2025 Tianxing Chen | 同上 |
| `Eter0109/a3_dual_arm_sim` | 同上，MIT © 2026 Eter0109 | 同上 |
| `robosuite` 模型资产 | 同上，LICENSE 在同目录 | 同上 |
| `Eter0109/a3-front-close-left-100`（原始 100 条） | **许可未声明 → 不复制、不上传 HF，只记录 revision** | README 第 11 节、THIRD_PARTY_NOTICES |
| Wan-AI 2.2 权重、`fasterwam_release` 权重 | **未重传**，只记录上游来源，按 D 档重下 | README 第 11.2 节 |
| 密钥与敏感信息 | GitHub 22 个提交全历史 `git grep` 扫描：HF token / GitHub PAT / AWS AK / 私钥 / `sk-` 均 **0 命中**；HF 1,264 个文件名扫描 token/secret/.env/credential/id_rsa/api_key/.pem **0 命中**；本次新增 9 个文件同样 0 命中 | 见本报告附录命令 |

HF 仓库目前是**公开**状态（`private: false`），第三方数据集与权重都没有放进去，因此公开不构成再分发；
是否保持公开由你决定。旧 pod 上的 HF token 仍建议由你本人在 HF 后台吊销（它未写入脚本、日志、commit 或聊天，
只存在于 pod 的 `/root/.cache/huggingface/token`，实例销毁即失效）——本次终审**没有使用任何 token**。

## 5. 本次提交新增的归档文件

| 路径 | 字节 | sha256 |
|---|---|---|
| `deploy/rcc_start.py` | 4,546 | `4acaf9cc697731360dacec600aeb2133e6c189d986464395b53e1ac78a4432a2` |
| `deploy/wait_and_launch.sh` | 962 | `9b8f008facf68c3494383ee7deb05006eae3f6c506763555095c64fe17ac51e2` |
| `pod_snapshot/2026-11-01/_r6_stage/light_state_v3.py` | 8,003 | `4d7ceb8764bba59c…` |
| `pod_snapshot/2026-11-01/_r6_stage/test_light_state_v3.py` | 5,823 | `9c41ebc91a66a229…` |
| `pod_snapshot/2026-11-01/_r6_stage/light_state_v2.py` | 8,319 | （同目录，v2 对照） |
| `pod_snapshot/2026-11-01/_r6_stage/test_light_state_v2.py` | 10,750 | |
| `pod_snapshot/2026-11-01/_r6_stage/repro_env.sh` | 1,198 | `bac096dd8d60a493…` |
| `pod_snapshot/2026-11-01/_r6_stage/out/repro/pip_freeze.txt` | 8,416 | `20b58593e8a2d2c9…` |
| `archive/2026-11-01/fasterwam-a3-cookie_worktree.diff` | 468,310 | `4e9a680f4a813bee…` |

前两个是 pod 上未纳管、也没有 git 历史的启动脚本（原来唯一的副本在即将销毁的 pod 上）；六个是只有 Windows
副本的修复代码 / 测试 / 环境配置；最后一个把旧 pod 仓库 11 个未提交改动的 diff 固化进 git。
全部是逐字节从 pod 取回、sha256 与 pod 侧 `SHA256SUMS.txt` 对拍一致后写入的；推送后在全新 clone 里再逐文件
sha256 复核，10 个文件（含 `PRE_SHUTDOWN_FINAL_REPORT_v1.md`）与本地逐字节相同。

补充：pod 上还有一份 `_r6_stage/out/repro/cookie_diff.patch`，与上面最后一行那个 diff 逐字节相同
（4e9a680f…），所以它在提交后也从 C 变成 B——这就是 B 增加 9 个而不是 8 个的原因。

## 6. 仍需你做的三件事（都不是数据阻断项）

1. 到 HF 后台吊销本会话用过的 token。
2. 确认 `Eter0109/a3-front-close-left-100` 与 FasterWAM / Wan-AI 权重的再分发许可；确认 HF 仓库保持公开还是转私有。
3. 决定默认分支 `a3-cookie-r27` 的 README 是否也要更新（我未改动默认分支）；然后由你本人执行实例终止。

pod 上的上传缓存 `/root/hfstage`（约 88 GB）、`/root/drill*` 演练目录、`_r6_stage` 临时脚本都还在，
清理清单一事一列等你批准，不自动删。
