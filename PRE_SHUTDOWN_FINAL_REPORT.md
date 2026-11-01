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

**满足退租条件。** 3,888 个文件中没有任何科研数据落在 F 档；F 档的 394 个（972,633 B）全部是
`fasterwam-a3-cookie/.git` 的内部对象与本地指针——其中 384 个 loose object 与 1 个远端跟踪引用已由第 8 节的
bundle 逐一覆盖（外部副本在 GitHub），剩下 9 个是 `index` / `HEAD` / `logs/**` 这类本地指针与日志，不含独立内容。
另有 2 项**你侧**待办（吊销 token、确认再分发许可与 HF 仓库可见性）；默认分支 README 已单独更新为最新版。
73 条原始示范的逐条回读见第 7 节，旧 pod 独有 Git 历史的保全见第 8 节。

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
| 73 条新增示范 | 派生数组 14 / 13.436 GiB；原始轨迹 73 个 / 7,657,304 B | A | HF `data/a3_fw*`（派生数组与统计）；HF `eval/a3_eval/demos_*/**.npz`（73 条原始轨迹，逐条回读见第 7 节） |
| 原始 dagger take | 118 / 4.717 GiB | A | HF `eval/bridge/dagger*`（`frames.dat` 与 npz） |
| 评估 NPZ / manifest / 视频 | 375 | A 374 + E 1（空文件） | HF `eval/a3_eval/**`、`eval/bridge/**`、`videos/**`；41 个 mp4 全部 A |
| dataset_stats / force_stats | 6 | A 5 + D 1 | HF `data/**/dataset_stats.json`、`data/a3_fw_force/force_stats.json`；FasterWAM libero 那份属上游 |
| checkpoint 修复代码与测试 | 4 | B（本次提交后） | 仓库 `pod_snapshot/2026-11-01/_r6_stage/{light_state_v2,light_state_v3,test_light_state_v2,test_light_state_v3}.py` |
| 恢复脚本与环境配置 | 4 | B（本次提交后） | 仓库 `scripts/restore_new_instance.sh`、`scripts/restore_drill.sh`、`pod_snapshot/.../_r6_stage/repro_env.sh`、`.../out/repro/pip_freeze.txt` |

F 档逐条（394 个）：`fasterwam-a3-cookie/.git/objects/**` 384 个（loose object）+ `index` / `HEAD` / `config` /
`COMMIT_EDITMSG` / `FETCH_HEAD` / `refs/remotes/origin/a3-cookie-r27` 6 个 + `logs/**` 4 个。不含任何工作区文件。
说明：该仓库的 HEAD `8c2ef40d…`（分支 `a3-cookie-r27`，2026-10-31 03:50:10Z，"bridge: near-failure DAgger
takeovers and resume-held expert"）不在 GitHub 上：它的父提交 `771eff1…` 是 GitHub tip `66f841c` 的祖先，
`8c2ef40d` 是从 `771eff1` 分叉出来的兄弟提交，远端从未收到过它。
它的工作区内容已被 A/B/C 完整覆盖，未提交的工作区改动另存为
`archive/2026-11-01/fasterwam-a3-cookie_worktree.diff`（468,310 B 文本）。

**本轮补上了历史对象**：这 11 个提交、4 个引用与 pack 里的 4 个不可达对象已打成
`archive/2026-11-01/fasterwam-a3-cookie_history_2026-11-01.bundle`（524,674 B，sha256 `c7946637…`），
并用 OID 集合对拍——pod 上 384 个 loose object 的 OID 与 bundle 内 384 个对象**逐一相同，缺 0 个、多 0 个**。
所以 F 档现在剩下的只是 `index` / `HEAD` / `config` / `COMMIT_EDITMSG` / `FETCH_HEAD` / `logs/**` 这 9 个
本地指针与日志文件（记录「上次检出到哪」，不含独立内容），以及那 6 个指针里 `refs/remotes/origin/a3-cookie-r27`
一项（它指向的 `771eff1…` 已在 bundle 的 refs 里）。详见第 8 节。

E 档构成（397 个 / 1.392 GiB）：`chk3_*` / `chk4_*` 自测检查点 32 个（1.379 GiB，light_state v3 自测产物，
不是训练结果）、上传探针切片与 scratch 日志 321 个（KB 级）、空文件 26 个、打包中间件 13 个、
启动日志 2 个、代码打包 1 个、已归档 npz 的派生切片 1 个、上传缓存 1 个。

重点项之外仍需注意的：`fasterwam-a3-cookie/README.md` 的 pod 版本（22,630 B）只存在于
`fasterwam-a3-cookie_worktree.diff` 的补丁文本里（C 档，补丁本身现已进仓库），当前仓库 README 是它的升级版。

## 2. GitHub 与 README 远端验收

以下都是对远端做只读查询的结果，不是本地 `git status`：

| 检查 | 结果 |
|---|---|
| `archive-2026-11-01` 远端 tip | `2170c614ec71c04e615fd698adafd348ea9ea01c`（`git ls-remote` 读回）——存入 bundle 与 73 条审计表的那次提交；把最终数字写进本报告的提交在其之上（sha 见对话） |
| 此前的文件提交 | `f40ab89cd5347750c5a1809a859d95d509d463fc`——把 9 个文件转成 B 的那次提交，父提交是要求核对的 `b82854f` |
| 分支列表 | `a3-cookie-r27` `c678beac…`（默认分支，只加了两次 README 提交）、`archive-2026-11-01` `2170c614…`、`backup-2026-11-01-r6-audit` `1fb28037…` |
| 远端 README 是否最新 | 是。推送后在全新 clone `_clone_verify6` 里读到第 11.3 节（上传手册）与第 11.4 节（GitHub 上传经验）都在；本次又修正了三处失真见下 |
| **默认分支 README** | **已更新，且只改了 README**。默认分支是 `a3-cookie-r27`（`git ls-remote --symref` → `ref: refs/heads/a3-cookie-r27`）。原 README 26,051 B、最后改动 `66f841c`（2026-10-30）、没有第 11 节；本次以普通快进推送 `66f841c..c678bea` 替换为最新版（+390/−358 行，另含一行交叉引用修正），最终 README blob `c641a579…`，该分支其余 238 个文件一个字节没动 |
| 恢复脚本可执行且正确 | `scripts/restore_new_instance.sh` 远端 blob 13,973 B / `745e20e7…`（与 README 第 11 节钉的值一致） |
| HF Model Card | 已提交且可远端读取：`resolve/main/README.md` 200，5,228 B，含 YAML front matter 与「已知缺口 / 数据与许可 / 校验口径」 |
| HF Dataset Card | **不存在独立 dataset 仓库**（`/api/datasets/...` 401）。数据以 `data/**` 放在模型仓库内，并在 Model Card 的「数据与许可」里说明 |
| HF revision | `refs` 与 `tree?expand=true` 一致：`main = f2aaf55c6ce1b164de97c5b6ec2cd75e942d0de0`（2026-11-01T08:48:11Z，"Archive M batch 1/1"）。注意 `GET /api/models/...` 会返回 2026-10-26 的陈旧缓存体（sha `dc10a5d4…`、9 siblings），不能当依据 |
| README 链接真实性 | 逐目录实测：`weights/r4`、`weights/r5`、`parts60gz`、`data`、`eval/a3_eval`、`eval/bridge`、`videos/*`、`runs`、`replay/bridge`、`tests`、`research`、`MANIFEST` 全部 200；**`weights/r2`、`weights/r3` 是 404**——台账里这两行是目标路径，远端只存了分片。README 本次已改 |
| 台账逐行对远端 | 715 行台账，0 行缺失：712 行按 path 命中，3 行由 532 个分片装配（178 + 177 + 177，索引连续 0..N） |

提交 sha 无法写进提交自身：默认分支的最终 tip `c678beacf3db47b2bb92512527a5b1cd33eff1f5` 是固定的；
归档分支上 `2170c614ec71c04e615fd698adafd348ea9ea01c` 承载了 bundle、73 条审计表与第 7、8 节，
带着这一版修正的提交是它的子提交（也就是本文件当前所在的提交），以 `git ls-remote origin` 或 `git log -1`
读回的 sha 为准，并在对话里报出。

### 2.1 默认分支的改动范围与提交

- 归档分支相对默认分支 `66f841c` 的差异：**984 个文件 = 982 个新增 + 2 个修改**（`.gitignore`、`README.md`），
  +132,473 / −370 行（`git diff --name-status 66f841c <archive tip>`）。新增的全部落在 `scripts/`、`pod_snapshot/`、
  `archive/2026-11-01/`、`docs/`、`research/`、`tests/` 这些归档目录。
- 默认分支上没有任何运行代码的改动，只有两次 README 提交，都是普通快进、没有 force、没有改写历史：
  1. `66f841cccf5d01204a5aba36a6e44b170aa02a21` → `39f47015b82f564a2f6eb70bd386e588c01110eb`
     （2026-11-01 19:13:44 +0800，"docs: bring the README up to the 2026-11-01 research state"，
     `git show --stat` 只有 README，+390 / −358）；
  2. `39f47015…` → **`c678beacf3db47b2bb92512527a5b1cd33eff1f5`**（2026-11-01 19:31 +0800，
     "docs: point the A-F table reference at the report it actually lives in"，只改 1 行）。
- 远端回读：另开目录全新 clone（`_clone_default_verify2`）到 `c678bea`，`git ls-tree HEAD README.md` = blob
  `c641a579eb4b6b5c833b1c28ab438921575b9528`，与本地相同；该分支共 239 个跟踪文件，
  `git diff --stat 66f841c HEAD` 只有 README，父链完整保留到最初的 `b1c450a`。
- 默认分支 README 与归档分支 README 有三处**刻意**不同：开头的说明段（默认分支写明归档材料在
  `archive-2026-11-01` 分支）、第 11 节恢复第 1 步的 `git clone -b archive-2026-11-01 …` 命令、
  以及交叉引用措辞。其余正文（含第 11.3、11.4 节与第 13 节未完成清单）逐字相同。

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
| 密钥与敏感信息 | GitHub 22 个提交全历史 `git grep` 扫描：HF token / GitHub PAT / AWS AK / 私钥 / `sk-` 均 **0 命中**；HF 1,264 个文件名扫描 token/secret/.env/credential/id_rsa/api_key/.pem **0 命中**；本次新增 11 个文件 0 命中（bundle 另按其 11 个提交全历史扫描，同样 0 命中） | 见本报告附录命令 |

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

### 5.1 第四项新增的归档文件

| 路径 | 字节 | sha256 |
|---|---|---|
| `archive/2026-11-01/fasterwam-a3-cookie_history_2026-11-01.bundle` | 524,674 | `c7946637a91f0059fe5b8ba43fa16c65abdf0c4b7ce0451e6799e34cf5d919a3` |
| `archive/2026-11-01/corpus73_audit.tsv` | 29,382 | `5e010119e1828bc27dbf7d227a46fb10fd2000234c0827e0ba5149bf90b31519` |

bundle 是第 2 项要保存的旧 pod Git 历史（第 8 节）；TSV 是第 1 项 73 条示范的逐条核对结果（第 7 节）。
两个文件原样从本地证据目录复制后提交，推送后在全新 clone 里复核 sha256，并重新跑一次 `git bundle verify`。

## 6. 仍需你做的两件事（都不是数据阻断项）

1. 到 HF 后台吊销本会话用过的 token。
2. 确认 `Eter0109/a3-front-close-left-100` 与 FasterWAM / Wan-AI 权重的再分发许可；确认 HF 仓库保持公开还是转私有。

实例终止与工作盘清理由你本人执行。pod 上的上传缓存 `/root/hfstage`（约 88 GB）、`/root/drill*` 演练目录、
`_r6_stage` 临时脚本，以及本地 `_clone_verify*`、`_clone_default*`、`verify_clone.git` 等验证用副本都还在；
清理清单一事一列等你批准，本轮没有删除任何东西。

## 7. 73 条原始示范的逐条核实

先把三样东西分开，它们不能互相顶替：

| 对象 | 数量 | 字节 | 位置 | 本轮结论 |
|---|---|---|---|---|
| **原始轨迹 NPZ**（`corpus_manifest_v1.json` 的 `entries` 逐条列出的 73 条） | 73 | 7,657,304 | HF `eval/a3_eval/demos_*/**.npz` | 逐条真实下载回读，**73/73 逐字节相同** |
| 派生训练数组（`data/a3_fw*`） | 14 | 13.436 GiB | HF `data/a3_fw_*` | 早前已按 sha256 归档（A 档）；它们是**从原始轨迹派生**的训练语料，不能当作原始轨迹的备份 |
| manifest 本身 | 1 | 41,118 | HF `eval/a3_eval/corpus_manifest_v1.json` | sha256 `b679fa6bd07e12c3be25e02f1d5aa3d14e3d785fe726e04f98b1f3a391dd0e00`，与本地逐字节相同 |
| 16 个裁定来源 JSON（`sources`） | 16 | 每个 0.9–14 KB | 同目录 `sources/`（C1..C7 各有 c3/correct_v2 两份，wrongclosure v8/v9 各一份） | 16/16 与本地逐字节相同 |

- 方法：按 manifest 的 `hf_path` 走
  `https://hf-mirror.com/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/<path>` 下载每个 NPZ，
  本地算 sha256 与 manifest 的 `npz_sha256` 比对——不是查上传记录，也不是只比大小。
- 结果：**73 条全部逐字节一致，没有一条需要补传**；逐条明细（路径 / 字节 / sha256 / HF 路径 / 回读 sha256 / 判定）
  在 `archive/2026-11-01/corpus73_audit.tsv`（73 行 + 表头，29,382 B）。
- GitHub：这 73 个 NPZ 从未进过 git（`.gitignore` 排除 `*.npz`），远端 clone 里 0 命中——不是缺口，HF 上已有逐字节副本。
- Windows：本机原先没有这些原始轨迹（对 5,753 个本地文件做路径 + sha 扫描，0 命中）；现在本机唯一的副本是
  本次审计下载的 `_r6_stage/audit73/npz/`，它是证据副本，不是备份来源。
- 家族分布（manifest `tally`，按录制驱动写进 `retry_log[0]` 的异常类型划分，互不重叠）：
  C1 压得太深 17、C3 闭合但空 18、C2 错误闭合 8、S1 单块低力 30。

按目录（每条明细见上面的 TSV）：

| 目录 | 条数 | 字节 |
|---|---|---|
| demos_correct_c1 | 2 | 219,874 |
| demos_correct_c2 | 6 | 1,073,822 |
| demos_correct_c3 | 6 | 1,034,478 |
| demos_correct_c4 | 5 | 904,056 |
| demos_correct_c5 | 6 | 964,184 |
| demos_correct_c6 | 4 | 719,637 |
| demos_correct_c7 | 6 | 1,345,531 |
| demos_single_v30b | 30 | 901,554 |
| demos_wrongclosure_v8 | 2 | 119,249 |
| demos_wrongclosure_v9 | 6 | 374,919 |
| 合计 | 73 | 7,657,304 |

## 8. 旧 pod 独有 Git 历史（8c2ef40d）的保全

- 现状：归档分支**不含** `8c2ef40d`。GitHub 默认分支 tip 是 `66f841c`，它的祖先链包含 `771eff1`；
  pod HEAD `8c2ef40d` 的父提交正是 `771eff1`，两者是**兄弟**关系——所以这个提交及其树、文件内容从来没进过 GitHub。
- 已做：在 pod 仓库里 `git bundle create … --all <未达对象 sha…>`，把
  `refs/heads/a3-cookie-r27 8c2ef40d…`、`refs/remotes/origin/HEAD 771eff1…`、
  `refs/remotes/origin/a3-cookie-r27 771eff1…`、`HEAD 8c2ef40d…` 四个引用、全部 11 个提交与 384 个对象一起打包。
- 不可达对象也收了：pack 里除 380 个可达对象外还有 4 个不可达对象（3 棵树 + 1 个 blob）——是 `bridge/` 下一次
  `git add` 留下的暂存树与索引快照（其中 `_next/README.md` 有 40,452 B）。它们与已归档的
  `fasterwam-a3-cookie_worktree.diff` 内容重叠，但既然能收就一并收进 bundle。
- 校验：
  - `git bundle verify` → `is okay`、`records a complete history`；
  - 从 bundle 全新 clone：11 个提交、384 个对象，tip `8c2ef40d`（2026-10-31 03:50:10Z），父提交 `771eff1`，
    `8c2ef40d` 的树是 `b9d5e7ed7b435cb564f4e71fba5a39cdce7df4a9`；
  - **与 pod 逐对象对拍**：pod `.git/objects` 下 384 个 loose object 的 OID 与 bundle 内 384 个对象的 OID 集合
    完全相同（缺 0、多 0）——这是第 1 节 F 档里那 384 个对象已获外部副本的直接证据。
- 落盘与外部副本：bundle 524,674 B、sha256
  `c7946637a91f0059fe5b8ba43fa16c65abdf0c4b7ce0451e6799e34cf5d919a3`，提交进私有仓库 `archive-2026-11-01`
  分支的 `archive/2026-11-01/` 下；推送后在全新 clone 里复核 sha256 并重跑 `git bundle verify`。
- 没有上传的：`.git/config`（含远端 URL 与本地配置）、`index`、`HEAD`、`FETCH_HEAD`、`COMMIT_EDITMSG`、`logs/**`。
  bundle 格式本身只装对象与引用，不装 config / reflog / hooks。
- 敏感信息：对 bundle 的全部历史跑 `git grep`（`hf_`、`ghp_`、`github_pat_`、`AKIA`、`PRIVATE KEY`、`sk-`、`glpat-`）
  全部 0 命中；唯一相关的文本是文档里的 `HF_TOKEN=...` 占位符。
