# 2026-11-01 科研成果冻结备份（第一阶段）

来源：Runpod 风格 GPU pod `rgapi-one-click-24316-57b6b870` 的 `/workspace`，
采集时间 2026-11-01T03:18:53Z（枚举）～ 03:30:14Z（哈希）。
本分支 `backup-2026-11-01-r6-audit` 基于 `a3-cookie-r27`（GitHub 侧 `66f841c`），
只新增 `backup/2026-11-01/` 目录，不改动任何既有文件、不覆盖历史结果。

## 里面有什么

| 文件 | 作用 |
|---|---|
| `BACKUP_MANIFEST.md` | 963 个条目的清单：绝对路径、字节、mtime、完整 sha256、版本、是否被现役配置引用、类别、其他已验证副本 |
| `backup_manifest.json` | 同一份清单的结构化版本（含按类别汇总与 pod 内同 sha 重复组） |
| `docs_light_state_param_sets_audit.md` | 轻量 checkpoint 参数集合审计报告（本轮结论的来源） |
| `docs_task1..task5*.md` | 本轮五个任务的方案与决策文档 |
| `repro/` | 复现证据：commit、分支、工作区 diff、`pip freeze`、python/torch 版本、磁盘与目录清单 |
| `pod_code/bundle_code_r6_stage.txt` | pod 上 `_r6_stage/` 的 16 个脚本（含 `light_state_v3.py`、合成测试、只读探针）逐字节打包 |

## 通道与校验方法

pod 只能经浏览器里已打开的 JupyterLab 页面读写（无 ssh）。文本文件用
`GET/PUT /api/contents/<path>` 按行分块写入，回读时按 100000 字符切片取回、
本地 `json.loads` 还原后拼接，再与 pod 端 `sha256sum` 记录逐字节对拍。
本轮通过该方式取回并校验通过的三份：

- `_r6_stage/out/backup_filelist.json` → `264034ac7fed097af08820ee49035da4945c3ae78f5be494464a963e13d8d5bc`
- `_r6_stage/out/backup_hashes.jsonl` → `d58949ae66c851d37dc62d7e7640b28a9b6a8c5d145fbb4d466c03d53098dc2f`
- `_r6_stage/out/repro/cookie_diff.patch` → `4e9a680f4a813beec4c9d72f5b4953dc2cb31e2303ac6e21c81d36653ee487a4`
- `_r6_stage/out/repro/bundle_code_r6_stage.txt` → `933c228ca5e0ad2b290411ae4087a054288f5d64cf8e3d2075a5833c49a4a0d3`
- `_r6_stage/out/repro/bundle_repro_evidence.txt` → `be76c8cbeccca3d3b29fcb4864541e55e9008384ca77224b1c7330f7048dd547`

`bundle_repro_evidence.txt` 整体与 pod 端哈希一致，但其中内嵌的 `cookie_diff.patch`
一份是 UTF-8 解码后再编码的副本（pod 上该文件含多字节字符），所以本目录里发布的是
单独取回、逐字节对拍通过的那一份；bundle 里的那一份只作同捆留档，不要拿它做 diff 依据。

## 历史限制（引用本备份时必须一起带上）

> r4/r5 训练末态的 `proprio_encoder` 权重（2 张量 / 36,864 元素）在 pod 上没有任何副本，
> 本次备份不能恢复它；备份的是 LoRA 与 r3 基座，历史评估组合可复现，训练末态模型不可复现。

## 还没备份出去的部分

`.pt / .npz / .mp4 / .parquet / .dat` 这类大文件按仓库约定不进 git（`.gitignore` 已排除），
清单里共 47.88 GiB，其中已在 HF 公开仓库的 `step_002000_merged.pt`(11,117,977,659 B)、
`data/frames.dat`(12,849,371,136 B) 等只有**大小**在 hf-mirror 侧对上了，
HF 没有把 sha256 暴露在镜像 API 里，逐字节复核要么走 26 MB/s 的分段下载要么走上传时回读。
pod 侧 `git push` 不可用（`.git` 里是 SSH remote 且 pod 上没有凭据），
`huggingface.co` 从 pod 直连可达（实测 307，0.45 s），所以下一步需要
一个有写权限的 HF token 才能把 `r3 merged + r4/r5 末态 + 原始 100 条示范 + 评估产物`
推到 `wsadwsad6779/fasterwam-a3-cookie-lora`。在这些上传逐文件比对过完整 sha256 之前，
不能把它们记为「已备份」。

## v2 补扫（同日，全量对账后追加）

第一版清单只覆盖了我们想到的目录。随后用一次全量 `find /workspace -type f`
（4,933 个文件）与清单做差集，发现 **3,970 个文件、47.47 GB 从未被记录**，
其中包含 r3 的完整训练状态、r4/r5 的全部中间 step、两批 dagger 语料与它们的未转换原始帧。
补扫后清单为 **3,888 个条目 / 98,876,695,253 B（92.09 GiB），每个条目都有 pod 端完整 sha256**。

新增文件：

| 文件 | 作用 |
|---|---|
| `BACKUP_REPORT.md` | 核验报告：远端路径与权限、逐字节对拍结果、10 类覆盖度、未备份部分、授权偏差 |
| `BACKUP_MANIFEST_v2.md` | 全量清单的目录级/类别级视图 + 尚无外部副本的大文件表 |
| `backup_manifest_v2.json` | v2 逐条目（3,888 条，含 `backup_status`、`other_verified_copy`、重复组） |
| `manifest_rows.tsv` | pod 端合并表原样（path/bytes/mtime/suffix/sha256/referenced/groups/hash_seconds） |
| `pod_code/scan_checkpoints.py` | 补扫脚本（差集入单，按路径规则分组） |
| `pod_code/manifest_rows.py` | 把 filelist + hashes 压成一张 TSV 供回传 |
| `pod_code/build_backup_manifest_v2.py` | 本地重建 v2 清单与报告的脚本（入口先断言输入 TSV 的 sha256） |

新链路的校验锚点：`manifest_rows.tsv` pod 端与本地重组后同为
`62afef81caed177f1746af7c5eec42dc1c30dbbc59caabd23a9084b94d7d782e`（699,851 B / 3,888 行）。
脚本双向对拍：`scan_checkpoints.py` = `b685830b8703487b1697e7da01aac1cc141c0ad57c8d535cef28d5e38629ad03`，
`manifest_rows.py` = `785f92968a52ae1f6339d7a109e8615f96e41e90a23b7857d9f3da8ebe5146c2`（pod 与本地一致，且 `py_compile` 通过）。

v1 的文件不改动、不覆盖；`backup_filelist.json` 补扫前的快照留在 pod 上作
`_r6_stage/out/backup_filelist.pass1.json`（`264034ac…`）。
本节里「清单里共 47.88 GiB」是 v1 口径，已被 v2 的 92.09 GiB 取代；
上一节关于 HF 大小对拍、pod 无法 push、需要写 token 的结论仍然成立。
