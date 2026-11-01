# 2026-11-01 归档快照说明

本目录是退租前的本地归档，配套分支 `archive-2026-11-01`。仓库是私有的，仅作存档，不对外分发。

## 1. 内容

| 路径 | 内容 | 字节 |
|---|---|---|
| `p0p1_manifest.tsv` | 715 行归档台账：档（P0/P1）、`path_in_repo`、pod 相对路径、字节数、sha256 | 125,559 |
| `parts60gz_manifest.tsv` | 三个 >2 GiB 权重的 60 MiB 分片台账（532 行，每片是 `part_NNN.gz`，同上五列；从 HF `MANIFEST/` 下载） | 110,662 |
| `parts60_manifest.tsv` | 同一批分片的裸字节台账（532 行，切分时生成；实际入库的是 gz 版本） | 105,342 |
| `batch_plan_summary.txt` | P0/P1 分批与总量统计 | 1,153 |
| `upload_batches.json` | 每批的文件列表与字节数（供重跑上传用） | 1,326,865 |
| `git_snapshot_20261101.tar.gz` | 旧 pod 上工作树的 git 快照 | 11,175,131 |
| `git_snapshot_verify.json` | 快照解包后逐文件 sha256 对拍结果 | 224 |
| `uploader/` | 实际使用的上传脚本（`hf_archive.py`、`run_slice.sh`、`go3.sh`、`verify_git_snapshot.py`、`split60.py`） | — |

`p0p1_manifest.tsv` 的 sha256：`54f4e3669a46b3af58331abee2b55c350c1292516b692c19cb4bb0520d596b08`
`parts60gz_manifest.tsv` 的 sha256：`307172a4f82d98f58391068c2ac4078ac4f8ff64efafa50470361456db6d0f04`（532 行、26,778,973,710 B）
`git_snapshot_20261101.tar.gz` 的 sha256：`53fc19b33396ee96e3448e556beffd2a4fa799131d5404c7b213aafb80cb8b4c`
`upload_batches.json` 的 sha256：`b3aba14ca8ad47c0efac7812e3a34409d1266b1f8342f1d84641f55548c677af`
`batch_plan_summary.txt` 的 sha256：`9302567875069d70d00b4106d44583968955a46c23f359a0ee654befdee2c107`

## 2. pod_snapshot

`pod_snapshot/2026-11-01/` 是旧 pod `/workspace` 下代码与脚本的逐文件副本，2,003 个文件、39,077,618 字节，逐文件 sha256 与原 pod 对拍通过（`rows=2003 ok=2003 missing=0 sha_bad=0`）。它不含权重与数据集，只覆盖代码、训练/评估脚本、配置与研究报告。

其中 `FasterWAM/` 与 `a3_dual_arm_sim/` 两个上游 checkout（1,054 个文件、约 36 MB）不进 git（见 `.gitignore`），因为它们与 `git_snapshot_20261101.tar.gz` 里的内容逐字节相同，没必要把同一份第三方代码放两遍；解包 tarball 即可得到。我们自己的 `fasterwam_a3_bridge/`、`_r4_stage/`、`_r5_stage/`、`_r6_stage/`、`_selftest/` 与根目录脚本全部照常入库。

## 3. 许可与再分发

| 来源 | 许可 | 处理 |
|---|---|---|
| `FasterWAM`（hustvl） | Apache-2.0 | 代码快照保留 LICENSE；再分发遵守 Apache-2.0 |
| `FasterWAM/third_party/RoboTwin` | MIT，© 2025 Tianxing Chen | 同上，LICENSE 随快照 |
| `a3_dual_arm_sim`（Eter0109） | MIT，© 2026 Eter0109 | 保留 LICENSE 后可再分发；本仓库只放快照，不再分发整包 |
| `Eter0109/a3-front-close-left-100` 数据集 | 未声明 | **不复制进本仓库也不上传到 HF 公开仓**，只记录 revision 引用 |
| Wan-AI `Wan2.2-TI2V-5B` / `Wan2.1-T2V-1.3B` tokenizer | 各自模型卡许可 | 不重传，恢复时按上游地址取 |

HF 模型仓 `wsadwsad6779/fasterwam-a3-cookie-lora` 里承载的是本项目自己产出的权重、训练状态、评估产物与录像；第三方权重与数据集一律只引用不重传。若后续要公开该仓，需要先确认上表「未声明」两项的再分发许可。

## 4. 与本次归档相关的已知限制

- 三个 >2 GiB 权重只能以 60 MiB 分片、每片再 gzip 压缩后上传（Xet 上传器在服务端 dedup 返回 `found` 后静默挂死，压缩改变内容即可绕开），恢复时 `cat part_*.gz | gunzip` 重装配并比对整文件 sha256。
- r4/r5 轻量 checkpoint 缺 `proprio_encoder`（36,864 元素），本次备份无法补回。
- 分片上传与回读校验的最终状态记在 `PRE_SHUTDOWN_FINAL_REPORT.md`。
