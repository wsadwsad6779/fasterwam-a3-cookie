# Third-party notices

This archive redistributes code snapshots taken from the pod. Every upstream tree
kept here is under a license that permits redistribution; the license texts travel
inside the snapshot itself. Nothing here is a claim of ownership over that code.

## Code included in `archive/2026-11-01/git_snapshot_20261101.tar.gz`

| Tree | Upstream | License | License file inside the snapshot |
|---|---|---|---|
| `FasterWAM/` | https://github.com/hustvl/FasterWAM | Apache-2.0 | `FasterWAM/LICENSE` |
| `FasterWAM/third_party/RoboTwin/` | https://github.com/RoboTwin-Platform/RoboTwin | MIT, © 2025 Tianxing Chen | `FasterWAM/third_party/RoboTwin/LICENSE` |
| `a3_dual_arm_sim/` | https://github.com/Eter0109/a3_dual_arm_sim | MIT, © 2026 Eter0109 | `a3_dual_arm_sim/LICENSE` |
| `a3_dual_arm_sim/assets/a3/` | robosuite model assets | see `a3_dual_arm_sim/assets/a3/ROBOSUITE_LICENSE.txt` | 同目录 |

`fasterwam_a3_bridge/`, `_r4_stage/`, `_r5_stage/`, `_r6_stage/`, `_selftest/`
以及根目录的脚本是本项目自己的东西。

## Code NOT redistributed here

The uncompressed checkouts under `pod_snapshot/2026-11-01/FasterWAM/` and
`pod_snapshot/2026-11-01/a3_dual_arm_sim/` are excluded from this repository
(see `.gitignore`). They are byte-identical to what is inside the tarball above,
only unpacked. Restoring from the tarball is enough.

## Data

- 原始第三方专家数据集 `Eter0109/a3-front-close-left-100`（revision
  `8fdd8a7caf8275eba8be8a83b53a99176a084062`）**只引用、不重传**。恢复脚本默认跳过它，
  需要时设 `WITH_DATASET=1`，再分发前必须拿到数据所有者的许可。
- `data/a3_fw*`、`research/**` 下的内容是本项目在本机采集/转换/训练的产物。
- 运行时依赖（mujoco、robosuite、transformers、torch 等）通过 pip 或实例镜像获得，
  本仓库和 HF 归档都不包含它们的二进制。
