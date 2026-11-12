# RESCUE_1113 — 实例销毁前的代码抢救快照（2026-11-13）

来源是本地工作树 `C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\`，不是 pod。
基线是 `a3-cookie-r27` @ `b59bd6e7ac95b2074c281340984097ed996cc816`；`main` 在它之上追加 285 个文件
（`stage_40k/` 212 + `local_1113/` 71 + `RESCUE_1113.md` + `.gitattributes`），没有改写 r27 已有的任何 blob。远端此前没有 `main` 分支，默认分支仍是 `a3-cookie-r27`。

推送时的通道状态：Runpod 实例 `rgapi-one-click-24316-57b6b870` 的 JupyterLab 入口自 2026-11-12 起返回
`403`（证据见 `stage_40k/dev/eval40k/ROOT_CAUSE_40K_20261112.md` §5 末尾的 2026-11-13 复查段），
pod 上的文件已读不出来，所以本快照只能覆盖"本地仍有副本"的那部分。

## 里面有什么

- `stage_40k/`（212 个文件）= `_v1_stage/` 这棵 40k 时代代码树的原样副本：
  - `fw/src/fasterwam/`：力感知模型接线（`models/wan22/force_condition.py`、`light_state_v3.py`）、
    lerobot 数据集与加载器、`base_load_guard.py`
  - `fw/scripts/`：分段训练驱动 `drive40k.py` / `train_seg.py`、备份 `backup40k.py` / `watch_backup40k.py`、
    发布 `export_merged_v3.py`、校验 `segment_check.py` / `verify_dataset_force.py` / `verify_publish65.py`
  - `fw/configs/`：`train.yaml` 与 data/model/task 配置
  - `dev/eval40k/`（33 个 .py + 3 个 .sh）：评估 `eval40k.py` / `batch_eval.py` / `report40k.py`、
    录像 `record40k.py`、面板合成 `panelvideo40k.py` / `panel_sample40k.py`、
    四道 CPU 闸门 `sensor_identity40k.py` / `sensor_identity_tamper40k.py` /
    `panelvideo_selftest40k.py` / `panelsink_selftest40k.py`、
    启动器 `RUN_RECORD_SEEDS.sh` / `RUN_BATCH_STEP6.sh` / `RUN_ON_POD.sh`、
    根因报告 `ROOT_CAUSE_40K_20261112.md`
  - `dev/p4_shadow/`、`dev/p5_dagger/`、`dev/_probe_1112/`（含 `install_spec_round*.json` 安装规格）、
    `dev/out40k/panel_layout_sample/`（版式样本的 meta + PNG 裁片）
  - 根级散件：`smoke_force_v1*.py`、`build_v1_dataset.py`、`make_force_calib.py`、各类 probe 脚本与
    `docs_*.md` 冻结文档
- `local_1113/`（71 个文件）：本地根目录的报告 `*.md`；`tools/` 里的工具脚本和从 pod 回读过的线上代码副本
  （`pod_trainer.py`、`pod_light_state.py`、`pod_r4_patch.py`、`rm_local.py`）；
  `ledgers/` 里的 HF 备份台账（`backup_manifest*.json`、`final_reconcile*.json`、`upload_batches.json`）、
  `r4_loss_curve.json` 与 `pod_r4_train_log.log`

## 字节一致性（为什么恢复后哈希闸门还能命中）

`.gitattributes` 对 `stage_40k/**`、`local_1113/**`、本文件设 `-text`，git 不做换行转换。
提交前把每个入库 blob（`git show :<path>`）和磁盘文件逐一对比 sha256：**285/285 相同**。
所以 `RUN_RECORD_SEEDS.sh` 里 0a（7 个计分文件）+ 0b（10 个录像文件）共 17 个 sha256 前缀闸门，
从 `main` 恢复后应当仍然命中；台账在 `stage_40k/dev/eval40k/_panel_hashes_1112.json`。

## 换机器后的自查（只用 CPU，不需要权重）

```sh
git clone -b main https://github.com/wsadwsad6779/fasterwam-a3-cookie.git
cd fasterwam-a3-cookie/stage_40k/dev/eval40k
py -3.13 sensor_identity40k.py          # 期望 39/39, rc=0
py -3.13 sensor_identity_tamper40k.py   # 期望 7/7 XML 篡改被拒 + 4/4 按压判例, rc=0
py -3.13 panelvideo_selftest40k.py      # 期望 52/52
py -3.13 panelsink_selftest40k.py       # 期望 30/30
bash -n RUN_RECORD_SEEDS.sh             # 启动器语法
```

`flow_test.py` 需要 torch，可先在有 torch 的机器上跑；`proprio14_test.py` 只用 numpy。

## 故意不在这里的东西

- **权重**：不进 GitHub。8 个训练节点（5k…40k）与语料的落点记在
  `local_1113/ledgers/backup_manifest_v2.json` 和 `local_1113/BACKUP_MANIFEST_v2.md`，实体在 Hugging Face。
  本分支不含任何 token、cookie 或 pod 凭证；提交前用凭证模式（`hf_…`、`ghp_…/github_pat_…`、
  `AKIA…`、`Bearer`、`token=`/`password=` 赋值）扫过全部待入库内容，命中数为 0。
- **大二进制**：仓库 `.gitignore` 排除 `*.mp4 *.npz *.pt *.zip`，所以面板版式样本的合成 mp4 不在分支里；
  它的分辨率、fps、通道映射由 `LAYOUT_SAMPLE_meta.json` 加 6 张 PNG 裁片固定，够重建。
- **一次性传输碎片**：`*.b64`、`p00.txt`…`p15.txt`、`_h1`/`_h2`、`parts/*.gz.pN`、`__pycache__`、
  `*.tar.gz` 未收录——它们是同一批代码的分片副本，不是另一份代码。
- **只有 pod 上才有的东西**（我无法从本地证实或否认，因此按"会随实例丢失"处理）：
  40k 八节点批评估（160 回合）的结果文件、被 patcher 在 pod 就地改过而从未回读的代码、
  pod 上构建出的 dataset/T5 文本嵌入缓存，以及三份 `merged.pt` 里除已归档者之外的合并产物。
  重建路径在代码里（`build_v1_dataset.py`、脚本 `export_merged_v3.py`），前提是 HF 上的 checkpoint 可下载。

## 已知未收口的判断

`_panel_hashes_1112.json` 里 `force_stream.py`、`rollout40k.py` 的 `ever_pushed_to_pod` 是 `unknown`
（pod 上是否存在同名字节无法确认），其余 9 个录像侧文件从未上传过 pod——它们此前只有本地一份副本，
本分支是第二个副本。录像前置项 5（step_040000 的 seed 11/17 面板录像）始终没有执行，
因为通道 403；它需要的是 pod + GPU + 权重，不是这段代码。
