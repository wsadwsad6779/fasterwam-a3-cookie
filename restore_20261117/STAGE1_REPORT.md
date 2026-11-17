# 阶段 1 报告：40k 原始资产恢复成果保存与 GitHub 差异审计

本轮目标：把云端 GPU pod 上抢救回来的 40k 推理资产、恢复脚本与取证清单，按字节身份固定到独立分支，
并完成 pod 实际源码与 GitHub `origin/main` 的差异审计。大模型权重继续留在原始 HF 归档，
GitHub 只保存代码、配置与身份清单。本报告所有数字均由本地实算复核，非记忆转述。

- 仓库：`https://github.com/wsadwsad6779/fasterwam-a3-cookie`
- 保存分支：`rescue-40k-restore-20261117`（从 `origin/main` = `e39e62f060d7201ad010bca0f869a020e377f44c` 切出）
- 新增目录：`restore_20261117/`（158 个文件，其中 155 个纳入 Git，3 个 `.npz` 测试产物按 `.gitignore` 留外部）
- 约束：不 force push、不覆盖历史；是否合入 `main` 待用户确认后再决定。

---

## 1. 身份哈希固定

权威清单：`restore_20261117/identity_small_sha256.txt`（27 条）。
本地复核命令与结果（在 `restore_20261117/` 内执行）：

```
sha256sum -c identity_small_sha256.txt
```

结果：**18 条 OK（本地逐字节一致）+ 9 条 FAILED open or read（外部资产，仅记录哈希）**，0 条哈希不符。

### 1.1 本地逐字节核验通过的 18 条（已纳入本分支）

| 文件 | sha256 |
|---|---|
| `gpu_shortest_action.py` | `b42c48f0fbc4ab6cecff246992c4452542c4fe5769ed85e522e56f8cb4e8ecd5` |
| `gpu_shortest_action.log` | `e29f3f130b4e07446f1d0e615e3a5d323b5cf5263915e1f40d017b46bd3fa8a7` |
| `gpu_shortest_action_run2.log` | `9f20b667c88a609b8f97ffbc5dfb6787090abf2af06ba6d0ebb3557698061307` |
| `gpu_shortest_action_run3.log` | `40a5aeb49665a6a815d9dbfe16023132433c642b45f31a894b4b6c597476e156` |
| `GPU_SHORTEST_STATUS.md` | `aae83bc2cdc5c937601a77fb3bc7d616c12dcf1053b0360387d463101b2f1581` |
| `step_040000/run_config.yaml` | `4aa66c57cdfe2804ae56cf244a907be9e13fe232d4ede65ba7ef0695eb79abfd` |
| `step_040000/manifest.json` | `54085b15f00b4e0e6a5850a92917678fb89ad3467dc1f5580a7dfa9d38f67e32` |
| `step_040000/trainer_state.json` | `9b1c150fb01f6c22c9fe0aa73245f819ba5a3d276aa446ff6ed09146a170e566` |
| `step_040000/base_weights.json` | `06d4e629c23ef7bde224ef893d60b6bb22de24a9ad7e4d4c5bdaf627eed43f5c` |
| `step_040000/force_calib_v1.json` | `2e821a8a019f966d0d2418df7f587d9d469ccd5596abcc39e9f20f5f830d1bd5` |
| `cpu_restore_acceptance.json` | `361a5fc9f5cce4d2bd4c367a8c68303bb0229142df14a1b509780a357f867848` |
| `transfer_manifest.json` | `d34ee695bd1d28a41f2799c59b923d0ac38d1bf0b1a702d1207f0e43b3ef30b8` |
| `export_merged4.log` | `4ced32f7c9aa163fff1487220deb73c126020e4eadfaed6addb923f14fe6ad0c` |
| `pod_fw_sha256.txt` | `de463bf1dc9ec30bd2cc347f854dbb61539872507e7b2bad512dbc8bf4bdc865` |
| `derived/dataset_stats_14d_DERIVED.json` | `354f59738d96e6962a0e1705635c5bf21c87c076c72bb512344c423093765b51` |
| `derived/provenance.json` | `b834a8e57fa24bad95cab58857fbb5c9332ed4df4949cba8166ca27eaccaef0a` |
| `archive_evidence/data/a3_fw/dataset_stats.json` | `ed833168725b0b63179a8aa3f870543f119b6891305fb94d17b3357989ccfffd` |
| `archive_evidence/data/a3_fw_force/force_stats.json` | `6333ed02482cba1fc36e38c4ddd11a6442b1d411591ece964b84d9417daec15c` |

### 1.2 外部资产：仅记录哈希，字节留在 HF 归档 / pod（9 条）

这些是大模型权重、二进制语料与 HF 辅助分词器，按用户要求**不进 GitHub**，只在清单里固定身份：

| 文件 | sha256 | 字节 | 存放 |
|---|---|---|---|
| `step_040000/adapter.pt` | `1e19f1e3f81f96b770311608fbff00a3ec6739c6346b1e464af41baf491a3852` | 70,257,165 | HF 归档 / pod |
| `step_040000_merged.pt` | `d118eb2172b04165990c032e291057aaa9313b1d9fd0e6cd7a9ada3637c9cc20` | 11,126,493,579 | pod（base+LoRA+force 合并） |
| `base_step_021700.pt` | `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d` | 11,117,679,929 | HF `hustvl/FasterWAM` libero 基座 |
| `archive_evidence/data/a3_fw/arrays.npz` | `b5468c274cc5eede761242478750e13e0a58529dba9b5400ccd2872f7f9fc7f0` | — | HF 归档 / pod |
| `archive_evidence/data/a3_fw_force/arrays.npz` | `d130363f13ea511fdbad277e9d1ddcc7c4c9712f0661c898bada2b34e7aac7dd` | — | HF 归档 / pod |
| `auxiliary/.../umt5-xxl/special_tokens_map.json` | `7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d` | — | HF 辅助 |
| `auxiliary/.../umt5-xxl/spiece.model` | `e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458` | — | HF 辅助 |
| `auxiliary/.../umt5-xxl/tokenizer.json` | `6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b` | — | HF 辅助 |
| `auxiliary/.../umt5-xxl/tokenizer_config.json` | `ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05` | — | HF 辅助 |

### 1.3 capture 测试产物（3 个 `.npz`，本地存在但按 `.gitignore` 留外部）

`deploy/capture/test_evaluation_capture.py` 用 `uuid.uuid4().hex` 命名生成验收产物，
下列 3 个是历史 CPU 测试运行的**输出**（非测试输入，测试每次自行重建），故不进 GitHub，仅记录哈希：

| 文件 | sha256 | 字节 |
|---|---|---|
| `capture_cpu_acceptance_66249110a83c4317a81af18f60a32e34.npz` | `4953e85d058f95ef859ca76aea07119b22bf55e2faab77eef5fff126c01a0107` | 16,681 |
| `capture_cpu_acceptance_7b10764484214d4d8dfc017369e46bd3.npz` | `6bd41a56867fcb15fef085b8c24d00aa334977873381fc7a592b38cbe9094c9e` | 16,686 |
| `capture_cpu_acceptance_868a50cad9dc41379461fd08cb60de3c.npz` | `cefacf03f82a8dd8d52c8b8e8508dae674348e3afa44f376cd81a945f998ac99` | 16,684 |

---

## 2. pod 实际源码 ↔ `origin/main` 差异审计

对 pod 上三棵已在 `main` 的源码树做归一化逐文件哈希比对（清单存于本地 `_main_compare/`，
格式 `sha256  path`）。实算结果：

| 树 | pod 文件数 | main 文件数 | 共有 | 哈希不符 | pod 独有 | main 独有 | 结论 |
|---|---|---|---|---|---|---|---|
| `stage_40k/fw` | 85 | 84 | 84 | **0** | 1 | 0 | 代码逐字节一致 |
| `bridge` | 73 | 73 | 73 | **0** | 0 | 0 | 完全一致 |
| `stage_40k/dev` | 47 | 88 | 47 | **0** | 0 | 41 | pod ⊂ main，47 条全一致 |

- `fw` 的 1 条 pod 独有是 `./export_merged_v3.log`（合并导出**日志**，非代码），不构成源码差异。
- `dev` 的 41 条 main 独有是历史保留文件，pod 当前实例未携带；不影响"pod 现有代码均已在 main"的结论。
- 因此这三棵树**无需重复保存**，本分支不复制它们。

### 2.1 真正新增的 pod 独有代码（已收入 `restore_20261117/`）

`origin/main` 的 `deploy/` 只有 5 个引导文件（`README.md`、`bootstrap_pod.sh`、`pod.env.example`、
`retry_launch.sh`、`watch_pod.bat`），**不含**仿真与取证代码。下列为 main 上不存在（`git ls-tree` 命中 0）的新增内容：

- `deploy/a3_dual_arm_sim/`：125 个文件（src 87 / models 3 / configs 8 / assets 24），双臂 MuJoCo 仿真部署树。
  关键资源 `a3_cookie_transfer.xml` sha256 `65bb94eeded431d95af874e0256edd247a84b1367ec4bd81b7f722b421af46c6`。
- `deploy/capture/`：3 个取证脚本 `evaluation_capture.py`、`execution_chain_capture.py`、`test_evaluation_capture.py`
  （+ 3 个 `.npz` 测试产物，见 1.3，留外部）。
- `deploy/cpu_restore_acceptance.py`：CPU 恢复验收脚本（7,450 B）。
- 根级恢复/取证脚本：`gpu_shortest_action.py`、`search_archives.py`、`restore_auxiliary.py`、`test_normalizer.py`。
- 恢复台账与清单：`restoration_inventory.json`、`cpu_restore_acceptance.json`、`auxiliary_inventory.json`、
  `archive_search.json`、`transfer_manifest.json`、`identity_small_sha256.txt`、`pod_fw_sha256.txt`、
  `pod_deploy_rest_sha256.txt`、`export_merged4.log`、`GPU_SHORTEST_STATUS.md`、3 个 `gpu_shortest_action*.log`。
- 配置与派生证据：`step_040000/`（5 个 json/yaml）、`derived/`（2）、`archive_evidence/data/`（2 个 stats json）。

文件统计：`deploy` 132 + `step_040000` 5 + `derived` 2 + `archive_evidence` 2 + 根级 17 = **158**；
扣除 3 个 `.npz` → Git 纳入 **155**。

---

## 3. 敏感信息扫描（推送前复核，结果干净）

对 `restore_20261117/` 全树扫描凭据与敏感端点模式，**命中 0**：
HF/GitHub/AWS token 前缀（`hf_`、`ghp_`、`github_pat_`、`AKIA` 后接 16 位）、私钥块、`Bearer` 令牌、
`password=` 赋值、`netrc`、`/root/.ssh`，以及 pod 实例访问端点（JupyterLab 网关主机名与实例 URL）。

出现的外链仅为公开 benign 主机：`hf-mirror.com`、`github.com`、`www.apache.org`（许可证）、`wiki.ros.org`（参考）。
HF token 与 pod 访问端点均未出现在任何待提交文件中；本报告亦不写出端点主机名或实例 URL。
配置/台账里的 `/workspace/a3_data/...`、`/workspace/.cache/huggingface/...` 是容器内通用路径，
属必要 provenance（阶段 2 归一化器加载路径即依赖它），非凭据或访问端点，予以保留。

---

## 4. 独立分支保存与推送约束

- 分支 `rescue-40k-restore-20261117` 从 `origin/main`（`e39e62f`）切出，仅新增 `restore_20261117/` 与
  `.gitattributes` 一行（`/restore_20261117/** -text`，禁止行尾归一化以保字节身份）。
- 推送用显式新分支 refspec：`git push -u origin rescue-40k-restore-20261117`，**不**推 `main`、**不** force、**不**覆盖历史。
- 是否合入 `main`：待用户确认后再决定，本轮不自动合并。

---

## 5. 权重与二进制留外部（HF 归档）

按用户要求"大模型权重继续使用原始 HF 归档，GitHub 只保存代码、配置及适当的身份清单"：
`adapter.pt`、`step_040000_merged.pt`、`base_step_021700.pt`、两个 `arrays.npz`、4 个 HF 辅助分词器文件
均**只在 `identity_small_sha256.txt` 记录哈希**，字节不进 Git（`.gitignore` 已排除 `*.pt`/`*.npz`）。
基座存储说明（`restoration_inventory.json`）：base 为指向此前已全量 SHA 核验的 HF 缓存的 symlink，非独立字节副本；
`base_target = /workspace/.cache/huggingface/hub/models--hustvl--FasterWAM/snapshots/6bf9471ced6919a15ab8fded89f7772f5060c44b/libero/step_021700.pt`。

---

## 6. 诚实局限（必须在阶段 2 之前明确）

1. **14D 归一化器为派生重建，非原始字节。**
   `derived/provenance.json` 标 `status = DERIVED_NOT_ORIGINAL`，`original_stats_sha256 = null`。
   派生算法 = `stage_40k/build_v1_dataset.py` global（lines 116–149、191–192），
   派生文件 sha256 `354f59738d96e6962a0e1705635c5bf21c87c076c72bb512344c423093765b51`，宽度 [14,7]，
   `calibration_std_match = true`。但派生值**不能自证**与原始训练输入逐字节同一。
   管线实际加载路径由 `run_config.yaml` 钉死：`pretrained_norm_stats: /workspace/a3_data/a3_fw_v1g/dataset_stats.json`；
   力归一化 `model.force.norm_stats.file: /workspace/a3_data/a3_fw_force/force_calib_v1.json`。
   原始 `a3_fw_v1g/dataset_stats.json` 字节/哈希在 HF 归档中不可恢复 → `cpu_restore_acceptance.json`
   记 `original_stats_found = false`、`inference_identity_accepted = false`。派生统计已隔离，未被静默替换为"原始"。

2. **GPU 最短动作测试用的是合成零张量 fixture，不是真实观测。**
   `gpu_shortest_action.py`（sha `b42c48f0…`）输入为 `input_image=zeros(1,3,224,448)`、`proprio=zeros(1,14)`、
   `context=zeros(1,1,4096)`、`force_history=zeros(1,16,6)`、`force_history_mask=ones(1,16)`。
   run2 在 cuda/bf16 下输出 (32,7) float32、range [-0.89453125, 1.515625]、66.1 s、`GPU_SHORTEST_ACTION_OK`。
   这仅证明**模型可构建、可前向、形状/数值有限**，**不**证明真实观测下的动作正确性。
   阶段 2 必须改用真实 MuJoCo 观测重跑单块。

3. **力通道口径（阶段 2 取证不得混淆）。**
   模型 6 维力输入 = 原始 FORCE 通道 `[0,1,2,12,13,16]`
   （左腕三轴 0/1/2 + 左指垫 12/13 + 左夹爪 16）。
   四路真实指垫力为原始通道 `12,13,14,15`；其中 **14、15（右指垫）不属于当前模型六维输入**，右臂通道整体排除。
   `force_calib_v1.json`：channels `[0,1,2,12,13,16]`，
   names `[L_wrist_fx,L_wrist_fy,L_wrist_fz,L_pad_inner,L_pad_outer,L_gripper]`，单位 newton（raw，无重力补偿）。

4. **CPU 验收中 normalizer 项曾 FAIL。**
   `cpu_restore_acceptance.json` 的 `normalizer_cpu` 记 `ModuleNotFoundError: No module named 'fasterwam'`，
   后经 `test_normalizer.py` 的 `sys.path` 插入解决；`capture_tests` 12/12 通过（AST-only，原文件未改），
   `sim_resources` 通过（xml `65bb94ee…`、8 传感器、`nsensordata=16`、`nq=585`、mujoco 4.0.1，未跑回合）。
   `gpu_execution = false`：未做任何 GPU 模型构建/合并/prompt 编码/推理（最短动作测试除外，见局限 2）。

---

## 附：恢复传输通道（provenance，可复用）

pod → 本地逐字节传输经浏览器同源通道完成：在 HTTPS JupyterLab 页面内 `fetch` pod contents API
（`credentials:'include'`，root=/workspace，路径不带 `/workspace` 前缀），再把 base64 POST 到本地回环接收器
（`127.0.0.1:8799`，带 CORS/OPTIONS 处理）。整包 `rescue_new_assets.tar.gz`（9,599,633 B，
sha256 `c3d2be1b24f2ad4365b5f0e588886d5cde3b79bdcabae8c629c37ddae90a4516`）分 37 片传输后重组，哈希精确一致。
该通道零转写、逐字节，避免了手工抄写导致的哈希偏差（曾出现 `gpu_shortest_action.py` 手抄本 3206 B / `2368e070…`
与真本 3207 B / `b42c48f0…` 不符，已弃用手抄本，全部改走此通道）。