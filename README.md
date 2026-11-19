# FasterWAM → A3 饼干搬运（main 分支总览，2026-11-19）

> 本文件是 `main` 分支的权威入口。历史交接材料保留在 `docs/` 与归档分支里；
> 当前代码、训练/评估链、证据与恢复说明以本文件和第 4 节的架构文档为准。
> 实例 `rgapi-one-click-24316-57b6b870` 随时可能销毁；**代码在本仓库，权重/大语料在 Hugging Face**，
> 两者合起来是恢复现场的全部内容。

---

## 1. 研究目标

用视频-动作模型 FasterWAM（正面 + 左腕双相机图像、14 维本体感知、18 路力、文本）在 A3 双臂 MuJoCo
仿真里完成「把源区饼干搬进目标盒」：自主接近 → 张开夹爪 → 接触饼干闭合 → 稳定搬运 → 下降 →
主动释放 → 再接近下一块，最终达到 **10 块 exact_2x5_fill**。

判定口径（钉死，不随报告变化）：

- 任务完成 = 松手后盒内留在目标槽位的块数与 `cookies_in_target` 一致，且 `exact_2x5_fill=true`；
  不用「末帧末端 xy 落在盒内」代替。
- 任何闭环评估都要记录：front+left_wrist 双视角 MP4、14 维 A3 动作、实测 opening、18 路力、
  抓取事件、checkpoint/seed/步数、有无外部门控；视频帧数必须与执行步数一致。

## 2. 系统身份 A / B / C

| 代号 | 含义 | 是否可作为修复证据 |
|---|---|---|
| **A** | 原始 40k 合并权重，裸模型 | 对照组（已知失败：高力 166.6/164.9 N 触发中止） |
| **B** | 诊断门控版（G2/G3/M5/oracle 等特权工具） | **否**，只用于定位机制 |
| **C** | 独立 checkpoint，只吃可部署输入（图像/14 维 proprio/6 路力/文本），无门控 | **是**，唯一验收对象 |

C 就是裸模型：无 G3、无 M5、无 oracle、无外部开合门控；只保留不替模型决策的被动安全守卫
（腕部 |F|>150 N、指垫>500 N 时中止并记录，不改动作）。

## 3. 当前状态（2026-11-19）

- **力缺陷已修复（C 类）**：v3/v4 里程碑 600 步全部无中止，峰值 12.6–14.4 N（A 为 166.6/164.9 N 中止）。
- **任务未修复**：C 从第一帧就把夹爪合上（A3 第 6 列 -1 占 95.8–100%，opening<0.005 占 91.7–97.8%），
  EEF 能到饼干盘上方（xy 距 batch_center 最小 0.6–1.4 mm）但抓取 0 次，`cookies_final=0`。
- v3：12000 步（多布局语料），val_action 最好 0.0027621@8000。
- v4：再 +4000 步（夹爪窗口过采样 ×8，517/6248 窗口），val_action 最好 0.00180@14000，
  但里程碑 13000/14000/15000/16000 的 C 类评估仍是 grasps 0/0–0/1，`DEV_PROGRESS no`，
  冻结的最终 seed 13/16 未动用。
- **定位**：第七维坍缩为“恒闭合”。教师标签 97.5% 为闭合（+1），而动作 loss 对第七维只是普通 MSE，
  “恒闭合”是低损失解。进一步怀疑标签生成 `grip_labels()` 把“保持张开”（物理张开但静止的帧）
  继承成了 +1 闭合：该函数初始 `prev=1.0`，只有在出现主动张开动作（opening 上升 >0.02）时才翻到 -1。
- **下一步（未完成）**：对完整成功专家轨迹做夹爪阶段语义核查（接近前/需保持张开/首次夹住前后/
  稳定携带/下降/主动释放/再接近），逐帧对照 expert joint_cmd、实测 opening、指垫接触、旧标签、
  40k/12k raw dim6、最终 A3 命令，确认“保持张开→+1”错误后再修标签并重训。

## 4. 本分支架构（先读这个）

- [docs/ARCHITECTURE_20261119.md](docs/ARCHITECTURE_20261119.md)：代码分层、数据契约、动作链路
  （7D→pm1→A3 14D 的符号约定）、训练器与评估器、checkpoint 格式、证据索引。
- [docs/RESTORE_FROM_ARCHIVE_20261119.md](docs/RESTORE_FROM_ARCHIVE_20261119.md)：实例销毁后
  仅凭 GitHub + Hugging Face 恢复训练/评估现场的完整步骤。

## 5. 仓库布局

```
bridge/             A3 动作适配（FasterWAMA3ActionAdapterV3：LIBERO 7D → A3 14D）
deploy/             pod 引导与守卫脚本（pod.env 不入库）
finetune/           早期 LoRA/数据集工具（历史路径，2026-10 的 r1 线）
stage_40k/          40k 训练链：trainer、驱动、数据构建、force 校准、cpu/gpu 门控
restore_20261117/
  deploy/           2026-11-17 恢复的代码树快照（与 stage_40k 同源）
  step_040000/      40k 轻量节点（adapter/optimizer/scheduler/rng/trainer_state/manifest）
  archive_evidence/ 原始 100 条专家语料（arrays + joint_cmd，7.1 MB）
  reconstructed_normalizer/
                    2026-11-18/19 的全部修复线脚本、证据 JSON、评估录像与状态文件（核心）
data/ results/      早期数据集清单与评估产物
local_1113/         2026-11-01 归档轮的备份报告与清理台账
tools/ docs/        归档工具与历史文档
```

`restore_20261117/reconstructed_normalizer/` 是当前主线，主要文件：

| 组 | 文件 |
|---|---|
| 语料 | `collect_expert_diverse_20261119.py`、`rebuild_teacher_arrays_20261119.py`、`accept_diverse_corpus_20261119.py`、`audit_teacher_*` |
| 训练 | `train_action_expert_segmented_20261119.py`(v3)、`train_action_expert_v4_20261119.py`(v4)、`drive_diverse_20261119.sh`、`drive_v4_20261119.sh` |
| 评估 | `c_class_eval_20261119.py`（C 类裸模型 + 双视角 MP4）、`rebuild_step8000_summary.py`、`beh_step8000.py` |
| 证据 | `MODEL_REPAIR_STATE.json`、`DATA_ACCEPTANCE.json`、`RECONSTRUCTED_NORMALIZER_ACCEPTANCE.json`、`ROOTCAUSE_EVIDENCE_LEDGER_20261118.json`、`SEED17_*.json`、`GRIPPER_*.json` |
| 评估产物 | `c_class_eval_v3_step*/`、`c_class_eval_v4_step*/`（summary.json / paired_A_vs_C.json / episode_*.npz / episode_*.mp4） |

## 6. 外部归档位置

- **GitHub**：`wsadwsad6779/fasterwam-a3-cookie` 的 `main`（代码、文档、小体积证据）与本分支。
- **Hugging Face（私有）**：`wsadwsad6779/fasterwam-a3-force-40k`
  - `v1_force_40k/`：40k 训练节点 `nodes/step_005000 … step_040000`、code 快照、force 校准；
  - `v1_force_rehearsal/`：恢复演练节点；
  - `v2_repair_20261119/`：本次修复线归档——
    `evidence/`（rn 文本证据包、archive_evidence、model_repair_runs 小文件）、
    `corpus/teacher_diverse_20261119/`（arrays/poses/force/manifest + frames.dat）、
    `checkpoints/action_expert_v3|v4/`（`last.pt` 全状态 ≈3.38 GB；`delta_step_*.pt` 纯权重 ≈1.13 GB）。
- 40k 合并权重可用「公开基座 `hustvl/FasterWAM:libero/step_021700.pt` + `v1_force_40k/nodes/step_040000/adapter.pt`」
  重新合并，因此 `step_040000_merged.pt`（11.1 GB）不单独上传。

## 7. 关键契约（冻结）

- 动作：模型输出 7 维 LIBERO 约定（dim6：+1=闭、-1=开）→ `policy40k.chunk()` 以 0 为界二值化
  （≥0→+1 闭）→ `FasterWAMA3ActionAdapterV3.convert()` 输出 14 维 A3 动作，**A3 第 6 列 −1=闭、+1=开**
  （`output[6] = -np.sign(action[6])`）。评估 npz 存的是 14 维 A3 动作；教师语料 `arrays.npz['action']`
  存的是 7 维标签。
- 夹爪标签（现状，待审）：`grip_labels(opening)` 由 opening 差分生成：d<−0.02 记闭(+1)，
  d>+0.02 记开(−1)，死区内保持前值，初始前值=+1，最小段长 3。
- 训练：action_expert **局部全参数微调**（602 张量 / 563,140,103 参数），**不是 LoRA**；
  checkpoint = `last.pt`（权重+优化器+RNG+step+data sha）+ 里程碑 `delta_step_XXXXXX.pt`（纯权重）。
- 数据划分：`teacher_diverse_20261119` 11 条训练布局 + 4 条**离网（off-grid）**验证布局，
  验证集永不参与训练；评估开发 seed 29/30，最终冻结 seed 13/16 只在开发集出现进展时才动用一次。

## 8. 恢复入口（最短路径）

```bash
git clone https://github.com/wsadwsad6779/fasterwam-a3-cookie.git
cd fasterwam-a3-cookie && git checkout main
# 权重与语料：按 docs/RESTORE_FROM_ARCHIVE_20261119.md 从 HF 拉取
```

## 9. 许可与引用

- 上游：`hustvl/FasterWAM` @ `83667817df0d4f823f39d90700e61ea2f432ac45`（Apache-2.0）；
  `Eter0109/a3_dual_arm_sim` @ `08ebe8cb370856051d0b32b07e691cc9b056716b`。
- 原始专家示范：HF dataset `Eter0109/a3-front-close-left-100` @ `8fdd8a7c…`，42,681 帧、20 Hz。
- 本仓库不放任何 HF token；运行期从 `/root/.cache/hf_token` 读取。
