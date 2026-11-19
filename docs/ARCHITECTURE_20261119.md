# 架构说明（2026-11-19，main 分支）

本文描述本分支的代码分层、数据契约、动作链路、训练/评估链与证据索引。
只写“当前实际运行的东西”，历史实验保留在归档分支。

---

## 1. 代码分层

```
┌──────────────────────────────────────────────────────────────────┐
│  上游（pinned）                                                   │
│  hustvl/FasterWAM @ 8366781       Wan2.2-TI2V-5B + SparseActionDiT│
│  Eter0109/a3_dual_arm_sim @ 08ebe8c   A3 双臂 MuJoCo 场景          │
└───────────────▲──────────────────────────▲───────────────────────┘
                │ vendor                    │ 观测/动作
┌───────────────┴───────────┐   ┌──────────┴───────────────────────┐
│ stage_40k/                │   │ bridge/                          │
│  fw/ 训练/推理代码        │   │  action_adapter_v3.py            │
│  dev/eval40k/ 策略封装    │   │   LIBERO 7D → A3 14D             │
│  force 校准/门控          │   │  a3_wrap.py 记录器/守卫          │
└───────────────▲───────────┘   └──────────▲───────────────────────┘
                │                          │
┌───────────────┴──────────────────────────┴───────────────────────┐
│ restore_20261117/reconstructed_normalizer/   （当前主线）         │
│  collect_*   专家语料采集（确定性回放 + 布局平移）                │
│  train_*     action_expert 局部全参数微调（分段驱动）             │
│  c_class_eval_*  C 类裸模型闭环评估（双视角录像）                 │
│  audit_*/analyze_*  机制审计与证据生成                            │
└──────────────────────────────────────────────────────────────────┘
```

## 2. 动作链路（唯一合法路径）

```
模型输出 action ∈ R^{B×T×7}      LIBERO 约定：dim6 +1=闭, −1=开
        │  policy40k.Policy40k.chunk()
        │    pm1 模式: action[...,-1] = where(action[...,-1] >= 0, +1, −1)
        ▼
7 维 pm1 动作（binerized）
        │  bridge.action_adapter_v3.FasterWAMA3ActionAdapterV3.convert(a, rot)
        │    output[6] = −np.sign(action[6])        ← 符号翻转，A3 约定 −1=闭, +1=开
        ▼
14 维 A3 环境动作 → env.step()
```

符号约定必须两处同时记住：

| 位置 | +1 含义 | −1 含义 |
|---|---|---|
| 模型/教师标签 7 维 dim6 | 闭 | 开 |
| A3 环境 14 维 col6 | 开 | 闭 |

评估 npz 的 `action` 是 14 维 A3 动作；教师语料 `arrays.npz['action']` 是 7 维标签。
“模型 raw dim6” 的符号 = −(A3 col6)。

## 3. 数据契约

### 3.1 原始 100 条专家语料（archive_evidence）

`restore_20261117/archive_evidence/data/a3_fw/arrays.npz`

| 键 | 形状 | 含义 |
|---|---|---|
| `action` | 42681×7 | 7 维标签；dim6 +1 占 66.2% |
| `opening` | 42681 | 实测夹爪开度 |
| `joint_cmd` | 42681 | 专家夹爪关节目标（真实单位） |
| `ep_start`/`ep_len` | 100 | 回合切分 |

### 3.2 多布局语料 teacher_diverse_20261119

- 15 条全部成功（`cookies_in_target=10`、`exact_2x5_fill=true`、600 帧），11 条训练布局 + 4 条离网验证布局；
- `frames.dat`：9000×301056 B = 2,709,504,000 B（front|left_wrist 224×448 → uint8 帧）；
- `arrays.npz`：`action`(9000×7)、`proprio`(14)、`ep_start/ep_len`；
- `poses.npz`：`poses_ep00..14`（EEF 位姿）、`opening_ep00..14`；
- `force_arrays.npz`：`force18`、`force6`（校准通道 [0,1,2,12,13,16]）；
- `manifest.json`：每回合 dx/dy 偏移、split、seed、成功标志。

采集轴：80 块饼干网格整体 xy 平移（`env_factory.apply_batch_offset`），安全范围 x∈[−10,10] mm、y∈[−8,8] mm；
锚点 (0,0) 必须与旧语料逐位一致（force bit-equal、action/opening 最大差 0）。

### 3.3 夹爪标签（当前实现，待审计）

`collect_expert_diverse_20261119.py::grip_labels(opening, eps=0.02, min_hold=3)`：

```
prev = +1 (闭)
for t: d=opening[t+1]-opening[t]
       d < −eps → prev=+1 (闭)
       d > +eps → prev=−1 (开)
       else     → 保持 prev
```

**已知风险**：物理张开但静止的帧（初始接近、保持张开）不会从 +1 翻到 −1，
因此“保持张开”被继承为“闭合”。这是第七维坍缩的首要嫌疑，审计脚本计划见 README §3。

## 4. 训练链

- 入口：`train_action_expert_segmented_20261119.py`（v3）与 `train_action_expert_v4_20261119.py`（v4）。
- 结构：action_expert **局部全参数微调**，602 张量 / 563,140,103 参数；视频/力/动作三头联合 loss，
  动作 loss 为 7 维逐元素 MSE（dim6 与其它维同权）。
- 分段驱动：`drive_diverse_20261119.sh`（500→12000）、`drive_v4_20261119.sh`（12000→16000，
  gripper 窗口 ×8 过采样）。
- checkpoint：
  - `last.pt` ≈3.38 GB：权重 + 优化器 + RNG + step + data sha（可精确续训）；
  - `delta_step_XXXXXX.pt` ≈1.13 GB：纯权重里程碑；
  - `segments.jsonl`：每段 val 指标、wall time、data sha。
- 早停：连续两次 held-out-layout `val_action` 比最优退步 >25%（force 项只报告不早停）。

## 5. 评估链（C 类验收）

`c_class_eval_20261119.py`：

- A 臂：原始 40k 合并权重；C 臂：独立 checkpoint（`C_CKPT=.../delta_step_XXXXXX.pt`）；
- 固定 seed 29/30（开发）；无任何门控；600 步上限、每 10 步 replan；
- 录制 front|left_wrist 224×224 拼接帧，20 fps mp4v，视频帧数 = 观测帧数 = 执行步数+1；
- 输出：`summary.json`、`paired_A_vs_C.json`、`episode_{A,C}_seed*.{json,npz,mp4}`；
  npz 内含 measured(6 路力)、eef、force18、opening、14 维动作、cookies/source/exact/hold、相机无关的逐步事件。
- 指标：cookies_final/source_final、exact_fill、force_absmax、jam_steps、gripper_flips、
  gripper_close_events、grasps_confirmed、eef_path_mm、opening_min/max。

## 6. 证据索引（reconstructed_normalizer）

| 主题 | 文件 |
|---|---|
| 状态总表 | `MODEL_REPAIR_STATE.json` |
| 语料验收 | `DATA_ACCEPTANCE.json`、`accept_diverse*`、`audit_teacher_*` |
| 归一化重建 | `RECONSTRUCTED_NORMALIZER_ACCEPTANCE.json`、`RECONSTRUCTION_EVIDENCE.json` |
| 根因台账 | `ROOTCAUSE_EVIDENCE_LEDGER_20261118.json`、`ROOT_CAUSE_EVIDENCE_MATRIX_20261118.md` |
| 夹爪机制 | `GRIPPER_STRUCTURE.json`、`GRIPPER_MATCHED_PHASE.json`、`GRIPPER_SUPERVISION_VS_GENERATION.json`、`GRIPPER_LABEL_CANDIDATE.json` |
| seed17 机制 | `SEED17_*_CPU_20261118.json`（12 份） |
| v3 评估 | `c_class_eval_v3_step{1000,2000,4000,8000,12000}/` |
| v4 评估 | `c_class_eval_v4_step{13000,14000,15000,16000}/`（含双视角 mp4） |

## 7. 已知失败链（摘要）

1. 力回路：A 臂 166.6/164.9 N 中止 → v3/v4 已修复（12.6–14.4 N，无中止）。
2. 第七维坍缩：恒闭合；教师标签多数类 + 标签语义缺陷（保持张开→+1）。
3. 前六维：v3 后已能把 EEF 送到饼干盘（0.6–1.4 mm xy 误差）但未形成抓取；
   搬运/放置/多块 exact_fill 未达成。
4. 冻结 seed 13/16 从未触碰。
