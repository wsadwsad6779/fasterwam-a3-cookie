# v3 修复流水线合规审计（对照 2026-11-18T10:23:42Z 指令）

NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL
TRAINING_DATA_IDENTITY=CANDIDATE_RECONSTRUCTED_TRAINING_DATA
original_stats_sha256=UNKNOWN

## 审计范围

- 指令第五节（有界微调 8 条约束）、问题A–E 的监督要求、第六节（冻结验收）、第七节（连续推进 + 禁止重复启动）。
- 审计对象：`drive_diverse_20261119.sh`、`train_action_expert_segmented_20261119.py`、
  `c_class_eval_20261119.py`、`collect_expert_diverse_20261119.py`、
  `accept_diverse_corpus_20261119.py`、`MODEL_REPAIR_STATE.json`。
- 本审计为只读核验，未改任何流水线代码，未启动训练。

## 第五节 8 条约束逐条核验

| # | 约束 | 判定 | 证据 |
|---|------|------|------|
| 1 | 原始 40k checkpoint 保持不变 | 满足 | trainer 只 `load_checkpoint(base)` 只读加载；保存物只有 delta（`trainable` 白名单内张量）；`c_class_eval.load_ckpt` 抽样非 action_expert 张量，要求 `max_abs_diff==0`（`delta_confined_to_whitelist`）才继续评估 |
| 2 | 冻结清单/标签规则/验证种子/配置/随机种子/停止条件 | 基本满足 | TRAIN_SEED=20261118、LR=1e-4、TOTAL_TARGET=12000、SEG_STEPS、里程碑集合均为 trainer 常量；dev 种子 29/30、final 13/16 为 drive/c_class_eval 常量；manifest 由 collector `manifest.json` 冻结 + accept 门校验。缺口见 D3（进程级） |
| 3 | 复用 40k 权重与训练程序 | 满足 | base 为 `step_040000_merged.pt`；数据集走原 A3VideoDataset/RobotVideoDataset 程序；`run_config.yaml` 复用（仅 force norm_stats 路径重指、text encoder 加载开启） |
| 4 | 最小规模、步数上限、资源限制、验证间隔 | 满足 | 12k 步封顶；每段 55 分钟时间帽；WORK≥12GB / PUB≥3GB 磁盘守卫；每段末验证（段长 500 步，即 ≤500 步间隔）；里程碑 100/500/1000/2000/4000/8000/12000 留存权重 |
| 5 | 分组干预保留归因 | 部分满足 | 见下文"归因矩阵"。当前分组只粗到两层：v2=标签配方+FT 联合、v3=布局+步数联合；没有单因子隔离运行 |
| 6 | 独立权重/日志/数据身份/可复现配置 | 满足 | `data_sha256`（arrays.npz 哈希）进 checkpoint 与 segments.jsonl；TRAINING_DATA_IDENTITY 标记在全部产物；WORK/PUB 目录分离（overlay 磁盘存训练物，/workspace 只存选定 delta）；checkpoint 含 delta+optimizer+RNG+step 可精确续段 |
| 7 | 数据不可信/目标错误/资源异常时停止 | 满足 | DATA_GUARD（ep_start 累计性、proprio 零行率、EXPERT_ACTIONS、train/val 存在且有序）；NONFINITE_LOSS→SystemExit；DISK_GUARD→SystemExit；val_action 连续 2 段退化→EARLY_STOP；时间帽兜底 |
| 8 | 不做未经核验的长时间重训 | 满足 | 分段落执行、12k 封顶、无超目标自动续跑逻辑；v1 历史教训（step_000200.pt 因磁盘满不可载、step_000100.pt 验证可载）已写入 predeclared.json |

## 偏差与风险

### D1：指令措辞"LoRA微调" vs 实际全量局部微调

指令原文写"LoRA微调"，实际 trainer 对 602 个 `action_expert` 张量做全量局部微调
（563.1M 参数，非 LoRA 因子）。`c_class_eval` 的 CHECKPOINT_IDENTITY 已显式标注
`ACTION_EXPERT_LOCAL_FULL_FINETUNE (not LoRA)`，未隐瞒。

- 支持理由：故障位点在 action_expert 权重本身，全量更新该模块可改变其内部表征；LoRA 只加旁路，旁路参数不属于原推理路径的一部分。
- 代价：563.1M 参数 × 12k 步的训练时间与显存显著高于 LoRA。
- 判定：措辞偏差、已文档化、属指令七允许的迭代实验空间；但"只动 action_expert"这一边界不可再放宽。

### D2：问题A 监督配方 vs 指令细项

指令问题A 要求：joint_cmd 的连续值必须先确认控制语义、方向、单位和时间对齐，不能直接取符号代替标签。
实际配方 `grip_labels(opening, eps=0.02, min_hold=3)`：只用实测 opening（死区 + 3 遍游程合并），未用 joint_cmd，也未用接触状态。

核对 `COLLECTOR_COMMAND_SEMANTICS_20261118.json`：
- NPZ 中 joint_cmd 仅为夹爪标量命令（ac[:,7]，夹爪通道），无独立连续关节语义可做监督；
- 基于 joint_cmd 的监督会退化为"命令空间自回归标签"，与已确认的"训练标签=实测 EEF 位移"惯例相悖；
- 因此弃用 joint_cmd 是有证据支持的决定，不是跳过核验。

未闭合缺口：接触状态（NPZ 有 18 维力通道，pad 12:16、腕 0:3）未并入夹爪标签
（例如"接触后禁开夹"规则）。留作 v4 候选，需记录而非默许。

### D3：drive 脚本无锁、无恢复

`drive_diverse_20261119.sh` 没有进程锁和恢复逻辑：重复启动会
- append `segments.jsonl`（段记录错乱）、
- 二次写 `last.pt`、`delta_step_*.pt`、
- 与在跑训练进程竞争共享 WORK 目录。

这违反指令七"禁止重复启动"的脚本级内建要求。缓解措施（操作规程）：
- 启动前必须 `pgrep` 确认无训练进程、检查 `/root/repair_runs/action_expert_v3` 与 segments.jsonl 状态；
- 本审计记录该风险，任何重新启动必须先核对该状态；
- 若主会话（b462d38d）正在操作 pod，本会话不启动训练（共享 WORK 目录冲突）。

## 归因矩阵（v1→v2→v3）

| 干预 | v1 (lora_300) | v2 (action_expert_v2) | v3 (action_expert_v3) |
|------|--------------|----------------------|----------------------|
| 标签配方 | 历史标签 | 去抖 grip_labels（同 v3） | 同 v2 |
| 语料 | teacher_replay（单轨迹复制 N） | teacher_replay | 多布局 13 train + 4 off-grid val |
| 步数 | 100（磁盘满中断） | 早停 1000 | ≤12000（里程碑评估） |
| 结构 | LoRA 300 步试验 | action_expert 全量局部 | 同 v2 |

- v2 vs A = 标签配方 + FT 联合效应；
- v3 vs v2 = 布局多样性 + 步数预算联合效应；
- 无单因子隔离运行。若 v3 在 dev 种子显著优于 A，收益不能唯一归因于布局或标签；
  需要时可在 v4 以 v3 权重为基线做单一干预隔离（成本另计，需重新走冻结流程）。

## 第六节验收协议映射（概要）

- dev 种子 29/30（冻结），final 种子 13/16 只在 dev 有进展时触碰（drive 脚本 HAS_PROGRESS 门）；
- EVAL_GATES=NONE：裸策略，无 G3 门、无 M5 控制器、无 cookie 世界坐标、无 oracle 输入；
  被动安全中止（腕>150N、pad>500N）双臂一致，不是放行门；
- 指标集：cookies_final/max、source_final/min、exact_fill 帧、10 槽 occupancy、
  force envelope（per-channel max、>14.3N 帧、>50N 帧）、eef_path_mm、jam_steps、
  gripper_flips、hold、infer 次数；
- 缺口：散落数未直接记录（可由 target+source 计数间接推算，需先确认 cookies_in_source 语义）；
  无显式"渐进 10 块阶梯"文档——由里程碑评估（1000/2000/4000/8000/12000）+ final 种子充当阶梯。

详细冻结协议见 `C_SYSTEM_ACCEPTANCE_PROTOCOL_20261119.md`。

## 审计结论

- 第五节 8 约束：7 条满足，1 条部分（约束 5 归因粒度）。
- 3 处偏差/风险（D1/D2/D3）均已文档化，无未声明的静默偏离。
- 指令七"禁止重复启动"：脚本级未内建（D3），以操作规程补足。
- 流水线待办：accept_diverse_corpus_20261119.py 尚未在 pod 执行；trainer 尚未启动；
  启动前必须 pgrep 确认无在跑训练进程，且确认主会话未在操作 pod。
