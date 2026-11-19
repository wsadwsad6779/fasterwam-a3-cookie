# C 系统冻结验收协议（第六节落实）

NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL
TRAINING_DATA_IDENTITY=CANDIDATE_RECONSTRUCTED_TRAINING_DATA
original_stats_sha256=UNKNOWN

## 定位

只有 C 系统（独立修复 checkpoint + 部署输入 + 裸策略）达成才算自主成功：
- A = 原始 40k + 原始协议（基线，仅对比用）；
- B = 原始 40k + G3/M5/P2 诊断干预（诊断用，不计入成功）；
- C = 修复后 checkpoint + 原部署输入 + 无外部门控。

本协议把指令第六节的基础验收与进阶验收映射到
`c_class_eval_20261119.py`（评估器）与 `drive_diverse_20261119.sh`（里程碑驱动）的
具体判定函数与产物，冻结以下各项。

## 冻结项清单

| 冻结项 | 值 | 冻结位置 |
|--------|-----|----------|
| dev 种子 | 29, 30 | c_class_eval 默认 C_SEEDS |
| final 种子 | 13, 16（仅 HAS_PROGRESS 时触碰） | drive 脚本 FINAL_SEEDS + HAS_PROGRESS 门 |
| 场景 | position_noise=0, yaw_noise=0, randomize_cookies=True（零噪声→确定性场景）, render_cameras=True | c_class_eval run_arm |
| 部署输入集 | images + 14 维 proprio + 6 通道冻结力流 + text | predeclared.json inputs |
| 禁止输入 | G3 门、M5 控制器、cookie 世界坐标、oracle 决策、仿真真值进策略路径 | predeclared.json forbidden_inputs |
| 循环 | rollout40k.run_episode 逐字拷贝 + 被动力中止（腕>150N 或 pad>500N，双臂一致，被动不决策） | c_class_eval rollout |
| 重规划/步数 | REPLAN=10, MAX_STEPS=600（与 A 臂同参数） | C_REPLAN / C_MAX_STEPS |
| checkpoint 规则 | 加载必须证明 delta 限于 action_expert 白名单（抽样冻结张量 bit_equal，否则 SystemExit） | c_class_eval load_ckpt |
| 指标集 | 见下表 | episode_metrics / paired |
| 身份标记 | NORMALIZER/TRAINING_DATA/CHECKPOINT_IDENTITY/EVAL_GATES 进全部产物 | MARKERS 展开 |

## 基础验收（第六节基础项）

判定条件（C 臂在 dev 种子 29/30 上逐项通过）：

1. 模型自己的合理夹爪动作：`gripper_flips` 显著低于 A 的 38/96，且 opening 在 [0,1] 内；
2. 真实观测下自主抓取+搬运：`cookies_max > 0`（出现过被抓起的饼干）且 `jam_steps` 不恶化；
3. 自主下降+释放：`cookies_final > 0` 且（`exact_fill_any` 或 `hold_final > 0`）；
4. ≥1 块稳定入框：`cookies_final >= 1` 且 `success_frames > 0`。

通过标准 = 第 4 条成立（对应指令"≥1 cookie 稳定在框内由模型控制完成"），
第 1–3 条为支撑性证据，任一严重退化（如夹爪翻转次数超过 A）应记为警示。

## 进阶验收（第六节进阶项）

固定种子，逐种子记录以下统计（episode_metrics 字段）：

- 稳定入框数：`cookies_final`、`hold_final`（success_hold_count）
- 散落数：缺口 G1（见下）
- 源区剩余：`source_final`、`source_min`
- 槽位占用：`occ`（10 槽逐槽）、`touches_all_walls_final`
- exact_fill：`exact_fill_any`、`exact_fill_frames`
- 接触力：`force_envelope`（absmax、per_channel_max、>14.3N 帧数、>50N 帧数）
- 辅助：`eef_path_mm`、`eef_step_mm_max`、`jam_steps`、`gripper_flips`、`infer_calls`

A vs C 配对：`paired_A_vs_C.json`（delta_cookies、delta_force_absmax、
C_beats_A_on_any_seed、C_exact_fill_any_seed、C_success_any_seed）。

进阶判定：
- HAS_PROGRESS = C 在 dev 种子 29/30 上与 A 配对比较，任一 C 指标严格优于 A 且无安全中止；
- final 种子 13/16 只在 HAS_PROGRESS 后执行，其结果为最终进阶验收证据；
- 无放水：不得以其他种子替代、不得放宽指标、不得把 B 系统运行计入 C。

## 渐进 10 块阶梯

- 里程碑评估点：1000/2000/4000/8000/12000 步 delta（drive 脚本在里程碑触发 c_class_eval，
  C_CKPT=delta_step_%06d.pt），形成步数维度的阶梯证据；
- final 种子 13/16 的一次性测试为阶梯终点；
- 缺口 G2：没有独立文档把"1 块 → 多块 → 10 块"写成显式阶梯；验收报告须按
  cookies_final 分档（0 / 1–4 / 5–9 / 10+exact_fill）呈现，不得只报总量。

## 防混线规则

- 任何含 G3 门/M5 控制器/oracle 输入的运行 → B 系统证据，禁止写入 C 验收目录；
- 被动力中止的 episode：记录并标注 `safety_abort`，不作为成功；
- 种子集合冻结（dev 29/30、final 13/16）；增删种子须重新冻结并文档化；
- 里程碑评估使用独立 OUT 目录，不得覆盖 dev 种子评估目录。

## 缺口记录

- G1 散落数：c_class_eval 未直接记录散落数。可由 10 − cookies_in_target − cookies_in_source
  推算，前提是 `cookies_in_source` 语义为"批次 10 块中仍在源区的数量"；该语义尚未在
  代码中确认，验收前必须核对 env info 定义，或补一帧直接计数（改 env 记录需重新走冻结流程）。
- G2 显式阶梯文档：见上。以里程碑评估序列充当，报告按分档呈现。

## 验收产物清单

- dev 评估：`c_class_eval_20261119/{predeclared, A_summary, C_summary, paired_A_vs_C, summary}.json`
  + `episode_{A,C}_seed{29,30}.{npz,json}`；
- 里程碑评估：drive 脚本指定独立目录，同构文件；
- 全部产物带 MARKERS；`checkpoint_sha256.txt` 记录所评 delta 的哈希。
