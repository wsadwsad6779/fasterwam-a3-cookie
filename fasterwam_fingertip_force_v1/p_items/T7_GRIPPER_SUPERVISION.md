# T7 夹爪修复 · 监督门槛（v1）

时间：2026-11-24T01:10Z
角色：任务7（worker）负责修复；本会话（supervisor）负责独立复核、门槛判定与纠偏。
本文件是监督合同，不替代 worker 的实现文档。

## 0. 边界（红线）
- 唯一基座官方 `step_021700.pt`（sha256 `01be4efc…`），只读。
- 最终交付为裸模型；G3/M5/oracle/外部夹爪控制器不得做最终决策；保留安全守卫。
- 不放宽 exact_fill；不扩大训练预算；不写 token 到任何文件/日志/Git；不删已有产物。
- supervisor 不生成/修改标签、不启动训练；只做只读取证、门槛判定、书面纠偏与叫停。

## 1. 事实基线（截至 2026-11-24T01:10Z）
1. **训练已停**：pod GPU 占用 28 MB；无 `t7_*` 进程；tmux `train_v3/chain_v3/watchdog_v3/finalizer_v3` 已关闭；V3 运行止于 step 200（`latest.pt` 保留，仅作诊断）。
2. **标签 V3 = 原始 `action[7]` 连续开度（1=开）**：`A3_845_7D_LABELS_V3.npz` sha256 `c4090dd2…`，845 集 / 458,332 行。没有阈值、没有 pm1、没有最短保持。
3. **三档开度的物理含义**（749 个受训集，`observation.state[7]` 实测 + 左指垫力 ch12/13）：
   | 指令档 | 实测夹口(中位) | 指垫力(中位) | >0.2 N 帧占比 | 含义 |
   |---|---|---|---|---|
   | 0.28（≤0.35） | 0.311 | **1.371 N** | 98.2% | 夹住饼干（夹持向厚 21.5 mm，2F-85 行程 85 mm，空载 0.28 档≈24 mm） |
   | 0.49（0.35–0.7） | 0.462 | **0.000 N** | 0.0013% | 已释放 |
   | 1.0（≥0.8） | 0.973 | 0.000 N | 0% | 张开/接近 |
   **结论：0.28 不是"全闭"，而是"夹住饼干的夹口"；把 0.28 叫成"闭合"正是 V2 崩溃的根源。**
4. **终态**：749/749 集存在物理释放（共 2,996 次，每集中位 4 次），且结尾停在释放位（末帧实测中位 0.49）。此前"结尾闭合 749/749"的结论由 0.8 阈值造成，已作废；旧文件保留，新增物理审计（`GRIPPER_V3_RELEASE_PHYSICS.json`）。
5. **翻转史**：旧标签 1,892 次翻转 / 318 个单帧脉冲；V2 用死区+最短保持把脉冲压到 0，但 48% 接近帧与全部 0.45 携带段/0.49 释放段被标为 +1 → 93.5% 闭合，策略"只闭不开"；V3 连续回归后 0 快速翻转 / 0 大跳变，749/749 有中途重开。
6. **部署漂移（高优先级，已开纠偏单 C-20261124-01）**：pod 与本地工作区的 `t7_data.py` 已被回退成 pm1 语义（`phase_open: g<=-0.8`），对 V3 连续标签会**静默关闭平衡采样**；`t7_train.py` 的 `close_frac` 也被改成 `>=0.8`（与 `open_frac` 同义、指标名不符）。HEAD（commit `e74e1ea`）是正确的 V3 版本：`g>=0.8` 与 `close_frac<=0.6`。证据：V3 日志每步 `close_frac≠open_frac`（0.648 vs 0.301），当前 pod 版本不可能产生该日志；00:45Z 采样探针（4.77%→20.84%，4.37×）证明当时跑的是 HEAD 版，回退发生在其后。**修复并复验前 G-D 保持 blocked。**

## 2. 验收门槛（worker 必须逐项交付证据）
- **G-A 标签语义**：连续、无阈值、无 pm1；V3 审计 pass；raw 与 label 逐帧一致。
- **G-B 映射回归**：评测 `g=2a−1`；`trans×2`、`rot×5`；`replan=10`；`exact_fill src==70`。修复已提交（commit `e74e1ea`），但回归脚本尚未跑 → pending。
- **G-C 物理**：回放 tracking RMSE<0.05；夹持段指垫力>0.5 N；释放段≈0；至少 3 集 × 3 档。（V3 三集回放已 pass：RMSE 0.0039、corr 0.9997。）
- **G-D 训练配比**：≥50% 批次锚定 open/rise 窗口；每 200 步 metrics 记 `close_frac/open_frac`；3k 时若 `close_frac>0.95` 或 `open_cmd_frac<0.05` 或 `reopen=0` → 判失败，停跑复查。**当前状态 blocked：部署漂移未修复（见事实 6 / C-20261124-01）。**
- **G-E 行为**：3k 里程碑至少 1 次成功复开且 exact_fill>0；6k 不得较 3k 显著退化；10 个固定 seeds。
- **G-F 账本**：`TRAINING_RUN_STATE` / `EVAL_SEED_MANIFEST` / `CHECKPOINT_MANIFEST` / `GPU_UPLOAD_MANIFEST` 同步；每条结论附 URL/branch/commit/sha256/统计口径。

## 3. 监督动作
- 每轮检查：pod 进程/GPU/日志步数；metrics 的 close/open 比例；产物 sha；"声明 vs 数据"一致性。
- 立即纠偏条件：只闭合指标回归；用阈值/二值标签替代连续标签；把 0.28 称作"全闭"；违规基座；绕过 G4/P9；评测参数回退（rot×1、replan 32、exact_fill src==0）。
- 记录方式：一般问题写 `PENDING_CORRECTIONS/`；严重问题写 `ESCALATIONS/`；不改 worker 代码。

## 3.5 判据修正（2026-11-24 用户指示，取代夹口阈值判据）
- 0.28/0.49/1.0 只描述**本数据集**的解剖学（饼干 21.5 mm 厚把指爪顶住），**不得**作为验收、评测或运行期判据；
  饼干数量/尺寸变化后夹口不再成立。
- 是否夹住由**力 + 任务结果**判定：夹持段左指垫力持续 >0.5 N、释放段回到 ~0、搬运无掉落、exact_fill。
- WAM 必须从力+视觉学夹持：**辅助力预测 / force dropout / 接触一致性**至少实现一项，并在每个里程碑报告
  with-force vs zero-force 消融；F2 不能是装饰。
- 泛化闸门 G-G：96 条 2x10 held-out；未来不同饼干数量/厚度数据必须作为新版本、经用户批准后加入。
- 禁止运行期"力超过阈值就张开"的外部夹爪控制器（外部控制器不得做最终决策）。
- 纠偏单：`FasterWAM_Supervisor/PENDING_CORRECTIONS/C-20261124-02-force-grounded-grasp.md`。

## 4. 待 worker 回答的开放问题
1. 0.28 指令 vs 0.311 实测的差值：评测解码是否需要补偿？（回放 RMSE 0.0039，初步看不需要，但评测必须用实测校验。）
2. 0.28 与 0.49 只差 0.21：策略输出抖动是否会在夹/放两档之间跳？3k 要看 grip 输出分布，不能只看 close_frac。
3. 每集只有 1 次 rise 锚点（2x5 类）：是否应把 0.28→0.49 的释放边也纳入锚定？作为 G-D 检查项，由 worker 决策并留证。

## 5. 证据索引
- `p_items/GRIPPER_V3_ACCEPTANCE.json` sha256 `1a29c6d4…`
- `p_items/GRIPPER_V3_LABEL_AUDIT.json` sha256 `2147901b…`
- `p_items/GRIPPER_V3_RELEASE_PHYSICS.json` sha256 `00b208b8…`
- `p_items/GRIPPER_V3_FORCE_BY_LEVEL.json` sha256 `25ed29df…`
- `p_items/GRIPPER_V3_TERMINAL_STATE_AUDIT.json` sha256 `5df52e0c…`（旧口径，作废但保留）
- `p_items/A3_845_7D_LABELS_V3.npz` sha256 `c4090dd2…`
- 脚本：`p_items/scripts/t7_v3_release_physics.py` `81451a2e…`；`t7_v3_force_by_level.py` `4425992a…`