# T7 GOAL V3（2026-11-24，用户批准）

## 目标（唯一主 Goal）
在原官方 `hustvl/FasterWAM` `step_021700.pt` 上，完成 **V3 指尖力 F2 主模型**的从零 21,000 步有界训练、每 3,000 步 10-seed 裸模型双视频评估，以及最终独立裸模型验收；训练/评估/模型/视频/可恢复状态持续保存。

## 技术定义（不可漂移）
- 基座：官方 `step_021700.pt`（sha256 `01be4efc…`，proprio 8 / action 7 / 无 force conditioner），只读；F2 结构不变。
- 标签：**V3 连续开度** `A3_845_7D_LABELS_V3.npz`（sha `c4090dd2…`；1=张开，0.28=夹住（左指垫 1.37 N），0.49=释放（0 N））。禁止 pm1/阈值/平滑标签。
- 力：F2 Force-gated FiLM + C-02 force-grounded 辅助头（264,705 参数）+ 10% force dropout；禁止运行期“力超阈值就张开”的外部控制器。
- 数据：V2 split（589/85/171）+ V3 标签；96 条 2x10 作为独立 held-out；P3-B `p3b_medadv_probe_v1` 独立，不混入。
- 评估：冻结场景 `EVAL_SCENE_MANIFEST_V3`（80 个不同 basic 布局，freeze `99194298…`，fail-closed）；eval `g=2a−1`、`trans×2`、`rot×5`、`replan=10`、`exact_fill src==70`；10 个固定 seed；20 步 hold；安全守卫保留。

## 实验范围（用户 2026-11-24T05:50Z 决定）
- **当前只跑 F2**，fresh 21000（2026-11-24T03:35Z 批准）；旧 V2 6140 与 V3 诊断 200 另记，总计约 27340；不得再扩。
- **F0（无 force）/F1（简单残差融合）后补**：需用户另行批准预算/时间；届时同数据、同冻结场景、同 10-seed 评估；不属于本 Goal 当前范围。

## 交付与门槛
- 每 3k：checkpoint（`t7_21k_v3run`）+ `CHECKPOINT_MANIFEST`（path+sha256）+ 10 seed 双视角视频 + `G_D_3K_VERDICT`：close_frac≤0.95、open_cmd_frac≥0.05、reopen≥1、carry 左指垫力>0.5N 占比≥95%、exact_fill≥1；任一 fail → STOP 只出根因报告与视频 zip，不自动恢复。
- 12k：相对 3k >20% 退化 → 停跑复查。
- 21k：最终独立裸模型 10-seed 双视角测试（frozen scenes、exact_fill、20 步 hold、安全守卫）。
- 每里程碑：外部 sha256 备份尝试（HF 401 → 拆分 GitHub release / 本机转存），记录失败原因；9k 仍 0 外部副本且 free<30GB → 升级用户。
- 红线：不扩预算、不删不可替代产物、不写 token、不改判据、不用旧 Wan/p7_verify。

## 当前状态（2026-11-24T06:00Z）
- F2 run `t7_21k_v3run` 运行中（step≈2100/21000，pid 306531），G-A/B/C/G-D 管线 pass；3k 闸门 armed，ETA≈06:57Z。
- 外部备份 0 份；磁盘 38.8GiB（low，floor 30）；GPU 约 50h 估计（待控制台刷新）。

## 交付补充（2026-11-24T06:20Z 用户）
- **每个 seed 只交付 1 个三段并排视频**：左腕夹爪视角 | 前视 | 指尖力曲线（ch12/ch13，0–3N，0.5N carry 线，当前/峰值/opening/夹爪指令/目标计数；有安全事件时红字标注）。三段互不遮挡；不再单独交付 front/left_wrist 两个文件。3k/6k/… 与最终测试同规格（10 个 seed → 10 个 composite + 一个 zip）。

## 评测模型修正（2026-11-24T07:10Z 用户）
- **必须使用采集用 sim config**：`/workspace/fwv1/sim_repo/configs/cookie_batch.yaml`（左指垫半厚 **0.0005 m 薄长垫片**；target bin y≈0.100）。默认 `SimConfig` 的 0.003175 m 厚垫 + prototype target bin（y≈-0.011）夹不住饼干，已弃用。
- 冻结场景随模型重生成：`EVAL_SCENE_MANIFEST_V3_1.json`（同 80 seeds/options；freeze `cef07926…`；sim config sha `62f6c863…`）。旧的 V3 manifest（freeze `99194298…`）作废保留。
- 3k 评测用 step_003000 + V3.1 + 薄垫重跑；旧的厚垫 3k eval 作废（目录 `aborted_3k_thickpads_20261124`）。G-D 判据不变。
