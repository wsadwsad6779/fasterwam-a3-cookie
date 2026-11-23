# fasterwam_fingertip_force_v1

新研究线：原始 FasterWAM + 正确夹爪监督 + 轻量指尖力融合（F0/F1/F2）。
旧实验（Force40k/G3/M5、action_expert_v2/v3/v4）冻结不动，其归档在 GitHub main 与两个 HF 仓。

## 当前进度
- 第一阶段进行中：`BASE_FASTERWAM_IDENTITY.md`（代码级已确认；checkpoint 级 1 项待验证）
- 任务表：`PROJECT_TASKS.json`；状态：`PROJECT_STATE.json`；证据台账：`PROJECT_EVIDENCE_LEDGER.json`
- 阻断：B1 无 GPU 实例（资源池满）；B2 HF 私有配额满（读 40k manifest 需 token 或公开副本）

## 交付物索引（第一阶段起）
- P1 身份：`BASE_FASTERWAM_IDENTITY.md` / `.json`
- P2 论文审计：`PAPER_REVIEW.md` / `FORCE_MODULE_PORTABILITY_MATRIX.json` / `FUSION_DESIGN_DECISION.md`
- P3 指尖力：`FINGERTIP_SENSOR_AUDIT.md` / `FINGERTIP_FORCE_PROCESSOR_TEST.json`
- P4 融合：`FINGERTIP_FUSION_ARCHITECTURE.md`
- P5 夹爪标签：`GRIPPER_LABEL_V2_AUDIT.md` / `GRIPPER_EVENTS_PER_EPISODE.csv` / `GRIPPER_LABEL_V2_MANIFEST.json`
- P6 数据审计：`DATASET_800_*`
