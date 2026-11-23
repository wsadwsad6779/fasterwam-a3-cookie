# HANDOFF / SUPERVISOR 协议

- Worker: fasterwam-fingertip-force-v1（session b462d38d-2bab-475d-a8b3-0dbd231385f9），generation 1。
- 共享状态：本目录 `WORKER_STATE.json`（每次真实执行后更新）；状态以 GitHub `fingertip-force-v1` 为准。
- 新 Worker 接管规则：若 generation 更高者出现，本 Worker 停止对 pod /workspace/fwv1 与 HF 的训练/数据写入。
- 基座锁定：唯一基座 hustvl/FasterWAM step_021700.pt（8D proprio/7D action，无 force 键）；禁止旧 A3 无力 checkpoint 与 Force40k/C 类权重。
- 已跑通：P1 基座身份、P3 处理器模块、P4 零初始化 FiLM、P5 标签 V2 事件审计、P6 元数据定位（845 episodes）。
- 下一步（已授权）：解析 parquet schema 与 stats，核对 joint_position→7D cartesian 的转换与力覆盖，然后做去重/多样性。
- 训练（P9）阻断：等待显式步数预算。
