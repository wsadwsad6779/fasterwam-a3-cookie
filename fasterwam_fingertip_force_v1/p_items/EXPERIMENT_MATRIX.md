# EXPERIMENT_MATRIX（修订 2026-11-23）

唯一基座：`hustvl/FasterWAM` 官方 `libero/step_021700.pt`（proprio 8 / action 7 / 无 force conditioner）。
禁止旧 A3 无力 checkpoint、旧 Force40k、C 类权重参与任何一档。

| 模型 | 夹爪标签 | 指尖力 | 说明 |
|---|---|---|---|
| F0 | V2 | 无 | 官方基座 + A3 示范，隔离“标签修复”收益 |
| F1 | V2 | 简单残差融合 | 同数据同预算 |
| F2 | V2 | 零初始化 FiLM | 同数据同预算 |

注意：官方 LIBERO 权重不自动具备 A3 能力；F0/F1/F2 均需新的 A3 任务适配训练，且必须同数据、同预算、同评估协议。
