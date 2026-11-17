# GPU 最短动作块前置验证 — 状态报告

## 验证结果

- **脚本**：`/workspace/eval_restore_20261117/gpu_shortest_action.py`
- **运行日志**：`/workspace/eval_restore_20261117/gpu_shortest_action_run2.log`
- **设备**：`cuda`
- **模型数据类型**：`torch.bfloat16`
- **推理接口**：`JointWAM.infer_action(..., num_video_frames=17, num_inference_steps=5, force_history=..., force_history_mask=...)`
- **输出动作形状**：`(32, 7)`
- **动作数值范围**：`[-0.89453125, 1.515625]`
- **耗时**：`66.1 s`
- **状态**：`GPU_SHORTEST_ACTION_OK`

## 关键资产清单

| 路径 | 说明 | 大小/状态 |
|---|---|---|
| `step_040000_merged.pt` | 40k 合并权重（基座 + LoRA + 力感知额外权重） | 11.1 GB |
| `step_040000/` | v3 轻量 checkpoint（adapter/optimizer/scheduler/rng/trainer_state/manifest/force_calib_v1.json） | 目录 |
| `deploy/stage_40k/fw/` | 固定提交代码与配置 | 目录 |
| `auxiliary/` | 本地 Wan2.2/T5/tokenizer 文件（DIFFSYNTH_MODEL_BASE_PATH） | 目录 |
| `cpu_restore_acceptance.json` | CPU 验收通过记录 | 文件 |
| `gpu_shortest_action_run2.log` | GPU 最短动作块运行日志 | 文件 |
| `export_merged4.log` | 合并发布日志 | 文件 |
| `transfer_manifest.json` | 同步资产清单 | 文件 |

## 已修复问题

- `gpu_shortest_action.py` 最初按 `FastWAM.infer_action` 签名调用，未提供 `num_video_frames`；实际实例化模型为 `FasterWAM(JointWAM)`，其 `infer_action` 要求该参数，并需同时传入力历史/掩码。脚本已更新并验证通过。

## 后续注意

- 本验证仅确认 40k 合并权重可在 GPU 上完成一次真实动作块推理，未启动完整回合评估或训练。
- 完整评估（160 回合固定场景）仍需单独排队授权。
