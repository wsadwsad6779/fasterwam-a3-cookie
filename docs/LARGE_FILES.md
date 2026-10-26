# 未打包的大文件（关闭实例前请自行下载/确认可重建）

| 路径 | 大小 (B) | 说明 |
|---|---|---|
| `/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt` | 11,117,977,659 | LoRA 合并后的 5.56B checkpoint（r26/r27 实际使用；已上传 HF） |
| `/workspace/a3_train_runs/a3_lora_r1/checkpoints/weights/step_002000.pt` | 11,179,613,340 | 训练器保存的 LoRA checkpoint（可续训 / 重新合并） |
| `/workspace/a3_train_runs/a3_lora_r1/checkpoints/state` | （目录） | 优化器与训练状态（仅精确续训需要） |
| `/workspace/a3_data/a3_fw/frames.dat` | 12,849,371,136 | 42681 帧 uint8 [224,448,3] 训练 memmap，可由 parquet 重新生成 |
| `/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt` | 11,117,679,929 | 官方基座 checkpoint（训练 resume 起点） |
| `/workspace/a3_data/a3-front-close-left-100` | （目录） | HF 专家数据集 4.1 GB（可重新下载） |

- merged ckpt 已上传到 HF：`hf download wsadwsad6779/fasterwam-a3-cookie-lora step_002000_merged.pt --local-dir <dir>`
- frames.dat / 数据集 / 基座权重都可由外部源重建，不必下载。
