# 大文件清单

本工作产生的 4 个大文件（约 48 GB）都在 HF 公开仓库 `wsadwsad6779/fasterwam-a3-cookie-lora`，云端实例销毁后直接从这里取。

| 云端路径 | 大小 (B) | HF 内路径 | 说明 |
|---|---|---|---|
| `/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt` | 11,117,977,659 | `step_002000_merged.pt` | 推理用，LoRA 已折叠，r26/r27 实际使用 |
| `/workspace/a3_train_runs/a3_lora_r1/checkpoints/weights/step_002000.pt` | 11,179,613,340 | `trainer/step_002000.pt` | 基座 + 未折叠 LoRA，可换 scaling 重新合并 |
| `/workspace/a3_train_runs/a3_lora_r1/checkpoints/state/step_002000/` | 12,712,079,410 | `trainer/state/step_002000/` | `model.safetensors` 12.59 GB + `optimizer.bin` 123 MB + scheduler/RNG |
| `/workspace/a3_data/a3_fw/frames.dat` | 12,849,371,136 | `data/frames.dat` | 42681 帧 uint8 [224,448,3] 训练 memmap，可由 parquet 重建 |

```bash
cd /workspace/a3_train_runs/a3_lora_r1
hf download wsadwsad6779/fasterwam-a3-cookie-lora step_002000_merged.pt --local-dir .
hf download wsadwsad6779/fasterwam-a3-cookie-lora trainer/step_002000.pt --local-dir ckpt_from_hf
hf download wsadwsad6779/fasterwam-a3-cookie-lora trainer/state/step_002000 --local-dir state_from_hf
```

以下两个不属于本工作，按外部地址重新下载即可：

| 云端路径 | 大小 | 来源 |
|---|---|---|
| `/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt` | 11,117,679,929 | HF `hustvl/FasterWAM`，sha256 `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d` |
| `/workspace/a3_data/a3-front-close-left-100/` | 4.1 GB 目录 | HF dataset `Eter0109/a3-front-close-left-100` |
