#!/usr/bin/env bash
set -x
export HF_HOME=/workspace/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
B=/workspace/FasterWAM/checkpoints
VAE=$B/Wan-AI/Wan2.2-TI2V-5B
TOK=$B/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
mkdir -p "$B/fasterwam_release" "$VAE" "$TOK" /workspace/a3_train_runs/a3_lora_r1
hf download hustvl/FasterWAM --include "libero/step_021700.pt" "libero/dataset_stats.json" --local-dir "$B/fasterwam_release"
echo BASE_RC_$?
hf download wsadwsad6779/fasterwam-a3-cookie-lora step_002000_merged.pt --local-dir /workspace/a3_train_runs/a3_lora_r1
echo LORA_RC_$?
curl -L -C - --retry 5 -o "$VAE/Wan2.2_VAE.pth" https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth
echo VAE_RC_$?
curl -L -C - --retry 5 -o "$VAE/models_t5_umt5-xxl-enc-bf16.pth" https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
echo T5_RC_$?
for f in special_tokens_map.json tokenizer_config.json spiece.model tokenizer.json; do curl -L --retry 5 -o "$TOK/$f" "https://hf-mirror.com/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/google/umt5-xxl/$f"; done
echo DL_ALL_DONE
