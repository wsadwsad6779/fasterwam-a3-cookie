#!/usr/bin/env bash
P=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
B=/workspace/FasterWAM/checkpoints
VAE=$B/Wan-AI/Wan2.2-TI2V-5B
TOK=$B/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
mkdir -p $B/fasterwam_release/libero $VAE $TOK
INC=$(ls $B/fasterwam_release/.cache/huggingface/download/libero/*.incomplete 2>/dev/null | head -1)
if [ -n "$INC" ] && [ ! -s $B/fasterwam_release/libero/step_021700.pt ]; then mv "$INC" $B/fasterwam_release/libero/step_021700.pt; echo RESUMED_FROM=$(stat -c %s $B/fasterwam_release/libero/step_021700.pt); fi
ms(){ curl -sL -x $P --retry 8 --retry-all-errors -C - -o "$2" "https://modelscope.cn/api/v1/models/$1/repo?Revision=master&FilePath=$3"; echo "MS_RC_$? $3 $(stat -c %s $2 2>/dev/null)"; }
ms hustvl/FasterWAM $B/fasterwam_release/libero/step_021700.pt libero/step_021700.pt
ms hustvl/FasterWAM $B/fasterwam_release/libero/dataset_stats.json libero/dataset_stats.json
ms Wan-AI/Wan2.2-TI2V-5B $VAE/Wan2.2_VAE.pth Wan2.2_VAE.pth
ms Wan-AI/Wan2.2-TI2V-5B $VAE/models_t5_umt5-xxl-enc-bf16.pth models_t5_umt5-xxl-enc-bf16.pth
for f in spiece.model tokenizer.json tokenizer_config.json special_tokens_map.json; do ms Wan-AI/Wan2.2-TI2V-5B $TOK/$f google/umt5-xxl/$f; done
echo MS_DONE
