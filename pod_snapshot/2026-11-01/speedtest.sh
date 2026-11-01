#!/bin/bash
P=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
t(){ s=$(curl -sL -x $P -m 8 -r 0-300000000 -o /dev/null -w '%{speed_download}' $2); echo $1 ${s}; }
t mirror_t5 https://hf-mirror.com/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
t hf_t5 https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
t ms_t5 'https://modelscope.cn/api/v1/models/Wan-AI/Wan2.1-T2V-1.3B/repo?Revision=master&FilePath=models_t5_umt5-xxl-enc-bf16.pth'
echo SPEED_DONE
