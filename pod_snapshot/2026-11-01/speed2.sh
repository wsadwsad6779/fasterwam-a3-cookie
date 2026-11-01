#!/bin/bash
P=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
U=https://hf-mirror.com/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
for i in 1 2 3 4; do (curl -sL -x $P -m 8 -r $((i*1000000000))-$((i*1000000000+400000000)) -o /dev/null -w "par$i %{speed_download}""\n" $U) & done; wait
echo ---ms_hustvl; curl -s -m 6 -x $P -o /dev/null -w '%{http_code}' https://modelscope.cn/api/v1/models/hustvl/FasterWAM/repo/files?Revision=master; echo
echo ---ms_lora; curl -s -m 6 -x $P -o /dev/null -w '%{http_code}' https://modelscope.cn/api/v1/models/wsadwsad6779/fasterwam-a3-cookie-lora/repo/files?Revision=master; echo
echo SPEED2_DONE
