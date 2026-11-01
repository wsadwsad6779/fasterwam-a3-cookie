#!/usr/bin/env bash
P=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
U=https://hf-mirror.com/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/step_002000_merged.pt
D=/workspace/a3_train_runs/a3_lora_r1
OUT=$D/step_002000_merged.pt
SZ=11117977659
N=6
CH=$(( (SZ + N - 1) / N ))
cd $D
for i in $(seq 0 $((N-1))); do s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SZ ] && e=$((SZ-1)); curl -sL -x $P --retry 10 --retry-all-errors -r $s-$e -o part$i "$U" & done
wait
ok=1
for i in $(seq 0 $((N-1))); do s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SZ ] && e=$((SZ-1)); want=$((e-s+1)); got=$(stat -c %s part$i); echo "part$i want=$want got=$got"; [ "$want" != "$got" ] && ok=0; done
if [ $ok = 1 ]; then cat part0 part1 part2 part3 part4 part5 > "$OUT" && rm -f part?* && echo LORA_CONCAT_OK; else echo LORA_PARTS_BAD; fi
stat -c %s "$OUT" 2>/dev/null; echo LORA_DONE
