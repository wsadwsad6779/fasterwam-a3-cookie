#!/usr/bin/env bash
P=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
U=https://hf-mirror.com/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/step_002000_merged.pt
D=/workspace/a3_train_runs/a3_lora_r1
SZ=11117977659; N=6; CH=$(( (SZ+N-1)/N ))
pkill -f dl_lora.sh; pkill -f 'step_002000_merged.pt'; sleep 2
cd $D
tail_fill() {
  i=$1
  base_off=$((i*CH)); end=$((base_off+CH-1)); [ $i -eq $((N-1)) ] && end=$((SZ-1))
  cur=$(stat -c %s part$i)
  a=$((base_off+cur)); b=$end; rem=$((b-a+1)); n=4; c=$(( (rem+n-1)/n ))
  echo "part$i cur=$cur filling $a..$b"
  for j in 0 1 2 3; do s=$((a+j*c)); e=$((s+c-1)); [ $e -gt $b ] && e=$b; [ $s -gt $b ] && continue; curl -sL -x $P --retry 10 --retry-all-errors -r $s-$e -o p${i}_$j "$U" & done
  wait
  cat part$i p${i}_0 p${i}_1 p${i}_2 p${i}_3 > part$i.new 2>/dev/null
  mv part$i.new part$i; rm -f p${i}_*
  echo "part$i now $(stat -c %s part$i) want $((end-base_off+1))"
}
tail_fill 0
tail_fill 5
ok=1
for i in $(seq 0 $((N-1))); do base_off=$((i*CH)); end=$((base_off+CH-1)); [ $i -eq $((N-1)) ] && end=$((SZ-1)); want=$((end-base_off+1)); got=$(stat -c %s part$i); echo "part$i want=$want got=$got"; [ "$want" != "$got" ] && ok=0; done
if [ $ok = 1 ]; then cat part0 part1 part2 part3 part4 part5 > step_002000_merged.pt && rm -f part? && echo LORA_CONCAT_OK; else echo LORA_PARTS_BAD; fi
stat -c %s step_002000_merged.pt; sha256sum step_002000_merged.pt; echo LORA_DONE >> /workspace/dl_lora.log
