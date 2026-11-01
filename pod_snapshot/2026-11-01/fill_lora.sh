#!/bin/bash
P=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
U=https://hf-mirror.com/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/step_002000_merged.pt
D=/workspace/a3_train_runs/a3_lora_r1
SZ=11117977659
N=6
CH=$(( (SZ + N - 1) / N ))
cd $D
mkdir -p pc
a0=$(stat -c %s part0); a1=$((5*CH+$(stat -c %s part5))); b0=$CH; b1=$SZ
fetch() {
  s=$1; e=$2; out=$3; want=$((e-s+1))
  for t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    cur=$(stat -c %s $out 2>/dev/null || echo 0)
    [ $cur -ge $want ] && break
    curl -sL -x $P -r $((s+cur))-$e --speed-limit 2000000 --speed-time 8 --max-time 180 -o $out.tmp $U
    [ -s $out.tmp ] && cat $out.tmp >> $out
    rm -f $out.tmp
  done
  echo "done $out $(stat -c %s $out) want $want"
}
p0=$(( (b0-a0+3)/4 )); p1=$(( (b1-a1+3)/4 ))
for ((x=a0; x<b0; x+=p0)); do e=$((x+p0-1)); [ $e -ge $b0 ] && e=$((b0-1)); fetch $x $e pc/g0_$x & done
for ((x=a1; x<b1; x+=p1)); do e=$((x+p1-1)); [ $e -ge $b1 ] && e=$((b1-1)); fetch $x $e pc/g1_$x & done
wait
bad=0
for f in pc/g0_* pc/g1_*; do [ -s $f ] || bad=1; done
if [ $bad = 0 ]; then
  { cat part0; cat $(ls pc/g0_* | sort -t_ -k2 -n); cat part1 part2 part3 part4 part5; cat $(ls pc/g1_* | sort -t_ -k2 -n); } > step_002000_merged.pt
fi
stat -c %s step_002000_merged.pt
sha256sum step_002000_merged.pt
echo LORA_DONE >> /workspace/dl_lora.log
