#!/bin/bash
set -u
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge || exit 1
mkdir -p /workspace/dagger_logs
for spec in 22:40 22:60 22:70 21:76 11:90; do
  s=${spec%%:*}
  t=${spec##*:}
  echo "[dump] start seed=$s take=$t $(date +%H:%M:%S)"
  python3 _next/dagger_recover.py --seed "$s" --take "$t" --dump-frames --no-video --out outputs/dagger > /workspace/dagger_logs/dump_${s}_${t}.log 2>&1
  echo "[dump] done seed=$s take=$t rc=$? $(date +%H:%M:%S)"
done
touch /workspace/dagger_logs/dump_all.done
