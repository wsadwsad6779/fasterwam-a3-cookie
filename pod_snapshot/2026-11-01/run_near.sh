#!/bin/bash
source /workspace/activate_fasterwam_a3.sh >/dev/null 2>&1
cd /workspace/fasterwam_a3_bridge/bridge
rm -f /workspace/dagger_logs/near_all.done
python3 _next/dagger_recover.py --seed 11 --take 570 579 --dump-frames --no-video --out outputs/dagger_near > /workspace/dagger_logs/near_11.log 2>&1 &
P1=$!
python3 _next/dagger_recover.py --seed 21 --take 214 223 --dump-frames --no-video --out outputs/dagger_near > /workspace/dagger_logs/near_21.log 2>&1 &
P2=$!
wait $P1 $P2
python3 _next/dagger_recover.py --seed 22 --take 56 68 72 --dump-frames --no-video --out outputs/dagger_near > /workspace/dagger_logs/near_22.log 2>&1
touch /workspace/dagger_logs/near_all.done
