date -u
df -B1 /workspace | tail -1
sha256sum /workspace/_r5_stage/replay_wrongclosure.py
while test -d /proc/455933; do sleep 30; done
echo WC4_GONE $(date -u +%H:%M:%S)
cd /workspace/_r5_stage
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 -u replay_wrongclosure.py --seeds 7031 7032 7033 7034 7035 7036 --noise 0.0003 --recover-variants 1 --max-steps 800 --out /workspace/a3_eval/demos_wrongclosure_v9
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 adjudicate_wc.py /workspace/a3_eval/demos_wrongclosure_v9 --write
date -u