date -u
df -B1 /workspace | tail -1
cd /workspace/_r5_stage
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 -u replay_wrongclosure.py --seeds 7043 7044 --noise 0.0003 --recover-variants 1 --max-steps 800 --out /workspace/a3_eval/demos_wrongclosure_v8
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 adjudicate_wc.py /workspace/a3_eval/demos_wrongclosure_v8 --write
date -u