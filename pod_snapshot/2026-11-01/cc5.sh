date -u
df -B1 /workspace | tail -1
cd /workspace/_r5_stage
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 -u replay_correct.py --seeds 7260 7261 7262 7263 7264 7265 --variants 11 27 3 7 11 27 --noise 0.0003 --max-steps 800 --out /workspace/a3_eval/demos_correct_c5
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 adjudicate_correct.py /workspace/a3_eval/demos_correct_c5
date -u