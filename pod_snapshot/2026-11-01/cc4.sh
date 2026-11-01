date -u
df -B1 /workspace | tail -1
cd /workspace/_r5_stage
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 -u replay_correct.py --seeds 7254 7255 7256 7257 7258 7259 --variants 3 7 11 27 3 7 --noise 0.0003 --max-steps 800 --out /workspace/a3_eval/demos_correct_c4
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 adjudicate_correct.py /workspace/a3_eval/demos_correct_c4
date -u