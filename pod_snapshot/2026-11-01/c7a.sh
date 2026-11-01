cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -u replay_correct.py --seeds 7272 7273 7274 7275 7276 7277 --variants 11 11 11 11 11 11 --noise 0.0003 --max-steps 800 --out /workspace/a3_eval/demos_correct_c7