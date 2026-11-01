cd /workspace/_r5_stage
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -u replay_correct.py --seeds 7266 7267 7268 7269 7270 7271 --variants 15 19 23 15 19 23 --noise 0.0003 --max-steps 800 --out /workspace/a3_eval/demos_correct_c6