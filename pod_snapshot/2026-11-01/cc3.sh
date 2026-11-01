export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
date
python3 -u /workspace/_r5_stage/replay_correct.py --seeds 7248 7249 7250 7251 7252 7253 --variants 3 7 11 27 3 7 --noise 0.0003 --retries 2 --recover-variants 0 37 5 --max-steps 2400 --out /workspace/a3_eval/demos_correct_c3
python3 /workspace/_r5_stage/adjudicate_correct.py --dirs /workspace/a3_eval/demos_correct_c3
date