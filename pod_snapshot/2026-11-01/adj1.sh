while pgrep -f 'replay_correct.py' > /dev/null; do sleep 15; done
sleep 5
date
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
python3 -u /workspace/_r5_stage/adjudicate_correct.py --dirs /workspace/a3_eval/demos_correct_c1
echo ADJ_DONE