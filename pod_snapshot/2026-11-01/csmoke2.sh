export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
date
python3 -u /workspace/_r5_stage/replay_correct.py --seeds 7401 7402 --variants 2 3 --noise 0.0003 --retries 1 --max-steps 1600 --out /workspace/a3_eval/demos_correct_smoke
date
echo __END__