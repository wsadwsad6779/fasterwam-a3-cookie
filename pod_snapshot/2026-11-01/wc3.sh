export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
date
python3 -u /workspace/_r5_stage/replay_wrongclosure.py --seeds 7031 7032 7033 7034 7035 7036 --noise 0.0003 --recover-variants 1 --max-steps 800 --out /workspace/a3_eval/demos_wrongclosure_v6
date