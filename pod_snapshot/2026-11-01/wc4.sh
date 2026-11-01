export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
date
python3 -u /workspace/_r5_stage/replay_wrongclosure.py --seeds 7037 7038 7039 7040 7041 7042 --noise 0.0003 --recover-variants 2 --max-steps 800 --out /workspace/a3_eval/demos_wrongclosure_v6b
python3 /workspace/_r5_stage/adjudicate_wc.py /workspace/a3_eval/demos_wrongclosure_v6b
date