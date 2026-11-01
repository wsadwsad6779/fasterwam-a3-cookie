cd /workspace
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
timeout 5400 python3 /workspace/_r5_stage/single_lowforce_probe.py --seeds 6002 6003 --noise 0.0003 --max-steps 3200 --out /workspace/a3_eval/pilot_single_lowforce_noisy
echo PILOT_B_EXIT