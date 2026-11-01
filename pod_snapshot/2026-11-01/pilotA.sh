cd /workspace
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
timeout 3000 python3 /workspace/_r5_stage/single_lowforce_probe.py --seeds 6001 --noise 0 --max-steps 3200 --out /workspace/a3_eval/pilot_single_lowforce_clean
echo PILOT_A_EXIT