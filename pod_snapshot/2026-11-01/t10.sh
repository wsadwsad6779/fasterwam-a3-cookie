cd /workspace
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -c 'import sys, mujoco, a3_dual_arm_sim; print(sys.version.split()[0], mujoco.__version__)'
echo ==2X10CFG==
cat /workspace/a3_dual_arm_sim/configs/cookie_2x10.yaml
echo ==RANDOMIZATION==
cat /workspace/a3_dual_arm_sim/configs/randomization_2x10.yaml
echo ==DRYRUN==
timeout 120 python3 /workspace/a3_dual_arm_sim/examples/collect_cookie_2x10.py --dry-run --first-grasp 1 --episodes 2 --root /workspace/a3_data/_pilot_check 2>&1 | tail -30
echo DONE