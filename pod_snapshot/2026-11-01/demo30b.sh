export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
df -B1 /workspace | tail -1
echo "=== A noise 0"
python3 /workspace/_r5_stage/replay_forces.py --mode single --clamps 1 --noise 0 --seeds 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110 --out /workspace/a3_eval/demos_single_n0
df -B1 /workspace | tail -1
echo "=== B noise 0.0003"
python3 /workspace/_r5_stage/replay_forces.py --mode single --clamps 1 --noise 0.0003 --seeds 7111 7112 7113 7114 7115 7116 7117 7118 7119 7120 --out /workspace/a3_eval/demos_single_n3
df -B1 /workspace | tail -1
echo "=== C noise 0.0006"
python3 /workspace/_r5_stage/replay_forces.py --mode single --clamps 1 --noise 0.0006 --seeds 7121 7122 7123 7124 7125 7126 7127 7128 7129 7130 --out /workspace/a3_eval/demos_single_n6
df -B1 /workspace | tail -1
echo DEMO30B_COMPLETE
