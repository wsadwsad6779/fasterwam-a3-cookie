grep -nE 'python|-m |collect|seed' /workspace/a3_dual_arm_sim/scripts/collect_sharded.sh | sed -n '1,40p'
echo '===== lerobot writers ====='
grep -rlnE 'LeRobotDataset|lerobot' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim | head
