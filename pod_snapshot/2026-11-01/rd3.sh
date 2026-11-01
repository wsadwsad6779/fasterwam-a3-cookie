export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim
echo '=== upright check ==='
grep -rnE 'not upright|settled and contained' controllers tasks sim | head -20
echo '=== first_grasp ==='
grep -rnE 'first_grasp' controllers workflows data | head -20
echo '=== cookie_expert knobs ==='
grep -nE 'target_grip_force|stable_grip_force|min_grip_opening|secure_grip_opening|release_opening|batch_size|num_batches|MAX_INSERTION|_max_lift_speed|class ' controllers/cookie_expert.py | head -40
echo '=== r5 coverage ==='
ls /workspace/a3_train_runs/a3_lora_r5/
grep -cE 'epoch=1' /workspace/a3_train_runs/a3_lora_r5/train.log
head -40 /workspace/a3_train_runs/a3_lora_r5/train.log
