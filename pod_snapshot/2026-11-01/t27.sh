cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
grep -nE '_max_lift_speed|_batch_pinch_height_m' batch_expert.py same_column_batch_expert.py cookie_2x10_expert.py | head -20
echo DONE