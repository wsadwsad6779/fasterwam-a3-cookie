cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
echo ==PLACEMENT_ORDER==
sed -n '1,40p' placement_order.py
echo ==PLACE_POSE_BASE==
grep -nE '_place_pose|ordered_column|def |rows|slot' batch_expert.py | head -30
echo ==SAMECOL_PLACE==
grep -nE '_place_pose|ordered_column' same_column_batch_expert.py | head
echo ==B==
tail -c 400 /workspace/pilotB.out
echo DONE