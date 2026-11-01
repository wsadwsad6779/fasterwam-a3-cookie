cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
grep -nE '_held_offsets' batch_expert.py same_column_batch_expert.py cookie_2x10_expert.py
echo ==CTX==
sed -n '425,455p' same_column_batch_expert.py
echo DONE