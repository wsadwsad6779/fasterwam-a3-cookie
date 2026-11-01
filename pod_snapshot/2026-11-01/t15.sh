cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
echo ==2X10_DONESTATE==
grep -nE 'DONE|required_stable|VERIFY|completed_cookie_indices|batch_index|num_batches|max_pad|_advance' cookie_2x10_expert.py
echo ==100_170==
sed -n '100,170p' cookie_2x10_expert.py
echo ==SAMECOL_500_528==
sed -n '500,528p' same_column_batch_expert.py
echo DONE