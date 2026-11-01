cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers
ls
echo === cookie_expert
grep -nE "^class |batch_size|num_batches|target_grip_force|single|_cookies_per|def __init__|"""" cookie_expert.py | head -50
echo === grasp_expert
grep -nE "^class |batch_size|target_grip_force|single|def __init__|"""" grasp_expert.py | head -40
echo === which expert makes a3_fw
grep -rnE "A3SameColumnBatchExpert|A3CookieBatchExpert|A3CookieTransferExpert" /workspace/a3_dual_arm_sim/src /workspace/a3_dual_arm_sim/examples 2>/dev/null | grep -v ".pyc" | head -20
echo ===DONE