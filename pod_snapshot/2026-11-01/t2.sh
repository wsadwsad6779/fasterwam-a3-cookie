cd /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim
echo ==EXPERT_SINGLE==
grep -nE '^class |force|FORCE|opening|OPENING' controllers/expert.py | head -45
echo ==EXPERT_HEAD==
sed -n '1,35p' controllers/expert.py
echo ==BATCH_HEAD==
sed -n '1,55p' controllers/batch_expert.py
echo ==TRANSFER_CLI==
sed -n '1,75p' workflows/run_cookie_transfer_cli.py
echo DONE