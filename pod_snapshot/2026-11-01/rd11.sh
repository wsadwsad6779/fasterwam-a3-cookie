export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace
echo '=== candidate batches ==='
grep -rnE 'def _candidate_batches' -A 25 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/batch_expert.py | head -45
echo '=== CookiePhase ==='
grep -rnE 'class CookiePhase' -A 16 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers | head -25
echo '=== batch reports ==='
grep -rnE 'batch_reports|def finished|def failed|failure_reason|max_pad_force' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/batch_expert.py | head -20
