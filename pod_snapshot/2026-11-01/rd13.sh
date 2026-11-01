cd /workspace
echo '=== run_matrix copies ==='
find /workspace -maxdepth 4 -name 'run_matrix*.py' -not -path '*/proc/*' 2>/dev/null | head -10
find /workspace -maxdepth 4 -name 'run_matrix.b64' -o -maxdepth 4 -name 'pod_run_matrix.b64' | head
echo '=== r4 launch cmd ==='
grep -rlE 'run_matrix' /workspace/_r4_stage/*.sh /workspace/_r4_stage/up/*.sh /workspace/*.sh 2>/dev/null | head
ls /workspace/_r4_stage /workspace/_r4_stage/up
echo '=== windows.log ==='
cat /workspace/a3_train_runs/a3_lora_r5/windows.log 2>/dev/null | tail -20
echo '=== preflight ==='
cat /workspace/a3_train_runs/a3_lora_r5/preflight.txt 2>/dev/null | head -30
