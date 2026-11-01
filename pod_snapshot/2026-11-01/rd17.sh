cd /workspace
grep -E '^{' /workspace/gate.out | head -30
echo '--- stops ---'
grep -oE '"stop": "[a-z_:]+' /workspace/gate.out | sort | uniq -c
ls /workspace/a3_eval/ | grep replay
