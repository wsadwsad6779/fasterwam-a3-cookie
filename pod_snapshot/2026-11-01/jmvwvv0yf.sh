echo WC5AFULL
cat /workspace/wc5a.out
echo CC3TAIL
grep -E "TALLY|WROTE|dir." /workspace/cc3.out | tail -6
echo SIG
grep -n -A 12 "def cookies_at_grip" /workspace/_r5_stage/replay_wrongclosure.py