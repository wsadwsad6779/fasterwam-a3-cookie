grep -n -B 4 -A 24 "RELEASE_OPEN$" /workspace/_r5_stage/replay_wrongclosure.py
echo CC12
grep -E "TALLY|dir." /workspace/cc1.out /workspace/cc2.out 2>/dev/null | tail -6