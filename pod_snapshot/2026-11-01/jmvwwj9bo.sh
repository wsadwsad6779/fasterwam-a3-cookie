ls -1 /workspace/_r5_stage/*.py
echo FLAGS
grep -n add_argument /workspace/_r5_stage/replay_correct.py 2>/dev/null || grep -rn add_argument /workspace/_r5_stage/replay_correct*.py | head -20