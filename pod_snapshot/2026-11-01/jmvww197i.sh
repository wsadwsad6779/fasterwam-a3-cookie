cd /workspace/_r5_stage
grep -n "release_opening\|min_grip_opening\|secure_grip_opening" replay_forces.py
echo ACTBODY
grep -n -A 30 "def act" replay_forces.py | head -70