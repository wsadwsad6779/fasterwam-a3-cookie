cd /workspace/_r5_stage
grep -nE "^def |^HOLD_XY|^LIFT_Z" replay_forces.py
echo == CLAMPSTATS
sed -n "$(grep -n "def clamp_stats" replay_forces.py | cut -d: -f1),+45p" replay_forces.py