grep -n "def run_episode" /workspace/_r5_stage/replay_forces.py
awk "/^def run_episode/,/^def [a-z_]+(.*):$/" /workspace/_r5_stage/replay_forces.py | sed -n "1,60p"
