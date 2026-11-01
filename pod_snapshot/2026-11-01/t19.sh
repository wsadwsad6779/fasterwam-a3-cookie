grep -rnE 'proprio_dim|text_dim' /workspace/FasterWAM/src/fasterwam/models/fastwam.py /workspace/FasterWAM/configs 2>/dev/null | head -20
echo ==TEXTDIM==
grep -rnE 'text_dim' /workspace/FasterWAM/src/fasterwam/*.py 2>/dev/null | head -12
echo ==STATE8==
grep -nE 'observation.state|state_dim|proprio|_state\b' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/data/recording.py | head -20
echo ==ENVSTATE==
grep -nE 'def _observation|observation.state' -A 8 /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py | sed -n '1,40p'
echo DONE