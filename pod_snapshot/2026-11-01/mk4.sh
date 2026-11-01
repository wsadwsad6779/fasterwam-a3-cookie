cd /workspace/_r5_stage
python3 - <<'PYEOF'
from pathlib import Path
p = Path('replay_forces.py')
t = p.read_text(encoding='utf-8')
old = '''    inbin = (np.abs(rest[:, 0] - BIN_XY[0]) < BIN_INNER_HALF[0]
             & np.abs(rest[:, 1] - BIN_XY[1]) < BIN_INNER_HALF[1])'''
new = '''    inbin = ((np.abs(rest[:, 0] - BIN_XY[0]) < BIN_INNER_HALF[0])
             & (np.abs(rest[:, 1] - BIN_XY[1]) < BIN_INNER_HALF[1]))'''
assert old in t
p.write_text(t.replace(old, new, 1), encoding='utf-8')
print('patched')
PYEOF
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -m py_compile replay_forces.py && echo COMPILE_OK
sed -n '168,176p' replay_forces.py
sha256sum replay_forces.py
