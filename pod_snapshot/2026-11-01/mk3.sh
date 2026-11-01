cd /workspace/_r5_stage
python3 - <<'PYEOF'
from pathlib import Path
p = Path('replay_forces.py')
txt = p.read_text(encoding='utf-8')
anchor = '    expert.reset()'
if 'z0_live' not in txt:
    txt = txt.replace(anchor, anchor + '\n    z0_live = np.asarray(env.cookie_positions, dtype=np.float64).copy()', 1)
old = '    z0 = np.asarray(info.get("cookie_positions", xyz[0]), dtype=np.float64)'
assert old in txt, txt.count('z0')
txt = txt.replace(old, '    z0 = z0_live', 1)
p.write_text(txt, encoding='utf-8')
print('patched')
PYEOF
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -m py_compile replay_forces.py && echo COMPILE_OK
grep -nE 'z0_live|z0 = z0_live|carry_peak_wrist|env_close' replay_forces.py
sha256sum replay_forces.py
