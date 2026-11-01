cd /workspace/_r5_stage
python3 - <<'PYEOF'
from pathlib import Path
p = Path('_patch1.py')
t = p.read_text(encoding='utf-8')
bad = "anchor = '        super()._select_batch()"
lines = t.split(chr(10))
for i, ln in enumerate(lines):
    if ln.startswith("anchor = "):
        lines[i] = 'anchor = "        super()._select_batch()" + chr(10)'
p.write_text(chr(10).join(lines), encoding='utf-8')
print(chr(10).join(lines[:12]))
PYEOF
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 _patch1.py && python3 -m py_compile replay_forces.py && echo COMPILE_OK
grep -nE 'clamp_limit|def _place_pose|--clamps' replay_forces.py
sha256sum replay_forces.py
