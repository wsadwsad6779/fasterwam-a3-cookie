cd /workspace/_r5_stage
cat > _patch1.py <<'PYEOF'
from pathlib import Path

NL = chr(10)
p = Path('/workspace/_r5_stage/replay_forces.py')
src = p.read_text(encoding='utf-8')
method = Path('/workspace/_r5_stage/_method.txt').read_text(encoding='utf-8')

anchor = '        super()._select_batch()' + NL
assert src.count(anchor) == 1, src.count(anchor)
if '_place_pose' not in src:
    src = src.replace(anchor, anchor + method, 1)

old_sig = 'def run_episode(seed, mode, max_steps, noise, out_dir):'
assert old_sig in src
src = src.replace(old_sig, 'def run_episode(seed, mode, max_steps, noise, out_dir, clamp_limit=0):', 1)

old_ctor = '    expert = (SingleCookieLowForce(env) if mode == ' + chr(34) + 'single' + chr(34) + ' else A3SameColumnBatchExpert(env))'
assert old_ctor in src, old_ctor
src = src.replace(old_ctor, old_ctor + NL + '    if clamp_limit:' + NL + '        expert.num_batches = int(clamp_limit)', 1)

old_call = '        row = run_episode(seed, args.mode, args.max_steps, args.noise, out)'
assert old_call in src
src = src.replace(old_call, '        row = run_episode(seed, args.mode, args.max_steps, args.noise, out, args.clamps)', 1)

old_noise = '    ap.add_argument(' + chr(34) + '--noise' + chr(34) + ', type=float, default=0.0)'
assert old_noise in src
src = src.replace(old_noise, old_noise + NL + '    ap.add_argument(' + chr(34) + '--clamps' + chr(34) + ', type=int, default=0)', 1)

p.write_text(src, encoding='utf-8')
print('patched ok')
PYEOF
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 _patch1.py && python3 -m py_compile replay_forces.py && echo COMPILE_OK
grep -nE 'clamp_limit|def _place_pose|--clamps' replay_forces.py
sha256sum replay_forces.py
