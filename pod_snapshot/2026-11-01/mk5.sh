cd /workspace/_r5_stage
cat > _patch1.py <<'PYEOF'
from pathlib import Path

p = Path('/workspace/_r5_stage/replay_forces.py')
src = p.read_text(encoding='utf-8')

method = Path('/workspace/_r5_stage/_method.txt').read_text(encoding='utf-8')
anchor = '        super()._select_batch()
'
assert src.count(anchor) == 1, src.count(anchor)
if '_place_pose' not in src:
    src = src.replace(anchor, anchor + method, 1)

old_sig = 'def run_episode(seed, mode, max_steps, noise, out_dir):'
assert old_sig in src
src = src.replace(old_sig, 'def run_episode(seed, mode, max_steps, noise, out_dir, clamp_limit=0):', 1)

old_ctor = '    expert = (SingleCookieLowForce(env) if mode == "single" else A3SameColumnBatchExpert(env))'
assert old_ctor in src
src = src.replace(old_ctor, old_ctor + '''
    if clamp_limit:
        expert.num_batches = int(clamp_limit)''', 1)

old_call = '        row = run_episode(seed, args.mode, args.max_steps, args.noise, out)'
assert old_call in src
src = src.replace(old_call, '        row = run_episode(seed, args.mode, args.max_steps, args.noise, out, args.clamps)', 1)

old_noise = '    ap.add_argument("--noise", type=float, default=0.0)'
assert old_noise in src
src = src.replace(old_noise, old_noise + '''
    ap.add_argument("--clamps", type=int, default=0,
                    help="override clamps per episode, single mode only")''', 1)

p.write_text(src, encoding='utf-8')
print('patched ok')
PYEOF
cat > _method.txt <<'MEOF'

    def _place_pose(self, clearance=0.0, column_index=None):
        # One cookie per clamp gets its own row. The target grid puts 6.33 mm
        # between rows and the slab is 6.34 mm thick, so releasing every cookie
        # at the column mean row leans the last ones against a neighbour.
        if column_index is None:
            order = getattr(self.env, "demonstration_column_order", (0, 1))
            column_index = order[0] if self.batch_index < 5 else order[1]
        position, rotation = A3SameColumnBatchExpert._place_pose(self, clearance, column_index)
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        column_x = np.unique(slots[:, 0])[column_index]
        rows = np.sort(slots[np.isclose(slots[:, 0], column_x), 1])
        offset = rows[self.batch_index % len(rows)] - rows.mean()
        box_r = self.data.xmat[self._tb_id].reshape(3, 3)
        return position + box_r @ np.array([0.0, offset, 0.0]), rotation
MEOF
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 _patch1.py && python3 -m py_compile replay_forces.py && echo COMPILE_OK
grep -nE 'clamp_limit|_place_pose|--clamps' replay_forces.py | head -20
sha256sum replay_forces.py
tail -c 1200 /workspace/sm2.out
