"""Patch a3_wrap.py: stop excluding the source column that sits above the bin."""
import io

P = "a3_wrap.py"
s = io.open(P, encoding="utf-8").read()
orig = s

old_near = ('    _b = np.asarray(env.data.xpos[env._target_bin_body]); _h = np.asarray(env.TARGET_INNER_HALF_SIZE); '
            '_keep = (np.abs(cps[:, 0] - _b[0]) > _h[0] + 0.06) | (np.abs(cps[:, 1] - _b[1]) > _h[1] + 0.06); '
            'cps = cps[_keep] if np.any(_keep) else cps')
new_near = ('    _b = np.asarray(env.data.xpos[env._target_bin_body]); _h = np.asarray(env.TARGET_INNER_HALF_SIZE); '
            '_keep = (np.abs(cps[:, 0] - _b[0]) > _h[0] + 0.008) | (np.abs(cps[:, 1] - _b[1]) > _h[1] + 0.008); '
            'cps = cps[_keep] if np.any(_keep) else cps')
assert s.count(old_near) == 1
s = s.replace(old_near, new_near, 1)

old_bc = ('    keep = (np.abs(cps[:, 0] - b[0]) > h[0] + 0.06) | (np.abs(cps[:, 1] - b[1]) > h[1] + 0.06)\n'
          '    cps = cps[keep] if np.any(keep) else cps')
new_bc = ('    # Only drop cookies that are actually in the bin volume: a wide xy\n'
          '    # margin also hides the whole source column that sits above it.\n'
          '    keep = (np.abs(cps[:, 0] - b[0]) > h[0] + 0.008) | (np.abs(cps[:, 1] - b[1]) > h[1] + 0.008)\n'
          '    cps = cps[keep] if np.any(keep) else cps')
assert s.count(old_bc) == 1
s = s.replace(old_bc, new_bc, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("patched bytes", len(orig), "->", len(s), flush=True)
for i, line in enumerate(s.split("\n"), 1):
    if "0.008)" in line or "actually in the bin" in line:
        print(i, line.rstrip(), flush=True)
