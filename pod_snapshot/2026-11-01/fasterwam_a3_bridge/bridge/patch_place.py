"""Patch a3_wrap.py: tight 5-window selection + measured placement."""
import io

P = "a3_wrap.py"
s = io.open(P, encoding="utf-8").read()
orig = s

old_state = '        "placed": 0,\n    }'
new_state = ('        "placed": 0,\n'
             '        "boff": 0.0,\n'
             '        "bmin": 0.0,\n'
             '        "bmax": 0.0,\n'
             '        "gx": 0.0,\n'
             '        "gcyc": -1,\n'
             '    }')
assert s.count(old_state) == 1
s = s.replace(old_state, new_state, 1)

# --- pick the tightest 5-consecutive window instead of the front 5 ---------
old_tail = ('    order = col[np.argsort(col[:, 1])]\n'
            '    w = order[:5]\n'
            '    return float(w[:, 0].mean()) + 0.0067, float(w[:, 1].mean()) + BATCH_DY')
new_tail = ('    order = col[np.argsort(col[:, 1])]\n'
            '    xs = order[:, 0]\n'
            '    if len(xs) >= 5:\n'
            '        # The front of a column gets sheared by the approach, so take\n'
            '        # the 5 consecutive cookies that are still aligned in x.\n'
            '        i = min(range(len(xs) - 4),\n'
            '                key=lambda k: xs[k:k + 5].max() - xs[k:k + 5].min())\n'
            '        order = order[i:i + 5]\n'
            '    w = order[:5]\n'
            '    return float(w[:, 0].mean()) + 0.0067, float(w[:, 1].mean()) + BATCH_DY')
assert s.count(old_tail) == 1
s = s.replace(old_tail, new_tail, 1)

anchor = 'def apply(a3, obs, env, st, prof):'
helpers = '''def batch_offsets(env, tcp, cpos):
    """Held batch x offsets relative to the tool (mean, min, max)."""
    cps = env.cookie_positions
    d = np.linalg.norm(cps[:, :2] - np.asarray(tcp)[:2], axis=1)
    xs = cps[np.argsort(d)[:5], 0]
    return float(xs.mean() - tcp[0]), float(xs.min() - tcp[0]), float(xs.max() - tcp[0])


def goal_tcp_x(env, tcp, bin_pos, half, st):
    """Freeze a tool-x goal that centres the held batch in the free gap.

    The batch is not centred on the tool, so aim its measured extremes instead
    of a fixed offset from the bin centre.
    """
    boff, bmin, bmax = st["boff"], st["bmin"], st["bmax"]
    cps = env.cookie_positions
    dxy = np.linalg.norm(cps[:, :2] - np.asarray(tcp)[:2], axis=1)
    inb = ((np.abs(cps[:, 0] - bin_pos[0]) < half[0] - 0.004)
           & (np.abs(cps[:, 1] - bin_pos[1]) < half[1] - 0.004)
           & (cps[:, 2] < bin_pos[2] + 0.030)
           & (dxy > 0.06))
    gap_l = float(bin_pos[0] - half[0]) + 0.0015
    if np.any(inb):
        gap_r = float(cps[inb, 0].min()) - 0.0265
        lo = gap_l - bmin + 0.025
        hi = gap_r - bmax - 0.025
        return float(0.5 * (lo + hi))
    return float(bin_pos[0] + 0.028 - boff)


'''
assert s.count(anchor) == 1
s = s.replace(anchor, helpers + anchor, 1)

old_aim = ('                st["steer"] += 1\n'
           '                ex = float(bin_pos[0] + (0.032 if st["placed"] % 2 == 0 else -0.024) - tcp[0])')
new_aim = ('                if st["gcyc"] != st["cycles"]:\n'
           '                    st["boff"], st["bmin"], st["bmax"] = batch_offsets(env, tcp, cpos)\n'
           '                    st["gx"] = goal_tcp_x(env, tcp, bin_pos, half, st)\n'
           '                    st["gcyc"] = st["cycles"]\n'
           '                st["steer"] += 1\n'
           '                ex = float(st["gx"] - tcp[0])')
assert s.count(old_aim) == 1
s = s.replace(old_aim, new_aim, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("patched bytes", len(orig), "->", len(s), flush=True)
for i, line in enumerate(s.split("\n"), 1):
    if any(k in line for k in ("def batch_offsets", "def goal_tcp_x", '"gcyc"', '"gx": 0.0',
                               "tight", "min(range(len(xs)")):
        print(i, line.rstrip(), flush=True)
