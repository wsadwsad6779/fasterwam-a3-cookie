import sys
import numpy as np
sys.path.insert(0, '/workspace/_r5_stage')
import replay_forces as R
p = sys.argv[1]
d = np.load(p)
ph = [R.PHASE_NAMES[int(x)] for x in d['phase']]
ag = np.asarray(d['cookies_at_grip'])
pad = np.asarray(d['pad_force_inner'])
op = np.asarray(d['opening'])
xyz = np.asarray(d['cookie_xyz'])
z0 = np.asarray(d['cookie_xyz0'])
BIN = np.asarray(R.BIN_XY); HALF = np.asarray(R.BIN_INNER_HALF)
n = len(ph); prev = None; start = 0
for i in range(n + 1):
    key = (ph[i], int(ag[i])) if i < n else None
    if key != prev:
        if prev is not None:
            print('%4d-%4d %-16s grip=%d padmax=%.3f open=%.3f..%.3f' % (start, i - 1, prev[0], prev[1], float(pad[start:i].max()), float(op[start:i].min()), float(op[start:i].max())))
        prev = key; start = i
last = xyz[-1]
moved = np.linalg.norm(last - z0, axis=1) > 0.05
inbin = (np.abs(last[:, 0] - BIN[0]) < HALF[0]) & (np.abs(last[:, 1] - BIN[1]) < HALF[1])
print('n', n, 'moved', int(moved.sum()), 'inbin', int(inbin.sum()), 'both', int((moved & inbin).sum()))
print('moved_idx', np.nonzero(moved)[0].tolist())
print('inbin_idx', np.nonzero(inbin)[0].tolist())
names = np.asarray(ph); print('segs_names', len(R.segments(names)))
idx = np.asarray([R.PHASE_NAMES.index(x) for x in ph], dtype=np.int8); print('segs_idx', len(R.segments(idx)))
