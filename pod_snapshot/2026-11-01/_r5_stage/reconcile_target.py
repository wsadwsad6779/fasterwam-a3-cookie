import sys
import numpy as np
BIN = np.asarray([0.095, 0.100])
HALF = np.asarray([0.057, 0.029])
for p in sys.argv[1:]:
    d = np.load(p, allow_pickle=True)
    xyz = np.asarray(d['cookie_xyz']); eef = np.asarray(d['eef_xyz'])
    op = np.asarray(d['opening']); pad = np.asarray(d['pad_force_inner'])
    n = len(xyz)
    dx = np.abs(xyz[:, :, 0] - BIN[0]); dy = np.abs(xyz[:, :, 1] - BIN[1])
    ins = (dx < HALF[0]) & (dy < HALF[1])
    cnt = ins.sum(axis=1)
    print('==', p.split('/')[-1], 'frames', n)
    print(' max_xy_inside_count', int(cnt.max()), 'at', int(cnt.argmax()), 'final', int(cnt[-1]))
    for t in [0, n // 3, 2 * n // 3, n - 1]:
        c = xyz[t]; e = eef[t]
        dd = np.linalg.norm(c[:, :2] - e[:2], axis=1)
        held = int(np.sum((dd < 0.05) & (np.abs(c[:, 2] - e[2]) < 0.06) & (op[t] < 0.05)))
        top = np.argsort(dx[t] + dy[t])[:4]
        print('  t=%3d ins=%2d open=%.3f eef=(%.3f,%.3f,%.3f) held=%d' % (t, int(ins[t].sum()), op[t], e[0], e[1], e[2], held))
        print('       near:', ['%d(%.3f,%.3f,z%.3f)' % (i, dx[t][i], dy[t][i], c[i][2]) for i in top])
    print(' pad max %.3f frames_over_2 %d' % (float(pad.max()), int((pad > 2.0).sum())))
