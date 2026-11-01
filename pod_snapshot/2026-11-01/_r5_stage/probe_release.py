import sys
import numpy as np
sys.path.insert(0, '/workspace/_r5_stage')
import replay_forces as R

D = '/workspace/a3_eval/demos_wrongclosure_v7/'
CUT = 105

for seed in (7043, 7044):
    z = np.load(D + 'expert_wrongclosure_' + str(seed) + '.npz')
    print('SEED', seed, 'keys', sorted(z.files))
    ag = np.asarray(z['cookies_at_grip'])
    n = len(ag)
    ph = [R.PHASE_NAMES[int(x)] for x in z['phase']]
    def arr(k):
        if k in z.files:
            return np.asarray(z[k], dtype=np.float64)
        return np.full(n, np.nan)
    ao = arr('applied_opening')
    op = arr('opening')
    pad = arr('pad_force_inner')
    lo = max(0, CUT - 24)
    hi = min(n, CUT + 24)
    for i in range(lo, hi):
        print('%4d %-16s grip=%d applied_open=%.3f open=%.3f pad=%.3f' % (i, ph[i], int(ag[i]), float(ao[i]), float(op[i]), float(pad[i])))
    idx = np.nonzero(ao >= 0.9)[0]
    g1 = np.nonzero(ag <= 1)[0]
    print('VAL first applied_open_ge_0.9', (int(idx[0]) if len(idx) else -1))
    print('VAL first grip_le_1', (int(g1[0]) if len(g1) else -1))
    print('VAL grip_min_window10', int(ag[CUT:min(n, CUT + 10)].min()))
    print('VAL applied_open_list', [round(float(x), 3) for x in ao[CUT:min(n, CUT + 20)]])
    print('VAL attempts', np.asarray(z['attempts']).tolist()[:6], 'changed_n', int(np.sum(np.asarray(z['changed']))))
    print('VAL open_list', [round(float(x), 3) for x in op[CUT:min(n, CUT + 20)]])
    print('VAL grip_list', [int(x) for x in ag[CUT:min(n, CUT + 20)]])
