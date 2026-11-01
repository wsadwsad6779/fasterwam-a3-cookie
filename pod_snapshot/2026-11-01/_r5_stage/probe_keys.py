import sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
for k in sorted(d.files):
    a = d[k]
    print('%-34s %s %s' % (k, str(a.shape), str(a.dtype)))
for k in ['cookies_in_target', 'success', 'cookies_in_target_mask', 'terminated', 'truncated', 'phase', 'safety_rewritten']:
    if k in d.files:
        a = np.asarray(d[k])
        print('VAL', k, a.shape, str(a.reshape(-1)[:12]), 'last', str(a.reshape(-1)[-6:]))
