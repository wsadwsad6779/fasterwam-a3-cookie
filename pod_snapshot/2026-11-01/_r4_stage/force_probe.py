import glob
import json

import numpy as np

p = sorted(glob.glob('/workspace/a3_eval/r3_c0_diag_600/rep0/lora_C0_*.npz'))[0]
m = np.load(p)
print('FILE', p.rsplit('/', 1)[1])
print('KEYS', json.dumps(sorted(m.files)))
print('SHAPES', json.dumps({k: list(m[k].shape) for k in sorted(m.files)}))
print('META', str(m['meta'])[:600] if 'meta' in m.files else 'none')
