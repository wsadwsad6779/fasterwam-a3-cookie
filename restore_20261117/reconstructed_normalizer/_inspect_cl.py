import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'closed_loop_p2clamp_20261118_v2'
for tag in ('B0', 'CL'):
    z = np.load(OUT / (tag + '.npz'), allow_pickle=False)
    ks = [k for k in z.files if not k.startswith('chunk_')]
    ck = [k for k in z.files if k.startswith('chunk_')]
    print(tag, 'non-chunk keys:', len(ks))
    print('  n chunk keys', len(ck), 'first', ck[:4])
    for k in ks:
        print('   ', k, z[k].shape, z[k].dtype)
