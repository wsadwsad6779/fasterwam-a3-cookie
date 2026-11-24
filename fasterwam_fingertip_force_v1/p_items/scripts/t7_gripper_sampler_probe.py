"""Verify the fixed gripper class balance in the training batch source."""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/workspace')

OUT = '/workspace/fwv1/p_items/GRIPPER_SAMPLER_PROBE.json'


def main():
    from t7_data import A3BatchSource
    t0 = time.time()
    src = A3BatchSource(split='train', batch_size=8, seed=123, balanced=True)
    n = 50
    open_fr = []
    close_fr = []
    for _ in range(n):
        bb = src.next_batch()
        a = bb['action'][:, :, 6]
        open_fr.append(float((a <= -0.6).float().mean()))
        close_fr.append(float((a >= 0.6).float().mean()))
    out = {'worker': 'supervisor', 'batches': n,
           'open_frac_mean': float(np.mean(open_fr)),
           'open_frac_median': float(np.median(open_fr)),
           'close_frac_mean': float(np.mean(close_fr)),
           'batches_with_open': int(sum(1 for x in open_fr if x > 0)),
           'batches_mixed': int(sum(1 for o, c in zip(open_fr, close_fr) if o > 0 and c > 0)),
           'per_batch_open_head': [round(x, 4) for x in open_fr[:12]],
           'elapsed_s': round(time.time() - t0, 1),
           'note': 'labels pm1: +1 close, -1 open; open fraction in sampled windows vs dataset global 6.5%'}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()