import json
import time

import numpy as np

from t7_data import A3BatchSource
import json as _json
import numpy as _np

OUT = '/workspace/fwv1/p_items/SAMPLER_BALANCE_CHECK.json'
_st = _json.load(open('/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json'))['action']['default']
_amin = _np.asarray(_st['global_min']); _amax = _np.asarray(_st['global_max'])
_rng = _np.where((_amax - _amin) < 1e-4, 2.0, _amax - _amin)
_thr = float(_np.clip(0.8 * (2.0 / _rng[6]) + (-1.0 - (2.0 / _rng[6]) * _amin[6]), -5, 5))


def run(balanced, n=20):
    src = A3BatchSource(split='train', batch_size=8, seed=1, balanced=balanced)
    opens = []
    for _ in range(n):
        bb = src.next_batch()
        opens.append(float((bb['action'][:, :, 6] >= _thr).float().mean()))
    return {'balanced': balanced, 'batches': n,
            'mean_open_frac_ge0.8': float(np.mean(opens)),
            'batches_with_gt10pct_true_open': int(sum(1 for x in opens if x > 0.10)),
            'open_frac_values': [round(x, 4) for x in opens]}


res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'unbalanced': run(False), 'balanced': run(True)}
res['improvement_x'] = (res['balanced']['mean_open_frac_ge0.8'] /
                        max(1e-9, res['unbalanced']['mean_open_frac_ge0.8']))
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: res[k] for k in ['improvement_x']}))
print(json.dumps(res['unbalanced']))
print(json.dumps(res['balanced']))