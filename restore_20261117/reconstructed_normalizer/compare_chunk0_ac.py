"""Fixed-input comparison: chunk 0 shares the reset observation and the noise seed between
A and C, so the first 10 executed actions are comparable without any new GPU run."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'c_class_eval_20261119'
NAMES = ['dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'grip']


def main():
    res = {}
    for seed in (29, 30):
        a = np.load(OUT / ('episode_A_seed%d.npz' % seed))
        c = np.load(OUT / ('episode_C_seed%d.npz' % seed))
        ja = json.loads((OUT / ('episode_A_seed%d.json' % seed)).read_text())
        jc = json.loads((OUT / ('episode_C_seed%d.json' % seed)).read_text())
        ca = ja['chunks'][0]
        cc = jc['chunks'][0]
        aa = a['action'][:10]
        ac = c['action'][:10]
        entry = {'same_noise_seed': ca['meta']['noise_seed'] == cc['meta']['noise_seed'],
                 'noise_seed': ca['meta']['noise_seed'],
                 'A_grip_signs': np.sign(aa[:, 6]).astype(int).tolist(),
                 'C_grip_signs': np.sign(ac[:, 6]).astype(int).tolist(),
                 'per_dim': {}}
        for i in range(7):
            x, y = aa[:, i], ac[:, i]
            entry['per_dim'][NAMES[i]] = {
                'A_mean': float(x.mean()), 'C_mean': float(y.mean()),
                'A_std': float(x.std()), 'C_std': float(y.std()),
                'mean_abs_delta': float(np.abs(x - y).mean()),
                'corr': float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None}
        entry['trans_dir_cosine'] = float(
            np.dot(aa[:, :3].mean(0), ac[:, :3].mean(0))
            / (np.linalg.norm(aa[:, :3].mean(0)) * np.linalg.norm(ac[:, :3].mean(0)) + 1e-12))
        entry['plan_absmax'] = {'A': ca['meta']['action_absmax'], 'C': cc['meta']['action_absmax']}
        entry['pred_force_absmax'] = {'A': ca['pred_force_absmax'],
                                      'C': cc['pred_force_absmax']}
        res['seed%d' % seed] = entry
    (OUT / 'fixed_input_chunk0.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
