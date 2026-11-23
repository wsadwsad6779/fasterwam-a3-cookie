import glob
import hashlib
import json
import os

import torch

RUN = '/workspace/fwv1/train_runs/t7_21k'
OUT = '/workspace/fwv1/p_items/CHECKPOINT_VERIFY_LATEST.json'
files = sorted(glob.glob(RUN + '/*.pt'))
res = {'files': [], 'checks': {}}
for fp in files:
    res['files'].append({'path': fp, 'bytes': os.path.getsize(fp)})
latest = RUN + '/latest.pt'
if os.path.exists(latest):
    h = hashlib.sha256()
    with open(latest, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    ck = torch.load(latest, map_location='cpu', weights_only=False)
    res['latest'] = {'bytes': os.path.getsize(latest), 'sha256': h.hexdigest(),
                     'step': int(ck.get('step', -1)),
                     'keys': sorted(ck.keys()),
                     'action_expert_tensors': len(ck.get('action_expert', {})),
                     'proprio_tensors': len(ck.get('proprio', {})),
                     'film_tensors': len(ck.get('film', {})),
                     'optimizer_state_groups': len(ck.get('optimizer', {}).get('state', {})),
                     'scheduler_present': 'scheduler' in ck,
                     'rng_present': 'rng_torch' in ck and 'rng_cuda' in ck,
                     'base_sha256': ck.get('base_sha256')}
    res['checks'] = {
        'step_positive': res['latest']['step'] > 0,
        'action_expert_loaded': res['latest']['action_expert_tensors'] > 0,
        'optimizer_nonempty': res['latest']['optimizer_state_groups'] > 0,
        'scheduler_present': res['latest']['scheduler_present'],
        'rng_present': res['latest']['rng_present'],
        'base_sha_matches': res['latest']['base_sha256'] == '01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d',
    }
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({'files': res['files'], 'latest': res.get('latest'), 'ok': all(res.get('checks', {}).values())}))