import sys
sys.path.insert(0, '/workspace/eval_restore_20261117/deploy/stage_40k/fw/src')
import torch
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json, LinearNormalizer
s = load_dataset_stats_from_json('/workspace/eval_restore_20261117/derived/dataset_stats_14d_DERIVED.json')
n = LinearNormalizer({'action': [{'key': 'default'}], 'state': [{'key': 'default'}]}, False, 'min/max', None, s)
x = torch.zeros(1, 14)
a = torch.zeros(1, 7)
r = n.forward({'state': {'default': x}, 'action': {'default': a}})
print('OK', r['state']['default'].shape)
