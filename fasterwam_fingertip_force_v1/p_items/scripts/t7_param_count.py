import json

import torch

BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
OUT = '/workspace/fwv1/p_items/T7_PARAM_BUDGET.json'
sd = torch.load(BASE, map_location='cpu', mmap=True, weights_only=False)
mot = sd['mot']
groups = {'video': 0, 'action': 0, 'other_mot': 0, 'proprio_encoder': 0}
for k, v in mot.items():
    n = int(v.numel()) if hasattr(v, 'numel') else 0
    if k.startswith('mixtures.video.'):
        groups['video'] += n
    elif k.startswith('mixtures.action.'):
        groups['action'] += n
    else:
        groups['other_mot'] += n
pe = sd.get('proprio_encoder') or {}
for k, v in pe.items():
    groups['proprio_encoder'] += int(v.numel()) if hasattr(v, 'numel') else 0
groups['total_mot_plus_proprio'] = groups['video'] + groups['action'] + groups['other_mot'] + groups['proprio_encoder']
groups['trainable_action_plus_proprio'] = groups['action'] + groups['proprio_encoder']
groups['adamw_fp32_two_states_gb'] = round(groups['trainable_action_plus_proprio'] * 8 / 1e9, 2)
groups['weights_trainable_bf16_gb'] = round(groups['trainable_action_plus_proprio'] * 2 / 1e9, 2)
groups['checkpoint_trainable_est_gb'] = round(groups['adamw_fp32_two_states_gb'] + groups['weights_trainable_bf16_gb'], 2)
groups['seven_ckpt_est_gb'] = round(7 * groups['checkpoint_trainable_est_gb'], 2)
json.dump(groups, open(OUT, 'w'), indent=1)
print(json.dumps(groups))