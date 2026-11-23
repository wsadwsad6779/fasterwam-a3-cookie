import os, json, time, hashlib, traceback

import torch
from omegaconf import OmegaConf

BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
OUT = '/workspace/fwv1/p_items/T7_BASE_IDENTITY.json'

res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'official_base': BASE}
try:
    res['bytes'] = os.path.getsize(BASE)
    h = hashlib.sha256()
    with open(BASE, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    res['sha256'] = h.hexdigest()
    sd = torch.load(BASE, map_location='cpu', mmap=True, weights_only=False)
    res['top_keys'] = list(sd.keys())
    res['step'] = int(sd.get('step', -1))
    res['torch_dtype'] = str(sd.get('torch_dtype'))
    mot = sd['mot']
    res['mot_keys'] = len(mot)
    fk = [k for k in mot if 'force' in k.lower()]
    res['force_key_count'] = len(fk)
    res['force_keys_sample'] = fk[:10]
    res['action_head_shape'] = list(mot['mixtures.action.head.weight'].shape)
    res['action_encoder_shape'] = list(mot['mixtures.action.action_encoder.weight'].shape)
    vk = [k for k in mot if k.startswith('mixtures.video.') and hasattr(mot[k], 'shape') and len(mot[k].shape) == 2]
    res['video_first_2d'] = [vk[0], list(mot[vk[0]].shape)] if vk else None
    pe = sd.get('proprio_encoder')
    if isinstance(pe, dict):
        res['proprio_keys'] = list(pe.keys())[:12]
        for k, v in pe.items():
            if hasattr(v, 'shape') and len(v.shape) == 2:
                res['proprio_first_2d'] = [k, list(v.shape)]
                break
    d = OmegaConf.load(CFG + '/data/a3_2cam.yaml')
    for key, getter in [
        ('num_frames', lambda: int(d.train.num_frames)),
        ('proprio_output_dim', lambda: int(d.train.processor.proprio_output_dim)),
        ('action_output_dim', lambda: int(d.train.processor.action_output_dim)),
        ('action_video_freq_ratio', lambda: int(d.train.action_video_freq_ratio)),
        ('video_size', lambda: [int(x) for x in d.train.video_size]),
        ('shape_meta_state', lambda: list(d.train.shape_meta.state.raw_shape)),
        ('shape_meta_action', lambda: list(d.train.shape_meta.action.raw_shape)),
    ]:
        try:
            res['config_' + key] = getter()
        except Exception as ex:
            res['config_' + key] = 'ERR:' + str(ex)[:80]
    res['action_horizon'] = int(res.get('config_num_frames', 33)) - 1
    res['gripper'] = {'action_index': 6, 'absolute': True,
                      'pm1': '+1 close / -1 open; verified in JOINT_TO_7D_VERIFICATION.json'}
    pe_in = None
    if res.get('proprio_first_2d'):
        pe_in = res['proprio_first_2d'][1][1]
    res['checks'] = {
        'proprio_in_8': pe_in == 8,
        'action_out_7': res['action_head_shape'][0] == 7,
        'action_in_7': res['action_encoder_shape'][1] == 7,
        'no_force_keys': len(fk) == 0,
        'step_21700': res['step'] == 21700,
        'dtype_bf16': 'bfloat16' in res['torch_dtype'],
        'action_horizon_32': res['action_horizon'] == 32,
        'num_frames_33': res.get('config_num_frames') == 33,
    }
    res['verdict'] = 'pass' if all(res['checks'].values()) else 'fail'
except Exception:
    res['error'] = traceback.format_exc()[:2000]
    res['verdict'] = 'error'
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: res.get(k) for k in ['verdict', 'checks', 'bytes', 'sha256', 'step', 'mot_keys', 'action_head_shape', 'proprio_first_2d']}))