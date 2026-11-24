import hashlib
import json
import os
import sys

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

MAN = '/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3.json'
CFG = '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'


def scene_hash(rm):
    cp = np.asarray(rm.get('cookie_positions', []), np.float64)
    sb = np.asarray(rm.get('source_bin_position', []), np.float64)
    tb = np.asarray(rm.get('target_bin_position', []), np.float64)
    h = hashlib.sha1()
    h.update(np.round(cp, 6).astype(np.float32).tobytes())
    h.update(np.round(sb, 6).astype(np.float32).tobytes())
    h.update(np.round(tb, 6).astype(np.float32).tobytes())
    h.update(json.dumps(rm.get('cameras'), sort_keys=True, default=str).encode())
    h.update(json.dumps(rm.get('light_diffuse') or rm.get('light'), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


man = json.load(open(MAN))
scene = man['scenes'][0]
from a3_dual_arm_sim import A3CookieTransferEnv
for cfg in (None, CFG):
    env = A3CookieTransferEnv(config=cfg, action_mode='cartesian_delta',
                              render_mode='rgb_array', render_cameras=False)
    u = getattr(env, 'unwrapped', env)
    env.reset(seed=int(scene['seed']), options=dict(scene['options']))
    rm = dict(getattr(u, 'randomization_metadata', {}) or {})
    h = scene_hash(rm)
    pad = float(u.model.geom('L_finger_inner_geom').size[1])
    print(json.dumps({'config': cfg, 'pad_half_thickness': pad,
                      'scene_hash': h, 'manifest_hash': scene['hash'], 'match': h == scene['hash'],
                      'cookies': len(rm.get('cookie_positions') or []),
                      'source_bin': [float(x) for x in np.asarray(rm.get('source_bin_position'), np.float64)],
                      'target_bin': [float(x) for x in np.asarray(rm.get('target_bin_position'), np.float64)]}))
    env.close()