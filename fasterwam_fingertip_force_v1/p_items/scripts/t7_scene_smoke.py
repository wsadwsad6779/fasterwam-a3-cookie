"""Prove the frozen eval scene manifest reproduces 80 distinct layouts and that
the default reset (no manifest options) is a single scene.

This is a training-free check of the eval precondition: every eval seed must map
to its frozen scene, with no silent default reset.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np

MAN = os.environ.get('T7_SCENE_MANIFEST', '/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json')
OUT = '/workspace/fwv1/p_items/T7_SCENE_SMOKE.json'


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


def main():
    t0 = time.time()
    m = json.load(open(MAN))
    canonical = hashlib.sha256(json.dumps(
        {'base_options': m['base_options'],
         'scenes': [{'seed': s['seed'], 'hash': s['hash']} for s in m['scenes']]},
        sort_keys=True).encode()).hexdigest()
    from a3_dual_arm_sim import A3CookieTransferEnv
    rows = []
    hashes = []
    for s in m['scenes']:
        env = A3CookieTransferEnv(config=os.environ.get('T7_SIM_CONFIG', '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'),
                                  action_mode='cartesian_delta', render_mode='rgb_array', render_cameras=False)
        u = getattr(env, 'unwrapped', env)
        env.reset(seed=int(s['seed']), options=dict(s['options']))
        rm = dict(getattr(u, 'randomization_metadata', {}) or {})
        h = scene_hash(rm)
        hashes.append(h)
        rows.append({'seed': int(s['seed']), 'expected': s['hash'], 'actual': h,
                     'match': h == s['hash'], 'profile': s.get('profile')})
        env.close()
    ctrl = []
    for seed in (int(m['scenes'][0]['seed']), int(m['scenes'][1]['seed'])):
        env = A3CookieTransferEnv(config=os.environ.get('T7_SIM_CONFIG', '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'),
                                  action_mode='cartesian_delta', render_mode='rgb_array', render_cameras=False)
        u = getattr(env, 'unwrapped', env)
        env.reset(seed=seed)
        rm = dict(getattr(u, 'randomization_metadata', {}) or {})
        ctrl.append(scene_hash(rm))
        env.close()
    out = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'manifest': MAN, 'manifest_file_sha256': hashlib.sha256(open(MAN, 'rb').read()).hexdigest(),
           'freeze_sha256': m.get('freeze_sha256'), 'freeze_recomputed': canonical,
           'freeze_match': canonical == m.get('freeze_sha256'),
           'scenes': len(rows), 'matched': int(sum(r['match'] for r in rows)),
           'distinct_hashes': int(len(set(hashes))),
           'default_reset_control': {'hashes': ctrl, 'distinct': len(set(ctrl))},
           'sample': rows[:5], 'mismatches': [r for r in rows if not r['match']][:5],
           'elapsed_s': round(time.time() - t0, 1)}
    out['verdict'] = ('pass' if (out['freeze_match'] and out['matched'] == out['scenes']
                                 and out['distinct_hashes'] == out['scenes']
                                 and out['default_reset_control']['distinct'] == 1)
                      else 'fail')
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ['scenes', 'matched', 'distinct_hashes',
                                          'default_reset_control', 'verdict', 'elapsed_s']}, indent=1))


if __name__ == '__main__':
    main()