"""P4: do the eval seeds actually produce different physical layouts?

Resets the sim for 7100-7709 and 7900-7909 twice:
  A) default eval reset (what the current chain uses)
  B) full collection-style reset with randomize_boxes and the recorded noise
Records cookie positions, bins, camera/light metadata and a scene hash.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np

OUT = '/workspace/fwv1/p_items/EVAL_SEED_SCENE_AUDIT.json'
SEEDS = [s for base in range(7100, 7800, 100) for s in range(base, base + 10)]
SEEDS += list(range(7900, 7910))


def scene_of(u):
    rm = dict(getattr(u, 'randomization_metadata', {}) or {})
    cp = np.asarray(rm.get('cookie_positions', []), np.float64)
    sb = np.asarray(rm.get('source_bin_position', []), np.float64)
    tb = np.asarray(rm.get('target_bin_position', []), np.float64)
    cam = rm.get('cameras')
    light = rm.get('light_diffuse') or rm.get('light')
    h = hashlib.sha1()
    h.update(np.round(cp, 6).astype(np.float32).tobytes())
    h.update(np.round(sb, 6).astype(np.float32).tobytes())
    h.update(np.round(tb, 6).astype(np.float32).tobytes())
    h.update(json.dumps(cam, sort_keys=True, default=str).encode())
    h.update(json.dumps(light, sort_keys=True, default=str).encode())
    return {'cookie_positions': cp.tolist(), 'source_bin': sb.tolist(), 'target_bin': tb.tolist(),
            'cameras': cam, 'light': light, 'hash': h.hexdigest()[:16],
            'profile': rm.get('profile')}


def summarize(items):
    hashes = {x['hash'] for x in items}
    cams = {json.dumps(x['cameras'], sort_keys=True, default=str) for x in items}
    lights = {json.dumps(x['light'], sort_keys=True, default=str) for x in items}
    bins = {(round(x['source_bin'][0], 6), round(x['source_bin'][1], 6),
             round(x['target_bin'][0], 6), round(x['target_bin'][1], 6)) for x in items}
    cps = [np.asarray(x['cookie_positions'], np.float64) for x in items if x['cookie_positions']]
    pair_max = None
    pair_mean = None
    if len(cps) > 1 and cps[0].shape == cps[1].shape:
        d = [float(np.max(np.abs(cps[i] - cps[j]))) for i in range(len(cps)) for j in range(i + 1, len(cps))]
        pair_max = max(d)
        pair_mean = float(np.mean(d))
    return {'n': len(items), 'distinct_hashes': len(hashes), 'distinct_cameras': len(cams),
            'distinct_lights': len(lights), 'distinct_bins': len(bins),
            'cookie_pairwise_max_m': pair_max, 'cookie_pairwise_mean_m': pair_mean}


def main():
    from a3_dual_arm_sim import A3CookieTransferEnv
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    settings = load_randomization_config(None)
    default_items = []
    full_items = []
    t0 = time.time()
    for seed in SEEDS:
        env = A3CookieTransferEnv(action_mode='cartesian_delta', render_mode='rgb_array',
                                  render_cameras=False)
        u = getattr(env, 'unwrapped', env)
        env.reset(seed=seed)
        d = scene_of(u)
        d['seed'] = seed
        default_items.append(d)
        env.close()
        env = A3CookieTransferEnv(action_mode='cartesian_delta', render_mode='rgb_array',
                                  render_cameras=False)
        u = getattr(env, 'unwrapped', env)
        ro = {'randomize_cookies': True, 'randomize_boxes': True,
              'randomize_source_bin': True, 'randomize_target_bin': True,
              'source_bin_noise_m': 0.010, 'target_bin_noise_m': 0.010,
              'target_bin_yaw_noise_rad': 0.030, 'randomization_profile': settings['profile'],
              'appearance_seed': seed, 'randomization_settings': settings}
        env.reset(seed=seed, options=ro)
        f = scene_of(u)
        f['seed'] = seed
        full_items.append(f)
        env.close()
    result = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'seeds': {'count': len(SEEDS), 'first': SEEDS[0], 'last': SEEDS[-1]},
              'default_eval_reset': summarize(default_items),
              'full_randomized_reset': summarize(full_items),
              'default_first5': [{k: x[k] for k in ('seed', 'hash', 'source_bin', 'target_bin')} for x in default_items[:5]],
              'full_first5': [{k: x[k] for k in ('seed', 'hash', 'source_bin', 'target_bin')} for x in full_items[:5]],
              'elapsed_s': round(time.time() - t0, 1),
              'verdict': {
                  'default_reset_same_layout': summarize(default_items)['distinct_bins'] == 1 and
                                               summarize(default_items)['distinct_cameras'] == 1,
                  'default_cookie_jitter_m': summarize(default_items)['cookie_pairwise_max_m'],
                  'full_reset_distinct_bins': summarize(full_items)['distinct_bins'],
                  'full_reset_distinct_hashes': summarize(full_items)['distinct_hashes'],
                  'note': 'the current chain eval uses the default reset, which leaves bins/cameras fixed; '
                          'the full reset varies bins/cookies within the 1 cm collection envelope. '
                          'Freeze the full-reset seeds + hashes before the next run.'}}
    json.dump(result, open(OUT, 'w'), indent=1)
    print(json.dumps(result['verdict'], indent=1), flush=True)
    print(json.dumps({'default': result['default_eval_reset'], 'full': result['full_randomized_reset']}, indent=1)[:3000], flush=True)


if __name__ == '__main__':
    main()