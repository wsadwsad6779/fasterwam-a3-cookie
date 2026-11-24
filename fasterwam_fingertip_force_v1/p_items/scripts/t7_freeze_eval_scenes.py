"""P4: freeze the collection-style evaluation scenes before any checkpoint result.

80 seeds (7100-7709, 7900-7909) are reset with the same randomization options
used by the data collection. Each seed gets a deterministic scene hash; the
eval script must reproduce the hash and may not choose scenes after the fact.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np

OUT = os.environ.get('T7_SCENE_MANIFEST_OUT', '/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json')
SIM_CONFIG = os.environ.get('T7_SIM_CONFIG', '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml')
SEEDS = [s for base in range(7100, 7800, 100) for s in range(base, base + 10)]
SEEDS += list(range(7900, 7910))


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
    from a3_dual_arm_sim import A3CookieTransferEnv
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    settings = load_randomization_config(None)
    base_options = {
        'randomize_cookies': True, 'randomize_boxes': True,
        'randomize_source_bin': True, 'randomize_target_bin': True,
        'source_bin_noise_m': 0.010, 'target_bin_noise_m': 0.010,
        'target_bin_yaw_noise_rad': 0.030,
        'randomization_profile': settings['profile'],
        'randomization_settings': settings,
    }
    scenes = []
    t0 = time.time()
    for seed in SEEDS:
        env = A3CookieTransferEnv(config=SIM_CONFIG, action_mode='cartesian_delta',
                                  render_mode='rgb_array', render_cameras=False)
        u = getattr(env, 'unwrapped', env)
        opts = dict(base_options)
        opts['appearance_seed'] = seed
        env.reset(seed=seed, options=opts)
        rm = dict(getattr(u, 'randomization_metadata', {}) or {})
        cp = np.asarray(rm.get('cookie_positions', []), np.float64)
        scenes.append({
            'seed': seed,
            'hash': scene_hash(rm),
            'options': opts,
            'cookie_positions': np.round(cp, 6).tolist(),
            'source_bin': [float(x) for x in np.asarray(rm.get('source_bin_position'), np.float64)],
            'target_bin': [float(x) for x in np.asarray(rm.get('target_bin_position'), np.float64)],
            'cameras': rm.get('cameras'), 'light': rm.get('light_diffuse') or rm.get('light'),
            'profile': rm.get('profile'),
        })
        env.close()
    hashes = sorted({s['hash'] for s in scenes})
    freeze = hashlib.sha256(json.dumps(
        {'base_options': base_options, 'scenes': [{'seed': s['seed'], 'hash': s['hash']} for s in scenes]},
        sort_keys=True).encode()).hexdigest()
    manifest = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'sim_config': SIM_CONFIG,
                'sim_config_sha256': hashlib.sha256(open(SIM_CONFIG, 'rb').read()).hexdigest(),
                'purpose': 'pre-run frozen evaluation scenes; no scene selection after checkpoint results',
                'scope': 'basic profile: 80 distinct physical layouts (bins+cookies within ~2 cm); '
                         'cameras/light fixed; P3-B medium/advanced profiles are a separate data version',
                'seeds': SEEDS, 'base_options': base_options, 'scenes': scenes,
                'distinct_hashes': len(hashes), 'freeze_sha256': freeze,
                'elapsed_s': round(time.time() - t0, 1)}
    json.dump(manifest, open(OUT, 'w'), indent=1)
    print(json.dumps({'scenes': len(scenes), 'distinct_hashes': len(hashes),
                      'freeze_sha256': freeze, 'elapsed_s': manifest['elapsed_s']}, indent=1), flush=True)


if __name__ == '__main__':
    main()