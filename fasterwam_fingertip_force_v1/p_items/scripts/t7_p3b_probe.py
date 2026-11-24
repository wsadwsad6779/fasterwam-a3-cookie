"""P3-B probe: medium/advanced layouts as a NEW, independent experiment identity.

Scope (no training, no basic-split changes):
  * 2 layouts: seed 9101 medium, seed 9102 advanced.
  * proves the layouts can be sampled (closed-loop expert completes a transfer)
    and evaluated (render_cameras + 14D cartesian step interface).
  * writes a separate P3B_SCENE_MANIFEST_V1.json; never mixed into the frozen
    basic split or the 21k training data.
"""
import hashlib
import json
import os
import sys
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np

CFG = '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'
OUT = '/workspace/fwv1/p_items/P3B_MEDADV_PROBE_V1.json'
MAN_OUT = '/workspace/fwv1/p_items/P3B_SCENE_MANIFEST_V1.json'
LAYOUTS = [{'seed': 9101, 'profile': 'medium'}, {'seed': 9102, 'profile': 'advanced'}]
MAX_STEPS = 1600


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


def reset_options(profile, settings, seed):
    p = settings['profiles'][profile]
    return {
        'randomize_cookies': True, 'randomize_boxes': True,
        'randomize_source_bin': True, 'randomize_target_bin': True,
        'source_bin_noise_m': p['source_bin_noise_m'],
        'target_bin_noise_m': p['target_bin_noise_m'],
        'target_bin_yaw_noise_rad': p['target_bin_yaw_noise_rad'],
        'randomization_profile': profile,
        'appearance_seed': seed,
        'randomization_settings': settings,
    }


def main():
    from a3_dual_arm_sim import A3CookieTransferEnv
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark
    from a3_dual_arm_sim.contracts import EpisodeContext

    settings = load_randomization_config(None)
    t0 = time.time()
    scenes = []
    results = []
    for spec in LAYOUTS:
        seed, profile = spec['seed'], spec['profile']
        opts = reset_options(profile, settings, seed)
        env = A3CookieTransferEnv(action_mode='cartesian_delta', render_mode='rgb_array',
                                  render_cameras=True)
        u = getattr(env, 'unwrapped', env)
        obs, _ = env.reset(seed=seed, options=opts)
        rm = dict(getattr(u, 'randomization_metadata', {}) or {})
        h = scene_hash(rm)
        eval_steps = 0
        eval_ok = True
        img_shapes = None
        try:
            for _ in range(5):
                a = np.zeros(14, np.float64)
                a[6] = 1.0
                a[13] = 1.0
                obs, r, term, trunc, info = env.step(a)
                eval_steps += 1
                if img_shapes is None and 'observation.images.front' in obs:
                    img_shapes = {k: list(np.asarray(obs[k]).shape) for k in obs
                                  if k.startswith('observation.images.')}
                if term or trunc:
                    break
        except Exception:
            eval_ok = False
            eval_steps = -1
        env.close()
        scenes.append({'seed': seed, 'profile': profile, 'hash': h, 'options': opts,
                       'cookie_count': int(len(rm.get('cookie_positions') or [])),
                       'source_bin': [float(x) for x in np.asarray(rm.get('source_bin_position'), np.float64)],
                       'target_bin': [float(x) for x in np.asarray(rm.get('target_bin_position'), np.float64)],
                       'cameras': rm.get('cameras'),
                       'light': rm.get('light_diffuse') or rm.get('light')})

        b = CookieBatchBenchmark(CFG, max_steps=MAX_STEPS, complete_demonstrations=True,
                                 success_hold_steps=20, profile=profile,
                                 source_column=settings['source_column'],
                                 randomization_settings=settings,
                                 placement_order='canonical')
        task = b._episode_task(1, seed)
        pol = b._episode_policy('same_column', 1, seed)
        env2 = b.create_env(render_cameras=False)
        obs, info = env2.reset(seed=seed, options=opts)
        if hasattr(pol, 'bind_env'):
            pol.bind_env(env2)
        pol.reset(EpisodeContext(seed=seed, task=task, action_mode=env2.action_mode))
        columns = sorted(set(np.asarray(env2.SOURCE_POSITIONS)[:, 0]))
        pol.expert._column_order = [columns[0]]
        steps = 0
        forces = []
        term = trunc = False
        info = {}
        c0 = time.time()
        while steps < MAX_STEPS and not (term or trunc):
            act = pol.act(obs, task)
            obs, r, term, trunc, info = env2.step(act)
            forces.append(np.asarray(obs['observation.force'], np.float64)[12:14])
            steps += 1
        env2.close()
        f = np.asarray(forces, np.float64)
        results.append({'seed': seed, 'profile': profile, 'scene_hash': h,
                        'evaluability': {'render_cameras': True, 'steps': eval_steps,
                                         'ok': eval_ok, 'image_shapes': img_shapes},
                        'collection': {'steps': steps, 'elapsed_s': round(time.time() - c0, 1),
                                       'success': bool(info.get('success')),
                                       'cookies_in_target': info.get('cookies_in_target'),
                                       'cookies_in_source': info.get('cookies_in_source'),
                                       'safety_reason': info.get('safety_reason'),
                                       'pad_force_q50': float(np.quantile(f, 0.5)) if f.size else None,
                                       'pad_force_q99': float(np.quantile(f, 0.99)) if f.size else None}})
    base_options = {'kind': 'p3b_medadv_probe', 'profiles': ['medium', 'advanced'],
                    'note': 'independent experiment identity; not mixed with the frozen basic split'}
    freeze = hashlib.sha256(json.dumps(
        {'base_options': base_options,
         'scenes': [{'seed': s['seed'], 'hash': s['hash']} for s in scenes]},
        sort_keys=True).encode()).hexdigest()
    manifest = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'experiment_id': 'P3B_MEDADV_PROBE_V1', 'dataset_version': 'p3b_medadv_probe_v1',
                'scope': 'medium/advanced layout feasibility only; not training data; not the basic split',
                'base_options': base_options, 'scenes': scenes,
                'distinct_hashes': len({s['hash'] for s in scenes}), 'freeze_sha256': freeze}
    json.dump(manifest, open(MAN_OUT, 'w'), indent=1)
    out = {'worker': 'supervisor', 'ts': manifest['ts'],
           'experiment_id': 'P3B_MEDADV_PROBE_V1', 'dataset_version': 'p3b_medadv_probe_v1',
           'base_split': 'DATASET_845_SPLIT_MANIFEST_V2 (untouched)',
           'not_mixed_with_basic': True, 'training_allowed': False,
           'scene_manifest': MAN_OUT, 'freeze_sha256': freeze,
           'layouts': results, 'elapsed_s': round(time.time() - t0, 1)}
    out['verdict'] = ('feasible' if all(
        r['evaluability']['ok'] and r['evaluability']['steps'] == 5
        and (r['collection']['success'] or (r['collection']['cookies_in_target'] or 0) >= 1)
        for r in results) else 'partial')
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({'verdict': out['verdict'], 'freeze_sha256': freeze,
                      'layouts': [{'seed': r['seed'], 'profile': r['profile'],
                                   'eval_ok': r['evaluability']['ok'],
                                   'in_target': r['collection']['cookies_in_target'],
                                   'success': r['collection']['success'],
                                   'steps': r['collection']['steps']} for r in results]}, indent=1))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('TOPFAIL', flush=True)
        print(traceback.format_exc()[:2500], flush=True)