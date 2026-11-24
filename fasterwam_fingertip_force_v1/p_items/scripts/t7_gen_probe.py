"""Probe: run the current-sim same_column expert for one fresh seed and
measure the pad force scale. This decides whether the 15-19x gap is an
open-loop replay artifact (expert regulates grip force to ~1.3 N) or a real
dataset-vs-sim scale difference.
"""
import json
import os
import sys
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np

CFG = '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'
OUT = '/workspace/fwv1/p_items/GEN_PROBE.json'
SEED = 90000
MAX_STEPS = 1600


def stats(a):
    a = np.asarray(a, np.float64)
    if a.size == 0:
        return None
    return {'n': int(a.size), 'mean': float(a.mean()), 'q50': float(np.quantile(a, 0.5)),
            'q99': float(np.quantile(a, 0.99)), 'max': float(a.max()),
            'contact_frac': float((np.abs(a) > 0.05).mean())}


def main():
    from a3_dual_arm_sim.workflows.benchmark import CookieBatchBenchmark
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    from a3_dual_arm_sim.contracts import EpisodeContext

    settings = load_randomization_config(None)
    b = CookieBatchBenchmark(CFG, max_steps=MAX_STEPS, complete_demonstrations=True,
                             success_hold_steps=20, profile=settings['profile'],
                             source_column=settings['source_column'],
                             randomization_settings=settings, placement_order='canonical')
    task = b._episode_task(1, SEED)
    pol = b._episode_policy('same_column', 1, SEED)
    env = b.create_env(render_cameras=False)
    ro = {
        'randomize_cookies': b.randomize_cookies,
        'randomize_boxes': b.randomize_boxes,
        'randomize_source_bin': b.randomize_boxes,
        'randomize_target_bin': b.randomize_boxes,
        'source_bin_noise_m': b.source_bin_noise_m,
        'target_bin_noise_m': b.target_bin_noise_m,
        'target_bin_yaw_noise_rad': b.target_bin_yaw_noise_rad,
        'randomization_profile': b.profile or 'basic',
        'appearance_seed': SEED,
        'randomization_settings': b.randomization_settings,
    }
    obs, info = env.reset(seed=SEED, options=ro)
    scene = dict(getattr(env.unwrapped, 'randomization_metadata', {}) or {})
    if hasattr(pol, 'bind_env'):
        pol.bind_env(env)
    pol.reset(EpisodeContext(seed=SEED, task=task, action_mode=env.action_mode))
    columns = sorted(set(np.asarray(env.SOURCE_POSITIONS)[:, 0]))
    pol.expert._column_order = [columns[0]]

    t0 = time.time()
    forces = []
    steps = 0
    term = trunc = False
    info = {}
    while steps < MAX_STEPS and not (term or trunc):
        act = pol.act(obs, task)
        obs, r, term, trunc, info = env.step(act)
        forces.append(np.asarray(obs['observation.force'], np.float64)[12:16])
        steps += 1
        if steps % 100 == 0:
            print(json.dumps({'step': steps, 'in_target': info.get('cookies_in_target'),
                              'L12_max': float(np.max(forces[-100:], axis=0)[0])}), flush=True)
    forces = np.asarray(forces, np.float64)
    result = {'worker': 'supervisor', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'seed': SEED, 'steps': steps, 'elapsed_s': round(time.time() - t0, 1),
              'terminated': bool(term), 'truncated': bool(trunc),
              'success': bool(info.get('success')), 'cookies_in_target': info.get('cookies_in_target'),
              'safety_reason': info.get('safety_reason'),
              'force': {'ch12': stats(forces[:, 0]), 'ch13': stats(forces[:, 1]),
                        'ch14': stats(forces[:, 2]), 'ch15': stats(forces[:, 3])},
              'scene': {'cookie_positions_head': (scene.get('cookie_positions') or [])[:2],
                        'source_bin_position': scene.get('source_bin_position'),
                        'target_bin_position': scene.get('target_bin_position')},
              'note': 'current-sim closed-loop expert; compare ch12/13 q99 to dataset 1.46/1.47 N'}
    json.dump(result, open(OUT, 'w'), indent=1)
    print(json.dumps(result, indent=1)[:4000], flush=True)
    env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('TOPFAIL', flush=True)
        print(traceback.format_exc()[:2500], flush=True)