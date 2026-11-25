#!/usr/bin/env python3
"""B4 (DIAGNOSTIC ONLY): physical consequence of commanding grip opening 0.43 vs 0.28.
Scripted top-down approach onto one cookie in a frozen scene, then hold+lift with the two
openings; compares finger force and whether the cookie is actually lifted.
Uses sim truth to aim => this is a DIAGNOSTIC intervention, NOT a bare-model result.
Pure MuJoCo, no model weights, no training."""
import json
import os
import sys
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')
sys.path.insert(0, '/workspace')

import numpy as np

OUT = os.environ.get('B4_OUT', '/workspace/fwv1/p_items/B4_STATE_CLONE_043_028.json')
FAIL = '/workspace/fwv1/p_items/B4_FAIL.json'
SIM = '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'
MANIFEST = '/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json'
SEEDS = [int(x) for x in os.environ.get('B4_SEEDS', '7504,7508').split(',') if x.strip()]


def gcmd(opening):
    return float(np.clip(2.0 * opening - 1.0, -1.0, 1.0))


def cmd_of(trans, grip):
    return np.r_[np.asarray(trans, float), [0, 0, 0], grip, 0, 0, 0, 0, 0, 0, 1.0]


def first_cookie(u):
    rm = getattr(u, 'randomization_metadata', {}) or {}
    cp = rm.get('cookie_positions')
    if cp:
        c = cp[0]
        return np.asarray(c, float).reshape(-1)[:3]
    return None


def run_opening(env, u, scene, seed, opening, reset_state=None):
    obs, _ = env.reset(seed=seed, options=dict(scene))
    if reset_state is not None:
        for name in ('set_state', '_set_state'):
            if hasattr(u, name):
                getattr(u, name)(reset_state)
                break
        obs = getattr(u, '_last_obs', obs)
    ck = first_cookie(u)
    if ck is None:
        return {'error': 'no cookie positions in sim metadata'}
    top = ck.copy(); top[2] = ck[2] + 0.10
    # descend (open grip)
    for _ in range(160):
        cur = np.asarray(obs['observation.state'], float)[:3]
        d = top - cur
        if float(np.linalg.norm(d[:2])) < 0.01 and abs(d[2]) < 0.01:
            break
        step = np.clip(d * 1.5, -0.02, 0.02)
        obs, r, term, trunc, info = env.step(cmd_of(step, 1.0))
    g = gcmd(opening)
    forces = []
    for _ in range(25):
        obs, r, term, trunc, info = env.step(cmd_of([0, 0, 0], g))
        fp = np.asarray(obs['observation.force'], float)[12:16]
        forces.append(float(np.max(fp)))
    hold_peak = max(forces) if forces else 0.0
    ck_before = first_cookie(u)
    # lift
    for _ in range(80):
        obs, r, term, trunc, info = env.step(cmd_of([0, 0, 0.01], g))
    ck_after = first_cookie(u)
    lift = float(ck_after[2] - ck_before[2]) if (ck_after is not None and ck_before is not None) else None
    return {'opening': opening, 'grip_cmd': g, 'hold_peak_force': hold_peak,
            'cookie_lift_z': lift, 'held': bool(hold_peak > 0.3 and (lift or 0) > 0.005)}


def main():
    from a3_dual_arm_sim import A3CookieTransferEnv
    sm = json.load(open(MANIFEST))
    scene_map = {int(s['seed']): s for s in sm['scenes']}
    env = A3CookieTransferEnv(config=SIM, action_mode='cartesian_delta',
                              render_mode='rgb_array', render_cameras=False)
    u = getattr(env, 'unwrapped', env)
    can_clone = any(hasattr(u, n) for n in ('get_state', '_get_state')) and any(
        hasattr(u, n) for n in ('set_state', '_set_state'))
    results = []
    for seed in SEEDS:
        scene = scene_map[seed]['options']
        rec = {'seed': seed, 'can_clone': can_clone, 'trials': []}
        try:
            if can_clone:
                obs, _ = env.reset(seed=seed, options=dict(scene))
                st = None
                for n in ('get_state', '_get_state'):
                    if hasattr(u, n):
                        st = getattr(u, n)()
                        break
                for opening in (0.43, 0.28):
                    rec['trials'].append(run_opening(env, u, scene, seed, opening, reset_state=st))
            else:
                for opening in (0.43, 0.28):
                    rec['trials'].append(run_opening(env, u, scene, seed, opening))
        except Exception:
            rec['error'] = traceback.format_exc()[:1200]
        results.append(rec)
        print(json.dumps({'seed': seed, 'trials': rec.get('trials'), 'err': 'error' in rec}), flush=True)
    out = {'artifact': 'B4_STATE_CLONE_043_028', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'can_clone': can_clone, 'results': results,
           'diagnostic_only': True,
           'note': 'scripted top-down aim uses sim truth; this is a DIAGNOSTIC intervention, NOT a bare-model success; compares hold/lift at opening 0.43 vs 0.28'}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({'B4_DONE': True, 'can_clone': can_clone}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        json.dump({'artifact': 'B4_FAIL', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'error': traceback.format_exc()[:2000]}, open(FAIL, 'w'), indent=1)
        sys.exit(3)
