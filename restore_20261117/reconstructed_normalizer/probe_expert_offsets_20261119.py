"""Probe: does the scripted expert still succeed when the cookie batch is translated?

For each (dx, dy) in OFFSETS (metres, shared batch translation inside the source bin)
this runs the expert physics-only, records the EEF path plus the batch centre the expert
actually picked, and reports cookies_in_target. No frames are stored: this is only the
go/no-go for the diversity collector.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SIM_SRC = '/workspace/eval_restore_20261117/deploy/a3_dual_arm_sim/src'
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'probe_offsets_20261119.json'
OFFSETS = [tuple(float(v) for v in s.split(','))
           for s in os.environ.get('PROBE_OFFSETS',
                                   '0,0;0.02,0;0,0.008;-0.02,0;0,-0.008').split(';') if s.strip()]
MAX_STEPS = int(os.environ.get('PROBE_STEPS', '600'))

E, factory, W, fw = D.setup_protocol()
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert


def run(i, off):
    seed = 900 + i
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=False)
    u = getattr(env, 'unwrapped', env)
    obs, _ = factory.reset(env, seed, randomize_cookies=True, randomize_boxes=False,
                           batch_offset_m=0.0)
    if any(off):
        factory.apply_batch_offset(u, 0.0, offset=np.asarray(off, dtype=np.float64))
        obs = u._observation()
    cookies0 = np.asarray(u.cookie_positions)
    expert = A3CookieBatchExpert(env=u, action_mode='joint_position')
    expert.reset(context=None)
    pick = np.asarray(expert._pick_center) if hasattr(expert, '_pick_center') else None
    pos = [np.asarray(obs['observation.eef_pose'])[:3].astype(np.float64)]
    info = {}
    for _ in range(MAX_STEPS):
        a = expert.act(obs, '')
        out = env.step(np.asarray(a, np.float64))
        obs = out[0]
        info = out[4] if len(out) > 4 else {}
        pos.append(np.asarray(obs['observation.eef_pose'])[:3].astype(np.float64))
    pos = np.asarray(pos)
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    rec = {'i': i, 'seed': seed, 'offset': list(off),
           'batch_center_first5': cookies0[:5].mean(0).round(5).tolist(),
           'expert_pick_center': None if pick is None else np.round(pick, 5).tolist(),
           'eef_start': pos[0].round(5).tolist(), 'eef_final': pos[-1].round(5).tolist(),
           'path_mm': round(path * 1000, 2),
           'cookies_in_target': int(info.get('cookies_in_target', -1)),
           'cookies_in_source': info.get('cookies_in_source'),
           'exact_fill': bool(info.get('exact_2x5_fill', False)),
           'expert_finished': bool(getattr(expert, 'finished', False)),
           'expert_phase': str(getattr(expert, 'phase', ''))}
    if hasattr(env, 'close'):
        env.close()
    print(json.dumps(rec), flush=True)
    return rec


recs = []
for i, off in enumerate(OFFSETS):
    recs.append(run(i, off))
OUT.write_text(json.dumps(recs, indent=2))
print('PROBE_OFFSETS_DONE', flush=True)