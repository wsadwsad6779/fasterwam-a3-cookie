"""CPU probe: is the teacher-replay eef frame the same as the deployment reset frame?

No rendering, no policy, no GPU. Compare the eef pose right after reset with the pose the
expert sees on its first step, and with the reconstructed corpus statistics.
"""
import json
import sys
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

E, factory, W, fw = D.setup_protocol()
SIM_SRC = '/workspace/eval_restore_20261117/deploy/a3_dual_arm_sim/src'
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert

out = {}
for seed in (29, 41):
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=False)
    reset = D.make_reset(E, factory, env, SimpleNamespace(
        cpu=True, gripper_mode='pm1', max_steps=600, replan=10, allow_invalid_scene=False))
    obs = reset(env, seed)
    p0 = np.asarray(obs['observation.eef_pose'], dtype=np.float64)
    u = getattr(env, 'unwrapped', env)
    ex = A3CookieBatchExpert(env=u, action_mode='joint_position')
    ex.reset(context=None)
    obs2 = ex.env.task.observation if hasattr(ex, 'env') and hasattr(ex.env, 'task') else obs
    rec = {'reset_eef_pose': p0.tolist()}
    try:
        a = ex.act(obs, '')
        o = env.step(np.asarray(a, np.float64))
        p1 = np.asarray(o[0]['observation.eef_pose'], dtype=np.float64)
        rec['action'] = np.asarray(a, dtype=np.float64).tolist()
        rec['eef_pose_after_1_step'] = p1.tolist()
        rec['delta_xyz'] = (p1[:3] - p0[:3]).tolist()
        rec['info'] = {k: o[4].get(k) for k in ('cookies_in_target', 'cookies_in_source')}
    except Exception as exc:
        rec['error'] = repr(exc)
    out['seed%d' % seed] = rec
    if hasattr(env, 'close'):
        env.close()

stats = json.load(open(D.BASE / 'reconstructed_normalizer/dataset_stats_14d_RECONSTRUCTED.json'))
st = stats['state']['default']
out['recon_state_mean'] = st['global_mean'][:8]
out['recon_state_std'] = st['global_std'][:8]
print(json.dumps(out, indent=2))
