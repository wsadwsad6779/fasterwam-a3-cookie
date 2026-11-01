'''Live probe of the C-2 trigger: step the bundle expert for a short horizon and print, every
10 frames, the phase and the live cookie-at-grip count, so the trigger condition can be
seen instead of inferred.  Costs ~150 simulation steps.
'''
import sys

import numpy as np

sys.path.insert(0, '/workspace/_r5_stage')
import replay_forces as R

env, obs, info = R.make_env(0.0003, 7031)
expert = R.A3SameColumnBatchExpert(env)
expert.reset()
z0 = np.asarray(env.cookie_positions, dtype=np.float64).copy()
for step in range(1, 151):
    obs, _, term, trunc, info = env.step(expert.act(obs))
    xyz = np.asarray(env.cookie_positions, dtype=np.float64)
    eef = np.asarray(obs['observation.eef_pose'][:3], dtype=np.float64)
    n = int(np.sum((np.linalg.norm(xyz[:, :2] - eef[:2], axis=1) < 0.025)
                                       & (xyz[:, 2] - z0[:, 2] > 0.02)))
    if step % 10 == 0 or n >= 2:
        f = np.asarray(obs[R.FORCE_KEY], dtype=np.float64)
        print('step %4d phase %-16s at_grip %d carry %s pad %.3f open %.3f term %s trunc %s'
              % (step, expert.phase.name, n, expert.phase.name in R.CARRY_PHASES,
                 max(f[R.L_INNER], f[R.L_OUTER]), float(obs['observation.state'][7]),
                 term, trunc), flush=True)
    if term or trunc or expert.finished or expert.failed:
        print('stop at step', step, 'failed', bool(expert.failed),
              str(getattr(expert, 'failure_reason', ''))[:80])
        break
