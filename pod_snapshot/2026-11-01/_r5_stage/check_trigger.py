'''Offline check of the C-2 trigger: replay the recorded bundle trajectory through the same
cookie-at-grip predicate the live driver uses, and say at which frame the trigger would
fire.  Uses the already-collected replay_bundle_dist npz, so it costs no simulation steps.
'''
import sys
import numpy as np

sys.path.insert(0, '/workspace/_r5_stage')
import replay_forces as R

HOLD, LIFT = 0.025, 0.02


def at_grip(xyz, z0, eef):
    rise = xyz[:, :, 2] - z0[:, 2]
    d = np.linalg.norm(xyz[:, :, :2] - eef[:, None, :2], axis=2)
    return ((d < HOLD) & (rise > LIFT)).sum(axis=1)


for seed in (7031, 7032, 7035):
    z = np.load('/workspace/a3_eval/replay_bundle_dist/expert_bundle_%d.npz' % seed)
    xyz, eef, z0 = z['cookie_xyz'], z['eef_xyz'], z['cookie_xyz0']
    phase = [R.PHASE_NAMES[int(i)] for i in z['phase']]
    g = at_grip(xyz, z0, eef)
    carry = np.isin(np.asarray(phase), list(R.CARRY_PHASES))
    fire = np.flatnonzero(carry & (g >= 2))
    print('seed', seed, 'frames', len(phase), 'max_at_grip', int(g.max()),
          'carry_frames', int(carry.sum()), 'first_fire_frame',
          int(fire[0]) if len(fire) else None,
          'phase_at_fire', phase[int(fire[0])] if len(fire) else '-',
          'at_grip_in_carry', sorted(set(g[carry].tolist())))
