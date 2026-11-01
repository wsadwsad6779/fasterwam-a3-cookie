"""Offline ground-truth snapshots for grasp metrics.

Contract: this module only logs. snapshot() copies simulator-truth cookie
positions into a list that is written to the npz after the episode. Nothing
it returns reaches policy.chunk/policy.adapted, a3_wrap.apply or
safety_wrap.apply, and the read goes straight to the env so the OracleProbe
counters stay untouched.

The snapshot is taken right after env.step, the same instant
observation.eef_pose is read, so cookie and TCP samples share a timestep.
The step index is stored with it so the alignment is explicit.
"""
import numpy as np


def initial(env):
    return np.asarray(env.cookie_positions, dtype=np.float64).copy()


def snapshot(env, step):
    return int(step), np.asarray(env.cookie_positions, dtype=np.float64).copy()
