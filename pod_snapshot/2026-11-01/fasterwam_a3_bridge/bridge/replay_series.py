"""Per-step tilt/position series for the r26 batches."""
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

NPZ = "outputs/wrapped/run_20261026_164924/full_episode_42.npz"
d = np.load(NPZ)
acts = d["executed_14d"]
phases = d["phase"]
eefs = d["eef_xyz"]
forces = d["pad_force"]
env = A3CookieTransferEnv(
    config=load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"),
    action_mode="cartesian_delta", render_cameras=False)
env.reset(seed=42)
b = env.data.xpos[env._target_bin_body].copy()
R = env.data.xmat[env._target_bin_body].reshape(3, 3).copy()


def tilt_of(j):
    M = R.T @ env.data.xmat[env._cookie_bodies[j]].reshape(3, 3)
    return np.degrees(np.arccos(np.clip(abs(M[2, 2]), 0, 1))), M


for i, a in enumerate(acts):
    env.step(a)
    if i % 10 or i < 240 or i > 530:
        continue
    line = "step %3d ph=%-7s eef=%s f=(%.1f,%.1f)" % (
        i, ["open", "preclose", "descend", "close", "hold", "release", "done", "raw"][int(phases[i])],
        np.round(eefs[i], 4), forces[i][0], forces[i][1])
    for j in (2, 22):
        p = env.data.xpos[env._cookie_bodies[j]]
        rel = R.T @ (p - b)
        t, M = tilt_of(j)
        axis = np.round(M[:3, :3] @ np.array([0, 0, 1.0]), 3)
        line += " | c%d rel=%s tilt=%.1f zaxis=%s" % (j, np.round(rel, 3), t, axis)
    print(line, flush=True)
env.close()
