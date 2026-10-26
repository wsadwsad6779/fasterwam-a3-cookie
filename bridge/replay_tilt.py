"""Source-pile tilt at reset and after the first transfer."""
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

NPZ = "outputs/wrapped/run_20261026_164924/full_episode_42.npz"
acts = np.load(NPZ)["executed_14d"]
env = A3CookieTransferEnv(
    config=load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"),
    action_mode="cartesian_delta", render_cameras=False)
env.reset(seed=42)


def report(tag):
    pos = env.cookie_positions
    print("== %s ==" % tag, flush=True)
    for col, x in enumerate((0.0994, 0.1498, 0.2002, 0.2506)):
        idx = np.flatnonzero(np.abs(pos[:, 0] - x) < 0.02)
        tilts = []
        for j in idx:
            M = env.data.xmat[env._cookie_bodies[j]].reshape(3, 3)
            tilts.append(np.degrees(np.arccos(np.clip(abs(M[2, 2]), 0, 1))))
        tilts = np.asarray(tilts)
        print("col%d x=%.4f n=%d max=%.1f mean=%.1f over15=%d"
              % (col, x, len(idx), tilts.max(), tilts.mean(), int((tilts > 15).sum())),
              flush=True)


report("reset")
for i, a in enumerate(acts):
    env.step(a)
    if i in (100, 200, 240, 260, 300, 400, 470):
        report("step %d" % i)
env.close()
