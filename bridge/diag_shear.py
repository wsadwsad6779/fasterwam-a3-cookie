"""When does the source front row get sheared, and where is the tool then?"""
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

d = np.load("outputs/wrapped/run_20261026_164924/full_episode_42.npz")
acts = d["executed_14d"]
env = A3CookieTransferEnv(
    config=load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"),
    action_mode="cartesian_delta", render_cameras=False)
env.reset(seed=42)
b = np.asarray(env.data.xpos[env._target_bin_body]).copy()


def report(i):
    cps = env.cookie_positions
    tcp = np.asarray(env.data.site_xpos[env._eef_sites[0]])
    parts = []
    for col, x0 in enumerate((0.0994, 0.1498, 0.2002, 0.2506)):
        idx = np.flatnonzero(np.abs(cps[:, 0] - x0) < 0.045)
        idx = idx[np.argsort(cps[idx, 1])][:5]
        xs = cps[idx, 0]
        yaws = []
        for j in idx:
            M = env.data.xmat[env._cookie_bodies[j]].reshape(3, 3)
            yaws.append(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        parts.append(f"c{col} sp={1000 * (xs.max() - xs.min()):.1f} "
                     f"yaw={max(abs(v) for v in yaws):.0f}")
    print(f"{i:3d} tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}) " + " | ".join(parts), flush=True)


for i, a in enumerate(acts):
    env.step(a)
    if i % 10 == 0 and i <= 300:
        report(i)
env.close()
