"""Where does the batch x-spread come from: source or grasp?"""
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

d = np.load("outputs/wrapped/run_20261026_164924/full_episode_42.npz")
acts = d["executed_14d"]
env = A3CookieTransferEnv(
    config=load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"),
    action_mode="cartesian_delta", render_cameras=False)
env.reset(seed=42)


def show(tag):
    cps = env.cookie_positions
    for col, x0 in enumerate((0.0994, 0.1498, 0.2002, 0.2506)):
        idx = np.flatnonzero(np.abs(cps[:, 0] - x0) < 0.02)
        if len(idx) == 0:
            continue
        xs = cps[idx, 0]
        print(f"{tag} col{col} n={len(idx)} xmin={xs.min():.4f} xmax={xs.max():.4f} "
              f"spread={1000 * (xs.max() - xs.min()):.2f}mm mean={xs.mean():.4f}", flush=True)


show("reset")
for i, a in enumerate(acts):
    env.step(a)
    if i in (300, 320, 340, 360, 380):
        show(f"step{i}")
        tcp = np.asarray(env.data.site_xpos[env._eef_sites[0]])
        print(f"   tcp={np.round(tcp, 4)}", flush=True)
env.close()
