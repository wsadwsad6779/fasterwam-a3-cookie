"""Find the tightest 5-consecutive window (by y) in the source column."""
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

d = np.load("outputs/wrapped/run_20261026_164924/full_episode_42.npz")
acts = d["executed_14d"]
env = A3CookieTransferEnv(
    config=load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"),
    action_mode="cartesian_delta", render_cameras=False)
env.reset(seed=42)


def scan(tag):
    cps = env.cookie_positions
    b = np.asarray(env.data.xpos[env._target_bin_body])
    h = np.asarray(env.TARGET_INNER_HALF_SIZE)
    keep = (np.abs(cps[:, 0] - b[0]) > h[0] + 0.06) | (np.abs(cps[:, 1] - b[1]) > h[1] + 0.06)
    src = cps[keep] if np.any(keep) else cps
    for x0 in (0.1498, 0.2002, 0.2506):
        col = src[np.abs(src[:, 0] - x0) < 0.02]
        if len(col) < 5:
            continue
        order = col[np.argsort(col[:, 1])]
        xs = order[:, 0]
        best = min(range(len(xs) - 4), key=lambda i: xs[i:i + 5].max() - xs[i:i + 5].min())
        rng = xs[best:best + 5].max() - xs[best:best + 5].min()
        first = xs[:5].max() - xs[:5].min()
        print(f"{tag} col@{x0}: n={len(col)} best5={1000 * rng:.2f}mm at y={order[best:best + 5, 1].mean():.4f} "
              f"| lowest5={1000 * first:.2f}mm | colrange={1000 * (xs.max() - xs.min()):.2f}mm", flush=True)


scan("reset")
for i, a in enumerate(acts):
    env.step(a)
    if i in (200, 300, 360, 379, 500, 560):
        scan(f"step{i}")
env.close()
