"""Track the second batch (col1 idx 20-24) tilt through lift/carry/insert."""
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

NPZ = "outputs/wrapped/run_20261026_164924/full_episode_42.npz"
d = np.load(NPZ)
acts = d["executed_14d"]
phase = d["phase"]
env = A3CookieTransferEnv(
    config=load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"),
    action_mode="cartesian_delta", render_cameras=False)
env.reset(seed=42)
binpos = np.asarray(env.data.xpos[env._target_bin_body]).copy()
print("bin", np.round(binpos, 4), flush=True)

IDX = list(range(20, 25))
OUT = [380, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 540, 560]


def tilt(bid):
    M = env.data.xmat[bid].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(abs(M[2, 2]), 0.0, 1.0))))


for i, a in enumerate(acts):
    env.step(a)
    if i not in OUT:
        continue
    tcp = np.asarray(env.data.site_xpos[env._eef_sites[0]])
    cps = env.cookie_positions
    ts = [tilt(env._cookie_bodies[j]) for j in IDX]
    loc = cps[IDX] - binpos[None, :]
    print(f"== step {i} ph={phase[i]} tcp={np.round(tcp, 4)} "
          f"tiltmax={max(ts):.1f} tiltmin={min(ts):.1f} "
          f"tilt={[round(t, 1) for t in ts]}", flush=True)
    for k, j in enumerate(IDX):
        print(f"   c{j} loc={np.round(loc[k], 4)} tilt={ts[k]:.1f}", flush=True)
env.close()
