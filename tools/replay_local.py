"""Local replay of cloud-recorded A3 episode with live MuJoCo GUI window.

Usage (Windows, local machine):
    python replay_local.py <path_to_episode_0.npz>

The A3 sim is deterministic: same seed + same actions = same trajectory
as recorded on the cloud. No GPU needed.
"""
import sys
import time
from pathlib import Path

import numpy as np

A3_SRC = r"E:\Innovative_practice\WAM\a3_dual_arm_sim\src"
sys.path.insert(0, A3_SRC)

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

def main():
    npz_path = Path(sys.argv[1] if len(sys.argv) > 1 else "episode_0.npz")
    data = np.load(npz_path)
    acts = np.asarray(data["executed_14d"], dtype=np.float64)
    print(f"Loaded {acts.shape} actions from {npz_path}")

    env = A3CookieTransferEnv(
        action_mode="cartesian_delta",
        render_mode="human",          # opens a live viewer window
        render_cameras=False,
    )

    try:
        obs, info = env.reset(seed=11)
        for i, a in enumerate(acts):
            obs, reward, terminated, truncated, info = env.step(a)
            if i % 25 == 0:
                print(f"step {i:03d} cookies_in_target="
                      f"{info.get('cookies_in_target', 'N/A')} "
                      f"opening={float(obs['observation.state'][7]):.3f}")
            if terminated or truncated:
                print(f"Episode ended at step {i}: "
                      f"terminated={terminated} truncated={truncated}")
                break
            time.sleep(0.02)  # ~real-time playback; remove for fast-forward
    finally:
        env.close()

if __name__ == "__main__":
    main()
