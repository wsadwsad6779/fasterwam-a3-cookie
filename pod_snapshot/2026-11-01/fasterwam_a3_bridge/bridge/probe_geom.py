"""Measure where the pads actually meet the cookie row (no policy needed).

For a handful of TCP heights, descend from the reset pose, then close to 0.26
and report the pad force and how far the row moved. Tells us the correct
Z_PAD for the wrapper and whether closing stops with contact or jams.
"""
import dataclasses

import numpy as np

from a3_dual_arm_sim.config import SimConfig
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv


def make_env():
    return A3CookieTransferEnv(
        config=dataclasses.replace(SimConfig(horizon=6000)),
        action_mode="cartesian_delta", render_cameras=False)


def run(zt):
    env = make_env()
    obs, info = env.reset(seed=42)
    tcp0 = np.asarray(obs["observation.eef_pose"][:3], np.float64)
    cps0 = env.cookie_positions.copy()
    cx, cy = cps0[:, 0].mean(), cps0[:, 1].mean()
    print("[geom] reset tcp=%s cookie0=%s" %
          (np.round(tcp0, 4).tolist(), np.round(cps0[0], 4).tolist()), flush=True)

    a3 = np.zeros(14)
    a3[6] = 1.0
    steps = 0
    for t in range(160):
        tcp = np.asarray(obs["observation.eef_pose"][:3], np.float64)
        a3[0] = float(np.clip(6.0 * (cx - tcp[0]) / 0.025, -0.20, 0.20))
        a3[1] = float(np.clip(6.0 * (cy - tcp[1]) / 0.025, -0.20, 0.20))
        a3[2] = float(np.clip(6.0 * (zt - tcp[2]) / 0.025, -0.08, 0.08))
        obs, reward, term, trunc, info = env.step(a3)
        steps = t + 1
        if abs(float(obs["observation.eef_pose"][2]) - zt) < 0.002:
            break

    tcp = np.asarray(obs["observation.eef_pose"][:3], np.float64)
    d = float(np.linalg.norm(env.cookie_positions - tcp[None, :], axis=1).min())
    print("[geom] zt=%.4f reached z=%.4f in %d steps dmin=%.4f open=%.3f" %
          (zt, tcp[2], steps, d, float(obs["observation.state"][7])), flush=True)

    a3[:3] = 0.0
    peak, open_end = 0.0, 1.0
    for t in range(60):
        a3[6] = 2.0 * max(0.26, 1.0 - 0.02 * (t + 1)) - 1.0
        obs, reward, term, trunc, info = env.step(a3)
        peak = max(peak, float(env._sensor("L_finger_inner_touch_sensor")[0]),
                   float(env._sensor("L_finger_outer_touch_sensor")[0]))
        open_end = float(obs["observation.state"][7])
        if term or trunc:
            break
    moved = float(np.linalg.norm(env.cookie_positions - cps0, axis=1).max())
    print("[geom] zt=%.4f close: peakF=%.2f open_end=%.3f row_moved=%.4f" %
          (zt, peak, open_end, moved), flush=True)
    env.close()


for zt in (0.90, 0.86, 0.82, 0.80, 0.785, 0.770, 0.755):
    run(zt)
