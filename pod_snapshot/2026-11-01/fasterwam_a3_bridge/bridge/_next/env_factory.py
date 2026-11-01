"""Environment factory for the stage-B/C experiments.

`cookie_batch.yaml` has no `position_noise_m` entry and
`A3CookieTransferEnv.__init__` builds `CookieTransferTaskConfig` with only
`cookie_count`, so the noise fields can never be reached from YAML. Everything
here passes `task_config=` explicitly so the perturbation is real.

Two traps this file exists to avoid:

1. Noise is sampled only when reset receives `randomize_cookies=True`
   (`_sample_source_cookie_pose`), and the collector resets with `False`. An
   eval that copies the collector's reset gets no noise even with the field set.
2. `randomize_boxes` defaults to False, so the target bin never moves. Any
   randomized eval must say which of the two it is randomizing.
"""
import dataclasses
import os

import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import (
    A3CookieTransferEnv,
    CookieTransferTaskConfig,
)

DEFAULT_CONFIG = os.environ.get(
    "A3_CONFIG_PATH", "/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml")


def make_env(
    config_path: str = DEFAULT_CONFIG,
    horizon: int = 6000,
    position_noise_m: float = 0.0,
    yaw_noise_rad: float = 0.0,
    render_cameras: bool = True,
) -> A3CookieTransferEnv:
    cfg = dataclasses.replace(load_config(config_path), horizon=horizon)
    task = CookieTransferTaskConfig(
        cookie_count=len(cfg.cookie_transfer.cookie_source_positions_m),
        position_noise_m=float(position_noise_m),
        yaw_noise_rad=float(yaw_noise_rad),
    )
    return A3CookieTransferEnv(
        config=cfg,
        task_config=task,
        action_mode="cartesian_delta",
        render_cameras=render_cameras,
    )


def reset(env, seed: int, randomize_cookies: bool = True, randomize_boxes: bool = False,
          batch_offset_m: float = 0.0):
    obs, info = env.reset(
        seed=seed,
        options={
            "randomize_cookies": bool(randomize_cookies),
            "randomize_boxes": bool(randomize_boxes),
        },
    )
    if batch_offset_m:
        offset = apply_batch_offset(env, batch_offset_m)
        obs = env._observation()
        info["batch_offset_xy_m"] = offset.tolist()
    return obs, info


def apply_batch_offset(env, magnitude: float, offset=None):
    """Translate the whole cookie batch by one shared xy vector.

    `position_noise_m` draws independently per cookie, and the nominal grid has
    only 0.4 mm between columns and 2.5 mm between rows, so per-cookie noise
    drives the cookies into each other (measured in `scene_audit.py`). A shared
    offset keeps every relative arrangement intact and changes only where the
    batch sits on the table, which is the question a vision policy has to
    answer: find the batch, instead of replaying calibrated coordinates.

    The draw uses `env.np_random`, so it is reproducible from the episode seed.
    """
    import mujoco

    if offset is None:
        offset = (np.zeros(2) if magnitude <= 0.0
                  else env.np_random.uniform(-magnitude, magnitude, size=2))
    offset = np.asarray(offset, dtype=np.float64)
    if not np.any(offset):
        return offset

    positions = np.asarray(env.cookie_positions, dtype=np.float64)
    quats = np.asarray(env.data.xquat[list(env._cookie_bodies)], dtype=np.float64)
    for index, (pos, quat) in enumerate(zip(positions, quats)):
        env.set_cookie_pose(index, (pos[0] + offset[0], pos[1] + offset[1], pos[2]), quat)
    # The batch rests on the bin floor; an xy translation does not change that,
    # so a forward pass is enough and no settle loop is needed.
    mujoco.mj_forward(env.model, env.data)
    if hasattr(env, "randomization_metadata"):
        env.randomization_metadata["cookie_positions"] = env.cookie_positions.tolist()
        env.randomization_metadata["batch_offset_xy_m"] = offset.tolist()
    return offset


def scene_sanity(env, max_penetration_mm: float = 0.5):
    """Minimal validity gate for an eval scene, not a research measurement.

    Only catches scenes that cannot be scored: cookies whose boxes interpenetrate
    by more than `max_penetration_mm` after reset (which is what per-cookie
    position noise does, see `scene_audit.py`), a cookie outside the source bin,
    or non-finite poses. A valid E0 or batch-offset scene passes trivially.
    """
    pos = np.asarray(env.cookie_positions, dtype=np.float64)
    half = np.asarray(env.COOKIE_HALF_SIZE, dtype=np.float64)
    dx = np.abs(pos[:, None, 0] - pos[None, :, 0])
    dy = np.abs(pos[:, None, 1] - pos[None, :, 1])
    # AABB separation is the larger of the two axis gaps: boxes overlap only
    # when both gaps are negative, exactly as in scene_audit.py.
    sep = np.maximum(dx - 2.0 * half[0], dy - 2.0 * half[1])
    np.fill_diagonal(sep, np.inf)
    i, j = np.unravel_index(int(np.argmin(sep)), sep.shape)
    penetration_mm = max(0.0, -1000.0 * float(sep[i, j]))

    center = (np.asarray(env.data.xpos[env._source_bin_body][:2], dtype=np.float64)
              if hasattr(env, "_source_bin_body")
              else np.asarray(env.SOURCE_CENTER, dtype=np.float64)[:2])
    inner = np.asarray(env.SOURCE_INNER_HALF_SIZE, dtype=np.float64)
    outside = int(np.sum(
        (np.abs(pos[:, 0] - center[0]) + half[0] > inner[0])
        | (np.abs(pos[:, 1] - center[1]) + half[1] > inner[1])
    ))
    finite = bool(np.all(np.isfinite(pos)))
    return {
        "max_penetration_mm": round(penetration_mm, 3),
        "worst_pair": [int(i), int(j)],
        "cookies_outside_bin": outside,
        "finite": finite,
        "ok": finite and outside == 0 and penetration_mm <= max_penetration_mm,
    }


def scene_geometry(config_path: str = DEFAULT_CONFIG) -> dict:
    """Fixed, calibration-time geometry. Never reads simulator state."""
    cfg = load_config(config_path)
    ct = cfg.cookie_transfer
    center = tuple(ct.target_bin_world_position_m)
    return {
        "target_center": center,
        "inner_half": (
            ct.target_bin_half_size_m[0] - ct.bin_wall_thickness_m,
            ct.target_bin_half_size_m[1] - ct.bin_wall_thickness_m,
        ),
        "wall_height": ct.target_bin_wall_height_m,
        "rim_z": center[2] + ct.target_bin_wall_height_m,
        "source_columns_x": sorted({p[0] for p in ct.cookie_source_positions_m}),
        "source_y_range": (
            min(p[1] for p in ct.cookie_source_positions_m),
            max(p[1] for p in ct.cookie_source_positions_m),
        ),
    }
