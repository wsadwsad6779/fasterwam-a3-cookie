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


def reset(env, seed: int, randomize_cookies: bool = True, randomize_boxes: bool = False):
    return env.reset(
        seed=seed,
        options={
            "randomize_cookies": bool(randomize_cookies),
            "randomize_boxes": bool(randomize_boxes),
        },
    )


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
