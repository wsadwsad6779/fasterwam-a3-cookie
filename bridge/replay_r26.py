"""Replay r26 and report per-cookie placement vs the target slots."""
import numpy as np

from a3_dual_arm_sim.config import load_config
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

NPZ = "outputs/wrapped/run_20261026_164924/full_episode_42.npz"
d = np.load(NPZ)
acts = d["executed_14d"]
phases = d.get("phase")

env = A3CookieTransferEnv(
    config=load_config("/workspace/a3_dual_arm_sim/configs/cookie_batch.yaml"),
    action_mode="cartesian_delta", render_cameras=False)
env.reset(seed=42)
b = env.data.xpos[env._target_bin_body].copy()
R = env.data.xmat[env._target_bin_body].reshape(3, 3).copy()
half = env.TARGET_INNER_HALF_SIZE
tol = env.CONTACT_CONTAINMENT_TOLERANCE_M
ch = env.COOKIE_HALF_SIZE
floor_top = env.config.cookie_transfer.target_floor_z_m + env.config.cookie_transfer.bin_wall_thickness_m / 2
print("bin", np.round(b, 4), "inner_half", np.round(half, 4),
      "wall", env.TARGET_WALL_HEIGHT, "floor_top", floor_top, flush=True)
print("cookie_half", ch, "tol", tol, flush=True)
print("slots", np.round(env.TARGET_SLOTS_LOCAL, 4).tolist(), flush=True)
src0 = env.cookie_positions.copy()


def dump_moved(tag):
    for j in range(env.task_config.cookie_count):
        p = env.data.xpos[env._cookie_bodies[j]]
        if np.linalg.norm(p - src0[j]) <= 0.02:
            continue
        rel = R.T @ (p - b)
        M = R.T @ env.data.xmat[env._cookie_bodies[j]].reshape(3, 3)
        he = np.abs(M) @ ch
        tilt = np.degrees(np.arccos(np.clip(abs(M[2, 2]), 0, 1)))
        print("  %s cookie %2d world %s local %s he %s tilt %.1f"
              % (tag, j, np.round(p, 4), np.round(rel, 4), np.round(he, 4), tilt),
              flush=True)


prev = None
for i, a in enumerate(acts):
    env.step(a)
    ph = int(phases[i]) if phases is not None else -1
    if ph == 5 and prev != 5:
        eef = np.asarray(env.data.site_xpos[env._eef_sites[0]])
        print("RELEASE START step", i, "eef", np.round(eef, 4), flush=True)
        dump_moved("pre")
    if prev == 5 and ph != 5:
        eef = np.asarray(env.data.site_xpos[env._eef_sites[0]])
        print("RELEASE END step", i, "eef", np.round(eef, 4), flush=True)
        dump_moved("post")
    prev = ph

inside = [env._cookie_inside_target(j) for j in range(env.task_config.cookie_count)]
print("inside idx", [j for j, v in enumerate(inside) if v], "count", sum(inside), flush=True)
for j in range(env.task_config.cookie_count):
    p = env.data.xpos[env._cookie_bodies[j]]
    if np.linalg.norm(p - src0[j]) <= 0.02:
        continue
    rel = R.T @ (p - b)
    M = R.T @ env.data.xmat[env._cookie_bodies[j]].reshape(3, 3)
    he = np.abs(M) @ ch
    margin = half + tol - (np.abs(rel[:2]) + he[:2])
    bottom = rel[2] - he[2]
    tilt = np.degrees(np.arccos(np.clip(abs(M[2, 2]), 0, 1)))
    print("FINAL %2d local=%s margin=%s bottom=%.4f tilt=%.1f inside=%s"
          % (j, np.round(rel, 4), np.round(margin, 4), bottom, tilt, inside[j]),
          flush=True)
occ = env._target_slot_occupancy(tuple(inside))
print("slot occupancy", occ, flush=True)
print("cookies_in_target", sum(inside), "success", env._target_fill_is_valid(tuple(inside)), flush=True)
env.close()
