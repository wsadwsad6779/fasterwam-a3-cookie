import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/a3_dual_arm_sim/src")
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

npz = Path(sys.argv[1])
seed = int(sys.argv[2])
d = np.load(npz)
acts = d["executed_14d"]
print(f"npz={npz.name} seed={seed} actions_7d={d['actions_7d'].shape} executed_14d={acts.shape}", flush=True)

env = A3CookieTransferEnv(action_mode="cartesian_delta", render_cameras=False)
obs, info = env.reset(seed=seed)
c0 = env.cookie_positions.copy()
eef0 = np.asarray(obs["observation.eef_pose"], dtype=float)[:3].copy()
eef = eef0
src0 = info.get("cookies_in_source")
print("cookies_in_source at reset:", src0, flush=True)

min_dist, min_step = np.inf, -1
max_force = 0.0
max_lifted = 0
events = []
prev_cmd = None
for i, a in enumerate(acts):
    obs, reward, terminated, truncated, info = env.step(a)
    eef = np.asarray(obs["observation.eef_pose"], dtype=float)[:3]
    opening = float(obs["observation.state"][7])
    cps = env.cookie_positions
    dists = np.linalg.norm(cps - eef[None, :], axis=1)
    j = int(np.argmin(dists))
    d3 = float(dists[j])
    dxy = float(np.linalg.norm((cps[j] - eef)[:2]))
    if d3 < min_dist:
        min_dist, min_step = d3, i
    f = np.atleast_1d(np.asarray(obs.get("observation.force", [0.0]), dtype=float))
    max_force = max(max_force, float(np.max(np.abs(f))))
    max_lifted = max(max_lifted, int(np.sum(cps[:, 2] - c0[:, 2] > 0.02)))
    cmd = round(float(a[6]), 3)
    if prev_cmd is None or cmd != prev_cmd:
        events.append((i, cmd, round(opening, 3), round(dxy, 4), round(d3, 4), j,
                       info.get("cookies_in_source"), info.get("cookies_in_target")))
    prev_cmd = cmd
    if i % 50 == 0:
        print(f"step {i:03d} eef={np.round(eef,3)} open={opening:.3f} cookie#{j} "
              f"dxy={dxy:.4f} d3d={d3:.4f} src={info.get('cookies_in_source')} "
              f"tgt={info.get('cookies_in_target')}", flush=True)
    if terminated or truncated:
        print("ended early at step", i, "terminated=", terminated, "truncated=", truncated, flush=True)
        break

print("eef start/end:", np.round(eef0, 3), np.round(eef, 3),
      "displacement:", round(float(np.linalg.norm(eef - eef0)), 3), flush=True)
print("min eef-cookie 3d dist:", round(min_dist, 4), "at step", min_step, flush=True)
print("cookies_in_source:", src0, "->", info.get("cookies_in_source"),
      "| target:", info.get("cookies_in_target"), "| success:", info.get("success"), flush=True)
print("max |force| (obs):", round(max_force, 3),
      "| env peak:", getattr(env, "_max_recorded_gripper_force", None),
      "| max cookies lifted >2cm:", max_lifted, flush=True)
print("gripper cmd changes (step, cmd, opening, dxy, d3d, cookie_idx, src, tgt):", flush=True)
for e in events:
    print("  ", e, flush=True)
env.close()
