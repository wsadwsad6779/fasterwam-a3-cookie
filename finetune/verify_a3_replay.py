import dataclasses, json, sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/workspace/fasterwam_a3_bridge")
sys.path.insert(0, "/workspace/a3_dual_arm_sim/src")
from action_adapter_v3 import FasterWAMA3ActionAdapterV3
from a3_dual_arm_sim.config import SimConfig
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

OUT = "/workspace/a3_data/a3_fw"
ROOT = "/workspace/a3_data/a3-front-close-left-100"
arr = np.load(OUT + "/arrays.npz")
act, prop, opening = arr["action"], arr["proprio"], arr["opening"]
starts, lens, ids = arr["ep_start"], arr["ep_len"], arr["ep_ids"]
meta = {int(json.loads(l)["episode_index"]): json.loads(l)
        for l in open(ROOT + "/a3_episode_metadata.jsonl")}
adapter = FasterWAMA3ActionAdapterV3()

def make_env():
    return A3CookieTransferEnv(config=dataclasses.replace(SimConfig(horizon=6000)),
                               action_mode="cartesian_delta", render_cameras=False)

def rollout(a_fn, seed, n):
    env = make_env()
    obs, info = env.reset(seed=seed)
    p0 = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64).copy()
    ps, os_, thrown = [], [], 0
    for t in range(n):
        try:
            a14 = a_fn(t, obs)
        except ValueError:
            thrown += 1
            break
        obs, r, term, trunc, info = env.step(a14)
        ps.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64))
        os_.append(float(obs["observation.state"][7]))
        if term or trunc:
            break
    env.close()
    return p0, np.asarray(ps), np.asarray(os_), thrown

for ep in (0, 1, 2):
    i = int(np.where(ids == ep)[0][0])
    s, L = int(starts[i]), int(lens[i])
    seed = int(meta[ep]["seed"])
    demo_p = prop[s:s + L, :3]
    demo_aa = prop[s:s + L, 3:6]
    demo_op = opening[s:s + L]

    def a_demo(t, obs, s=s):
        R = Rotation.from_rotvec(prop[s + t, 3:6]).as_matrix()
        return adapter.convert(act[s + t], R)

    p0, ps, ops, thrown = rollout(a_demo, seed, L - 1)
    ref = demo_p[1:1 + len(ps)]
    off = p0 - demo_p[0]
    rel = (ps - ps[0]) - (ref - ref[0])
    z0, zs, zops, _ = rollout(lambda t, o: np.r_[np.zeros(6), 1.0, np.zeros(7)], seed, len(ps))
    zref = demo_p[1:1 + len(zs)]
    zrel = (zs - zs[0]) - (zref - zref[0])

    print(f"[ep{ep}] seed={seed} L={L} steps={len(ps)} adapter_raise={thrown}")
    print(f"   reset offset vs demo (m): {np.round(off, 4).tolist()}")
    print(f"   pos rel-RMSE: demo-act {np.sqrt((rel ** 2).sum(1).mean()):.4f} m | zero-action {np.sqrt((zrel ** 2).sum(1).mean()):.4f} m")
    print(f"   pos final drift: demo-act {np.linalg.norm(rel[-1]):.4f} m | zero-action {np.linalg.norm(zrel[-1]):.4f} m")
    print(f"   opening MAE: {np.abs(ops - demo_op[1:1 + len(ops)]).mean():.4f} | demo range {demo_op.min():.3f}-{demo_op.max():.3f}")
