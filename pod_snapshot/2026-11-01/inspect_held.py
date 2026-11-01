import json
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, "/workspace/fasterwam_a3_bridge/bridge")
import env_factory

PTS = [(11, 570), (11, 579), (21, 214), (21, 223)]
ROOT = Path("/workspace/fasterwam_a3_bridge/bridge/outputs/dagger_near")


def restore(env, st):
    mujoco.mj_resetData(env.model, env.data)
    d = env.data
    for key in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart"):
        getattr(d, key)[:] = st[key]
    d.time = float(st["time"])
    env._last_applied_action = np.asarray(st["last_applied_action"], dtype=np.float64).copy()
    env._step_count = int(st["step_count"])
    if hasattr(env, "_success_hold_count"):
        env._success_hold_count = int(st["success_hold_count"])
    if hasattr(env, "_source_initially_filled"):
        env._source_initially_filled = bool(st["source_initially_filled"])
    mujoco.mj_forward(env.model, env.data)


def main():
    for seed, take in PTS:
        p = ROOT / f"seed{seed}_s{take}" / "state_takeover.npz"
        if not p.exists():
            print(f"seed={seed} take={take} MISSING", flush=True)
            continue
        st = np.load(p, allow_pickle=False)
        env = env_factory.make_env(render_cameras=False)
        try:
            restore(env, st)
            pads = list(env._left_finger_geoms)
            pad_mid = env.data.geom_xpos[pads].mean(axis=0)
            eef = env.data.site_xpos[env._eef_sites[0]]
            cmd = np.asarray(env.last_applied_action)
            print(f"== seed={seed} take={take} eef={np.round(eef,4).tolist()} "
                  f"padmid={np.round(pad_mid,4).tolist()} "
                  f"cmd_g={round(float(cmd[6]),3)}", flush=True)
            for i in range(env.task_config.cookie_count):
                pos = env.data.xpos[env._cookie_bodies[i]]
                touching = any(env.privileged_left_finger_contacts(i))
                print(json.dumps({
                    "i": i,
                    "contact": bool(touching),
                    "d_mm": round(float(np.linalg.norm(pos - pad_mid)) * 1000, 1),
                    "z": round(float(pos[2]), 4),
                    "x": round(float(pos[0]), 4),
                    "y": round(float(pos[1]), 4),
                    "src": bool(env.privileged_cookie_in_source(i)),
                    "tgt": bool(env.privileged_cookie_in_target(i)),
                }), flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            env.close()


main()
