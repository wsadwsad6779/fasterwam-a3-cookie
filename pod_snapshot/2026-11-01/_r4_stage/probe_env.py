import sys, os, json
sys.path.insert(0, "/workspace/fasterwam_a3_bridge/bridge/_next")
import numpy as np
import env_factory
env = env_factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
obs, info = env_factory.reset(env, 22, randomize_cookies=True, randomize_boxes=False, batch_offset_m=0.0)
print("INFOKEYS", sorted(info.keys()))
print("cookies_in_target", info.get("cookies_in_target"))
imgs = {k: v for k, v in obs.items() if "images" in k}
for k, v in imgs.items(): print("IMG", k, np.asarray(v).shape, np.asarray(v).dtype)
cp = np.asarray(env.cookie_positions, dtype=np.float64)
print("NCOOKIE", cp.shape)
print("Z0", np.unique(np.round(cp[:, 2], 4)).tolist())
print("XY_SRC_MIN", cp[:, :2].min(0).tolist(), "XY_SRC_MAX", cp[:, :2].max(0).tolist())
print("STATE", obs["observation.state"].shape)
try:
    print("CFG", env_factory.DEFAULT_CONFIG)
except Exception as e:
    print("CFGERR", e)
print(__import__("os").environ.get("MUJOCO_GL"))