import mujoco
import numpy as np

from a3_dual_arm_sim.sim.env import A3DualArmEnv

np.set_printoptions(precision=9, suppress=True)

env = A3DualArmEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    obs_xyz = np.asarray(
        obs["observation.eef_pose"][:3], dtype=np.float64
    )

    site_id = env._eef_sites[0]

    original_xyz = env.data.site_xpos[site_id].copy()

    # Independently recompute forward kinematics.
    copy = mujoco.MjData(env.model)
    copy.qpos[:] = env.data.qpos
    copy.qvel[:] = env.data.qvel

    if env.model.na:
        copy.act[:] = env.data.act

    mujoco.mj_forward(env.model, copy)

    copied_xyz = copy.site_xpos[site_id].copy()

    print("Observation XYZ:", obs_xyz)
    print("Original MuJoCo XYZ:", original_xyz)
    print("Copied MuJoCo XYZ:", copied_xyz)

    print(
        "Observation vs original (mm):",
        (obs_xyz - original_xyz) * 1000,
    )
    print(
        "Copied vs original (mm):",
        (copied_xyz - original_xyz) * 1000,
    )

    assert np.allclose(
        copied_xyz, original_xyz, atol=1e-7
    ), "Forward kinematics consistency failed"

    print("FK consistency PASSED")

finally:
    env.close()
