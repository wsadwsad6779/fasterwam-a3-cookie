import numpy as np
import mujoco

from a3_dual_arm_sim.sim.env import A3DualArmEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2

np.set_printoptions(precision=9, suppress=True)

env = A3DualArmEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    previous = env._last_applied_action.copy()
    state = np.asarray(obs["observation.state"], dtype=float)

    action = FasterWAMA3ActionAdapterV2().convert(
        [0.05, 0, 0, 0, 0, 0, float(state[7])],
        right_gripper_opening=float(state[15]),
    )

    site = env._eef_sites[0]

    # Reconstruct the SAME reference qpos used by A3.convert().
    ref = mujoco.MjData(env.model)
    ref.qpos[:] = env.data.qpos
    ref.qvel[:] = env.data.qvel

    if env.model.na:
        ref.act[:] = env.data.act

    left_qids = list(env._ik._qpos_ids[0])
    right_qids = list(env._ik._qpos_ids[1])

    ref.qpos[left_qids] = previous[:7]
    ref.qpos[right_qids] = previous[8:15]

    mujoco.mj_kinematics(env.model, ref)

    start_xyz = ref.site_xpos[site].copy()
    desired_xyz = start_xyz + action[:3] * env.config.cartesian_translation_scale_m

    raw = env._ik.convert(env.data, action, previous)

    # Restore reference, then substitute raw IK joint targets.
    ref.qpos[left_qids] = raw[:7]
    ref.qpos[right_qids] = raw[8:15]
    mujoco.mj_kinematics(env.model, ref)

    solved_xyz = ref.site_xpos[site].copy()

    print("=== IK reference-aligned diagnostic ===")
    print("Measured XYZ:", env.data.site_xpos[site])
    print("IK start XYZ:", start_xyz)
    print("Desired XYZ:", desired_xyz)
    print("Raw IK FK XYZ:", solved_xyz)

    print("\nIK start minus measured (mm):",
          (start_xyz - env.data.site_xpos[site]) * 1000)
    print("Raw IK target error (mm):",
          (solved_xyz - desired_xyz) * 1000)
    print("Raw IK target error norm (mm):",
          np.linalg.norm(solved_xyz - desired_xyz) * 1000)

finally:
    env.close()
