import mujoco
import numpy as np

from a3_dual_arm_sim.sim.env import A3DualArmEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2

np.set_printoptions(precision=9, suppress=True)

env = A3DualArmEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)
    site = env._eef_sites[0]

    # Make a fresh reference state and recompute its FK.
    reference = mujoco.MjData(env.model)
    reference.qpos[:] = env.data.qpos
    reference.qvel[:] = env.data.qvel
    if env.model.na:
        reference.act[:] = env.data.act
    mujoco.mj_forward(env.model, reference)

    baseline = reference.site_xpos[site].copy()

    # Independently recompute FK from the same state.
    duplicate = mujoco.MjData(env.model)
    duplicate.qpos[:] = reference.qpos
    duplicate.qvel[:] = reference.qvel
    if env.model.na:
        duplicate.act[:] = reference.act
    mujoco.mj_forward(env.model, duplicate)

    print("=== FK baseline consistency ===")
    print("Reference XYZ:", baseline)
    print("Duplicate XYZ:", duplicate.site_xpos[site])
    print(
        "Duplicate difference (mm):",
        (duplicate.site_xpos[site] - baseline) * 1000,
    )

    state = np.asarray(obs["observation.state"], dtype=float)
    action = FasterWAMA3ActionAdapterV2().convert(
        [0.05, 0, 0, 0, 0, 0, float(state[7])],
        right_gripper_opening=float(state[15]),
    )

    previous = env._last_applied_action.copy()
    raw = env._ik.convert(env.data, action, previous)

    target = baseline + np.array([0.0025, 0, 0])

    # Apply raw IK joint target to an independent FK state.
    for name, value in zip(env._ik._arms[0], raw[:7]):
        qpos_index = env._qpos_ids[name]
        duplicate.qpos[qpos_index] = value

    mujoco.mj_forward(env.model, duplicate)
    raw_xyz = duplicate.site_xpos[site].copy()

    print("\n=== IK target check ===")
    print("Baseline XYZ:", baseline)
    print("Desired XYZ:", target)
    print("Raw IK FK XYZ:", raw_xyz)
    print("Raw IK error (mm):", (raw_xyz - target) * 1000)
    print("Raw IK joint delta:", raw[:7] - previous[:7])

finally:
    env.close()
