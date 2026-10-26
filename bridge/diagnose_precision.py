import mujoco
import numpy as np

from a3_dual_arm_sim.sim.env import A3DualArmEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2

np.set_printoptions(precision=8, suppress=True)

env = A3DualArmEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

def fk_left(joint_action):
    """Calculate left EEF position for a hypothetical joint command."""
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.data.qpos
    data.qvel[:] = 0.0

    for name, value in zip(env._ik._arms[0], joint_action[:7]):
        qid = env._qpos_ids[name]
        data.qpos[qid] = value

    mujoco.mj_forward(env.model, data)
    site_id = env._eef_sites[0]
    return data.site_xpos[site_id].copy()

def mm(vector):
    return np.asarray(vector) * 1000.0

try:
    obs, _ = env.reset(seed=0)

    initial = np.array(obs["observation.eef_pose"][:3], dtype=float)
    state = np.array(obs["observation.state"], dtype=float)

    action = FasterWAMA3ActionAdapterV2().convert(
        [0.05, 0, 0, 0, 0, 0, float(state[7])],
        right_gripper_opening=float(state[15]),
    )

    ideal = initial + np.array([0.0025, 0, 0])
    previous = env._last_applied_action.copy()

    raw = env._ik.convert(env.data, action, previous)
    limited = env._limit_joint_action(raw)

    raw_fk = fk_left(raw)
    limited_fk = fk_left(limited)

    print("=== Precision decomposition ===")
    print("Ideal target XYZ:", ideal)
    print("Raw IK FK XYZ:", raw_fk)
    print("Limited FK XYZ:", limited_fk)

    print("\nRaw IK target error (mm):", mm(raw_fk - ideal))
    print("Limited target error (mm):", mm(limited_fk - ideal))
    print("Rate limit joint max (rad):",
          np.max(np.abs(limited[:7] - raw[:7])))

    obs, _, terminated, truncated, info = env.step(action)

    hold = np.zeros(14, dtype=float)
    hold[6] = action[6]
    hold[13] = action[13]

    for _ in range(40):
        if terminated or truncated:
            break
        obs, _, terminated, truncated, info = env.step(hold)

    actual = np.array(obs["observation.eef_pose"][:3], dtype=float)

    print("\nActual EEF XYZ:", actual)
    print("Actual target error (mm):", mm(actual - ideal))
    print("Execution minus limited FK (mm):", mm(actual - limited_fk))
    print("Safety reason:", info.get("safety_reason"))
    print("Terminated:", terminated, "Truncated:", truncated)

finally:
    env.close()
