import numpy as np

from a3_dual_arm_sim.sim.env import A3DualArmEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2

np.set_printoptions(precision=6, suppress=True)

env = A3DualArmEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)
    adapter = FasterWAMA3ActionAdapterV2()

    left_grip = float(obs["observation.state"][7])
    right_grip = float(obs["observation.state"][15])

    action = adapter.convert(
        [0.05, 0, 0, 0, 0, 0, left_grip],
        right_gripper_opening=right_grip,
    )

    previous = env._last_applied_action.copy()

    # 计算 IK 的原始关节目标，不推进仿真。
    ik_target = env._ik.convert(
        env.data,
        action,
        previous,
    )

    # 计算经过 A3 关节限速后的目标。
    limited_target = env._limit_joint_action(ik_target)

    ik_delta = np.r_[
        ik_target[:7] - previous[:7],
        ik_target[8:15] - previous[8:15],
    ]

    limited_delta = np.r_[
        limited_target[:7] - previous[:7],
        limited_target[8:15] - previous[8:15],
    ]

    print("=== IK diagnostic ===")
    print("A3 Cartesian X command:", action[0])
    print("Expected target X delta: 5.0 mm")
    print("Max raw IK joint delta (rad):", np.max(np.abs(ik_delta)))
    print("Max applied joint delta (rad):", np.max(np.abs(limited_delta)))
    print("Configured max joint step (rad):", env.config.max_joint_step_rad)

    if np.max(np.abs(ik_delta)) > env.config.max_joint_step_rad:
        print("Joint rate limit: TRIGGERED")
    else:
        print("Joint rate limit: NOT TRIGGERED")

    print("Raw IK left joint delta:", ik_delta[:7])
    print("Limited left joint delta:", limited_delta[:7])

finally:
    env.close()
