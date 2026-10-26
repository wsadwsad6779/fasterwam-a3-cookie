import numpy as np
from a3_dual_arm_sim.sim.env import A3DualArmEnv

np.set_printoptions(precision=8, suppress=True)

env = A3DualArmEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    target = env._last_applied_action.copy()
    measured = env._current_joint_action()

    print("=== After reset ===")
    print("Left target joints:", target[:7])
    print("Left measured joints:", measured[:7])
    print("Left tracking error (rad):", measured[:7] - target[:7])
    print("Max left joint error (rad):",
          np.max(np.abs(measured[:7] - target[:7])))

    print("Right tracking error (rad):", measured[8:15] - target[8:15])

    # Zero Cartesian delta while maintaining initial gripper opening.
    hold = np.zeros(14, dtype=float)
    hold[6] = 2 * measured[7] - 1
    hold[13] = 2 * measured[15] - 1

    for _ in range(100):
        obs, _, terminated, truncated, info = env.step(hold)
        if terminated or truncated:
            break

    target = env._last_applied_action.copy()
    measured = env._current_joint_action()

    print("\n=== After 100 hold steps ===")
    print("Left target joints:", target[:7])
    print("Left measured joints:", measured[:7])
    print("Left tracking error (rad):", measured[:7] - target[:7])
    print("Max left joint error (rad):",
          np.max(np.abs(measured[:7] - target[:7])))

    print("Right tracking error (rad):", measured[8:15] - target[8:15])
    print("Safety reason:", info.get("safety_reason"))

finally:
    env.close()
