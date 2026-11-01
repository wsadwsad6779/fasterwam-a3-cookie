import numpy as np
import mujoco

from a3_dual_arm_sim.sim.env import A3DualArmEnv
from a3_dual_arm_sim.contracts import LEFT_JOINTS

np.set_printoptions(precision=5, suppress=True)

env = A3DualArmEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    target = env._last_applied_action.copy()
    hold = np.zeros(14)
    hold[6] = 2 * target[7] - 1
    hold[13] = 2 * target[15] - 1

    for _ in range(100):
        obs, _, terminated, truncated, info = env.step(hold)
        if terminated or truncated:
            break

    print("=== Steady-state torque diagnostic ===")
    print(
        f"{'Joint':<16} {'Error(rad)':>11} "
        f"{'Bias(Nm)':>11} {'Actuator(Nm)':>14} "
        f"{'Limit(Nm)':>11}"
    )

    for i, name in enumerate(LEFT_JOINTS):
        jid = env._joint_ids[name]
        did = env.model.jnt_dofadr[jid]
        aid = env._actuator_ids[name]

        error = target[i] - env.data.qpos[env._qpos_ids[name]]
        bias = env.data.qfrc_bias[did]
        actuator = env.data.qfrc_actuator[did]
        limit = env.model.actuator_forcerange[aid, 1]

        print(
            f"{name:<16} {error:>11.5f} "
            f"{bias:>11.3f} {actuator:>14.3f} "
            f"{limit:>11.1f}"
        )

    print("Safety reason:", info.get("safety_reason"))

finally:
    env.close()
