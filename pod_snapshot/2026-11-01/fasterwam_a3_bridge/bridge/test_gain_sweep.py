import numpy as np
import mujoco

from a3_dual_arm_sim.sim.env import A3DualArmEnv
from a3_dual_arm_sim.contracts import LEFT_JOINTS, RIGHT_JOINTS

np.set_printoptions(precision=6, suppress=True)


def test_gain(multiplier):
    env = A3DualArmEnv(
        action_mode="cartesian_delta",
        render_cameras=False,
    )

    try:
        obs, _ = env.reset(seed=0)

        # Change gains only in this temporary simulation.
        # Preserve existing actuator force limits.
        for name in [*LEFT_JOINTS, *RIGHT_JOINTS]:
            aid = env._actuator_ids[name]

            original_kp = float(env.model.actuator_gainprm[aid, 0])
            original_kv = -float(env.model.actuator_biasprm[aid, 2])

            new_kp = original_kp * multiplier
            new_kv = original_kv * np.sqrt(multiplier)

            env.model.actuator_gainprm[aid, 0] = new_kp
            env.model.actuator_biasprm[aid, 1] = -new_kp
            env.model.actuator_biasprm[aid, 2] = -new_kv

        target = env._last_applied_action.copy()

        # FK of the commanded initial joint configuration.
        ref = mujoco.MjData(env.model)
        ref.qpos[:] = env.data.qpos
        ref.qvel[:] = env.data.qvel

        left_qids = list(env._ik._qpos_ids[0])
        right_qids = list(env._ik._qpos_ids[1])

        ref.qpos[left_qids] = target[:7]
        ref.qpos[right_qids] = target[8:15]

        mujoco.mj_forward(env.model, ref)
        goal_xyz = ref.site_xpos[env._eef_sites[0]].copy()

        hold = np.zeros(14, dtype=float)
        hold[6] = 2 * target[7] - 1
        hold[13] = 2 * target[15] - 1

        positions = []
        terminated = truncated = False
        info = {}

        for _ in range(100):
            obs, _, terminated, truncated, info = env.step(hold)
            positions.append(
                np.asarray(
                    obs["observation.eef_pose"][:3],
                    dtype=float,
                ).copy()
            )
            if terminated or truncated:
                break

        actual_xyz = positions[-1]
        measured = env._current_joint_action()

        joint_error = measured[:7] - target[:7]
        position_error_mm = (actual_xyz - goal_xyz) * 1000

        if len(positions) >= 20:
            tail = np.asarray(positions[-20:])
            fluctuation_mm = np.max(
                np.linalg.norm(tail - tail.mean(axis=0), axis=1)
            ) * 1000
        else:
            fluctuation_mm = float("nan")

        print(f"\n=== Gain multiplier: {multiplier} ===")
        print("Max joint error (rad):", np.max(np.abs(joint_error)))
        print("Position error XYZ (mm):", position_error_mm)
        print("Position error norm (mm):",
              np.linalg.norm(position_error_mm))
        print("Last 20 steps fluctuation (mm):", fluctuation_mm)
        print("Safety reason:", info.get("safety_reason"))
        print("Terminated:", terminated, "Truncated:", truncated)

    finally:
        env.close()


for multiplier in [1.0, 1.5, 2.0]:
    test_gain(multiplier)

print("\nGain sweep completed.")
