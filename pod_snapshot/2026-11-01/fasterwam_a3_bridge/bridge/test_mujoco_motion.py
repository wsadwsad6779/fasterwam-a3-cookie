import numpy as np

from a3_dual_arm_sim.sim.env import A3DualArmEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2


def run_test():
    env = A3DualArmEnv(
        action_mode="cartesian_delta",
        render_cameras=False,
    )

    adapter = FasterWAMA3ActionAdapterV2()

    try:
        obs, info = env.reset(seed=0)

        pose_before = np.asarray(
            obs["observation.eef_pose"], dtype=np.float64
        ).copy()

        state_before = np.asarray(
            obs["observation.state"], dtype=np.float64
        ).copy()

        print("=== Initial state ===")
        print("Left EEF XYZ:", pose_before[:3])
        print("Right EEF XYZ:", pose_before[7:10])
        print("Left gripper:", state_before[7])
        print("Right gripper:", state_before[15])

        # Candidate LIBERO-style normalized action:
        # +0.1 along X, no rotation, keep left gripper opening.
        action7 = np.array([
            0.05, 0.0, 0.0,
            0.0, 0.0, 0.0,
            state_before[7],
        ])

        action14 = adapter.convert(
            action7,
            right_gripper_opening=float(state_before[15]),
        )

        print("\n=== Converted action ===")
        print(action14)

        # Send the displacement target ONCE.
        obs, reward, terminated, truncated, info = env.step(
            action14
        )

        # Let the robot settle while holding its joint target.
        for i in range(40):
            hold = np.zeros(14, dtype=np.float64)
            hold[6] = 2.0 * state_before[7] - 1.0
            hold[13] = 2.0 * state_before[15] - 1.0

            obs, reward, terminated, truncated, info = env.step(
                hold
            )

            if obs.get("safety_stop", False):
                print("Safety stop triggered!")
                break

            if terminated or truncated:
                print("Episode finished early.")
                break

        pose_after = np.asarray(
            obs["observation.eef_pose"], dtype=np.float64
        )

        state_after = np.asarray(
            obs["observation.state"], dtype=np.float64
        )

        left_delta = pose_after[:3] - pose_before[:3]
        right_delta = pose_after[7:10] - pose_before[7:10]

        print("\n=== Motion result ===")
        print("Left XYZ before:", pose_before[:3])
        print("Left XYZ after :", pose_after[:3])
        print("Left delta (m):", left_delta)
        print("Right delta (m):", right_delta)
        print("Left gripper after:", state_after[7])
        print("Right gripper after:", state_after[15])
        print("Safety stop:", obs.get("safety_stop", False))

        assert np.all(np.isfinite(left_delta))
        assert np.all(np.isfinite(right_delta))
        print("\nMotion diagnostic completed.")

    finally:
        env.close()


if __name__ == "__main__":
    run_test()
