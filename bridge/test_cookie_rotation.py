import numpy as np
from scipy.spatial.transform import Rotation

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2

np.set_printoptions(precision=7, suppress=True)


def to_rotation(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)
    before = np.asarray(obs["observation.eef_pose"], dtype=float).copy()
    state = np.asarray(obs["observation.state"], dtype=float)

    # Candidate LIBERO rotation action.
    # 0.01 * 0.5 rad = 0.005 rad target rotation.
    action7 = [
        0, 0, 0,
        0, 0, 0.01,
        float(state[7]),
    ]

    action14 = FasterWAMA3ActionAdapterV2().convert(
        action7,
        right_gripper_opening=float(state[15]),
    )

    obs, _, terminated, truncated, info = env.step(action14)

    hold = np.zeros(14, dtype=float)
    hold[6] = action14[6]
    hold[13] = action14[13]

    for _ in range(40):
        if terminated or truncated:
            break
        obs, _, terminated, truncated, info = env.step(hold)

    after = np.asarray(obs["observation.eef_pose"], dtype=float)

    R_before = to_rotation(before[3:7])
    R_after = to_rotation(after[3:7])

    # World-frame and body-frame relative rotation vectors.
    rot_world = (R_after * R_before.inv()).as_rotvec()
    rot_local = (R_before.inv() * R_after).as_rotvec()

    print("=== A3 rotation validation ===")
    print("A3 action:", action14)
    print("Target rotation magnitude (rad):", 0.005)
    print("Actual world rotation vector:", rot_world)
    print("Actual local rotation vector:", rot_local)
    print("Actual angle (rad):", np.linalg.norm(rot_world))
    print("Actual angle (deg):", np.degrees(np.linalg.norm(rot_world)))
    print("Left XYZ drift (mm):", (after[:3] - before[:3]) * 1000)
    print("Right XYZ drift (mm):", (after[7:10] - before[7:10]) * 1000)
    print("Safety reason:", info.get("safety_reason"))

finally:
    env.close()
