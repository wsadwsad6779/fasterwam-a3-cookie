import numpy as np
from scipy.spatial.transform import Rotation

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v3 import FasterWAMA3ActionAdapterV3

np.set_printoptions(precision=8, suppress=True)


def rotation_from_wxyz(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    before = np.asarray(
        obs["observation.eef_pose"], dtype=float
    ).copy()

    state = np.asarray(
        obs["observation.state"], dtype=float
    )

    R_before = rotation_from_wxyz(before[3:7])
    R_matrix = R_before.as_matrix()

    # LIBERO world Z rotation: 0.01 * 0.5 = 0.005 rad.
    action7 = np.array([
        0, 0, 0,
        0, 0, 0.01,
        -1.0,
    ])

    adapter = FasterWAMA3ActionAdapterV3()

    action14 = adapter.convert(
        action7,
        left_eef_rotation=R_matrix,
        right_gripper_opening=float(state[15]),
    )

    print("=== V3 world-frame rotation test ===")
    print("Requested world rotvec:", [0, 0, 0.005])
    print("Converted A3 action:", action14)

    obs, _, terminated, truncated, info = env.step(action14)

    # Hold grippers and zero Cartesian deltas while robot settles.
    hold = np.zeros(14)
    hold[6] = action14[6]
    hold[13] = action14[13]

    for _ in range(40):
        if terminated or truncated:
            break
        obs, _, terminated, truncated, info = env.step(hold)

    after = np.asarray(
        obs["observation.eef_pose"], dtype=float
    )

    R_after = rotation_from_wxyz(after[3:7])

    actual_world = (
        R_after * R_before.inv()
    ).as_rotvec()

    target_world = np.array([0, 0, 0.005])

    print("Actual world rotvec:", actual_world)
    print("Rotation error vector:", actual_world - target_world)
    print(
        "Rotation error magnitude (rad):",
        np.linalg.norm(actual_world - target_world),
    )
    print(
        "Left XYZ drift (mm):",
        (after[:3] - before[:3]) * 1000,
    )
    print(
        "Right XYZ drift (mm):",
        (after[7:10] - before[7:10]) * 1000,
    )
    print("Safety reason:", info.get("safety_reason"))

finally:
    env.close()
