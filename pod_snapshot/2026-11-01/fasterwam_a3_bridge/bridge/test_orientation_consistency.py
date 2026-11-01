import numpy as np
from scipy.spatial.transform import Rotation

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from observation_adapter import quat_wxyz_to_axis_angle

env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    pose = np.asarray(
        obs["observation.eef_pose"], dtype=np.float64
    )

    quat_wxyz = pose[3:7]
    axis_angle = quat_wxyz_to_axis_angle(quat_wxyz)

    # SciPy expects xyzw; A3 provides wxyz.
    quat_xyzw = np.r_[
        quat_wxyz[1:],
        quat_wxyz[0],
    ]

    original_rotation = Rotation.from_quat(quat_xyzw)
    restored_rotation = Rotation.from_rotvec(axis_angle)

    difference = original_rotation.inv() * restored_rotation
    error_rad = difference.magnitude()

    print("A3 quaternion wxyz:", quat_wxyz)
    print("Converted axis-angle:", axis_angle)
    print("Axis-angle magnitude (rad):",
          np.linalg.norm(axis_angle))
    print("Reconstruction error (rad):", error_rad)
    print("Reconstruction error (deg):", np.degrees(error_rad))

    assert error_rad < 1e-5
    print("Orientation consistency PASSED")

finally:
    env.close()
