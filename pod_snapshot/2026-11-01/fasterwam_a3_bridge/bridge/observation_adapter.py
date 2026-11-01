import numpy as np


def quat_wxyz_to_axis_angle(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=np.float64)

    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion (4,), got {quat.shape}")

    if not np.all(np.isfinite(quat)):
        raise ValueError("Quaternion contains NaN or Inf")

    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError("Invalid zero quaternion")

    quat = quat / norm

    # Match LIBERO's xyzw quaternion convention.
    xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])

    w = np.clip(xyzw[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - w * w))

    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis_angle = xyzw[:3] * (2.0 * np.arccos(w) / denominator)

    return axis_angle.astype(np.float32)


def extract_left_eef_state(observation):
    pose = np.asarray(
        observation["observation.eef_pose"],
        dtype=np.float64,
    )

    if pose.shape != (14,):
        raise ValueError(f"Expected A3 eef_pose (14,), got {pose.shape}")

    xyz = pose[:3]
    quat_wxyz = pose[3:7]
    rotation = quat_wxyz_to_axis_angle(quat_wxyz)

    result = np.concatenate([xyz, rotation]).astype(np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError("Invalid EEF state")

    return result
