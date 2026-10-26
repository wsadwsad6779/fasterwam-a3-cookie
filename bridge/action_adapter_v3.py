import numpy as np


class FasterWAMA3ActionAdapterV3:
    """LIBERO OSC_POSE action -> A3 cartesian_delta action."""

    def convert(self, action7, left_eef_rotation, right_gripper_opening=1.0):
        action = np.asarray(action7, dtype=np.float64)
        R = np.asarray(left_eef_rotation, dtype=np.float64)

        if action.shape != (7,):
            raise ValueError("Expected 7D LIBERO action")

        if R.shape != (3, 3):
            raise ValueError("Expected 3x3 left EEF rotation matrix")

        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(R)):
            raise ValueError("Non-finite input")

        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4):
            raise ValueError("Invalid rotation matrix")

        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
            raise ValueError("Invalid rotation determinant")

        if not 0.0 <= right_gripper_opening <= 1.0:
            raise ValueError("Invalid right gripper opening")

        output = np.zeros(14, dtype=np.float64)

        # LIBERO: 1 unit = 0.05 m
        # A3:     1 unit = 0.025 m
        output[:3] = action[:3] * 2.0

        # LIBERO rotation is expressed in world coordinates.
        # A3 rotation is expressed in local EEF coordinates.
        world_rotvec = action[3:6] * 0.5
        local_rotvec = R.T @ world_rotvec

        # A3: 1 unit = 0.1 rad
        output[3:6] = local_rotvec / 0.1

        # Match official Faster-WAM EVALUATION.binarize_gripper=true:
        # discretize to {-1, 0, +1} before sign flip, preventing values
        # like -1.015625 from exceeding A3 [-1, 1] limits after inversion.
        # LIBERO convention: +1 close, -1 open.
        # A3 convention: -1 close, +1 open.
        output[6] = -np.sign(action[6])

        # Keep right gripper at requested opening.
        output[13] = 2.0 * right_gripper_opening - 1.0

        return np.clip(output, -1.0, 1.0)
