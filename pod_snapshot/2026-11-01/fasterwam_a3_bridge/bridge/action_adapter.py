import numpy as np


class FasterWAMA3ActionAdapter:
    """Experimental single-arm Faster-WAM -> A3 Cartesian adapter."""

    def __init__(
        self,
        translation_scale_m=0.025,
        rotation_scale_rad=0.10,
    ):
        self.translation_scale_m = translation_scale_m
        self.rotation_scale_rad = rotation_scale_rad

    def convert(self, action7, right_gripper_opening=0.6):
        action = np.asarray(action7, dtype=np.float64)

        if action.shape != (7,):
            raise ValueError(f"Expected 7D action, got {action.shape}")

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or Inf")

        if not 0.0 <= right_gripper_opening <= 1.0:
            raise ValueError("Invalid right gripper opening")

        output = np.zeros(14, dtype=np.float64)

        # ASSUMPTION: Faster-WAM translation is measured in meters.
        # Coordinate-frame compatibility is NOT verified.
        output[:3] = action[:3] / self.translation_scale_m

        # ASSUMPTION: Faster-WAM rotation is compatible with
        # A3's quaternion integration convention.
        output[3:6] = action[3:6] / self.rotation_scale_rad

        # Faster-WAM 0=closed, 1=open
        # A3 cartesian_delta -1=closed, +1=open
        output[6] = 2.0 * action[6] - 1.0

        # Right arm: zero Cartesian motion, preserve opening.
        output[13] = 2.0 * right_gripper_opening - 1.0

        # Never silently clip unknown model actions.
        if np.any(np.abs(output) > 1.0):
            raise ValueError(
                f"Converted action exceeds A3 limits: {output}"
            )

        return output
