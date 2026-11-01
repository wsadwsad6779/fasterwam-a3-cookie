import numpy as np


class FasterWAMA3ActionAdapterV2:
    """Candidate LIBERO OSC_POSE -> A3 Cartesian action adapter.

    Experimental only. Coordinate-frame and rotation compatibility
    are not yet verified.
    """

    def __init__(
        self,
        libero_translation_scale_m=0.05,
        a3_translation_scale_m=0.025,
        libero_rotation_scale=0.5,
        a3_rotation_scale_rad=0.10,
    ):
        self.translation_ratio = (
            libero_translation_scale_m / a3_translation_scale_m
        )
        self.rotation_ratio = (
            libero_rotation_scale / a3_rotation_scale_rad
        )

    def convert(self, action7, right_gripper_opening=0.6):
        action = np.asarray(action7, dtype=np.float64)

        if action.shape != (7,):
            raise ValueError(f"Expected (7,), got {action.shape}")

        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains NaN or Inf")

        if not 0 <= right_gripper_opening <= 1:
            raise ValueError("Invalid right gripper opening")

        output = np.zeros(14, dtype=np.float64)

        # Candidate normalized OSC translation -> normalized A3 delta.
        output[:3] = action[:3] * self.translation_ratio

        # Candidate rotation scaling.
        # Rotation semantics and reference frame still need verification.
        output[3:6] = action[3:6] * self.rotation_ratio

        # Model gripper: 0 closed, 1 open.
        # A3 Cartesian input: -1 closed, +1 open.
        if not 0 <= action[6] <= 1:
            raise ValueError("Model gripper outside [0, 1]")
        output[6] = 2 * action[6] - 1

        # Hold right arm position and preserve its gripper opening.
        output[13] = 2 * right_gripper_opening - 1

        # Reject rather than silently clip.
        if np.any(np.abs(output) > 1):
            raise ValueError(
                f"Converted action exceeds A3 limits: {output}"
            )

        return output
