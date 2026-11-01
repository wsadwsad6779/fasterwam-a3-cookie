import numpy as np


def a3_opening_to_panda_fingers(opening):
    """Approximate Panda finger joint positions from A3 opening.

    A3:
        0.0 = fully closed
        1.0 = fully open

    Panda:
        finger_joint1: [0, +0.04] meters
        finger_joint2: [-0.04, 0] meters
    """
    g = float(opening)

    if not np.isfinite(g):
        raise ValueError("Gripper opening is not finite")

    if not 0.0 <= g <= 1.0:
        raise ValueError("Gripper opening must be within [0, 1]")

    return np.array([0.04 * g, -0.04 * g], dtype=np.float32)


if __name__ == "__main__":
    for g in [0.0, 0.5, 1.0]:
        print(f"A3 opening={g:.1f} -> Panda fingers={a3_opening_to_panda_fingers(g)}")

    assert np.allclose(a3_opening_to_panda_fingers(0.0), [0, 0])
    assert np.allclose(a3_opening_to_panda_fingers(0.5), [0.02, -0.02])
    assert np.allclose(a3_opening_to_panda_fingers(1.0), [0.04, -0.04])

    print("Gripper proprio adapter test PASSED")
