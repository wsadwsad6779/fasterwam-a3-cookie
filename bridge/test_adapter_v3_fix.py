"""Offline validation of action_adapter_v3 gripper binarization fix.

Verifies that the np.sign binarization prevents the gripper channel from
exceeding A3 [-1, 1] limits, matching official Faster-WAM behavior.
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from action_adapter_v3 import FasterWAMA3ActionAdapterV3


def make_identity_rotation():
    return np.eye(3, dtype=np.float64)


def test_gripper_binarize_problematic_value():
    """The exact value that caused the original crash: -1.015625."""
    adapter = FasterWAMA3ActionAdapterV3()
    action7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.015625])
    R = make_identity_rotation()

    result = adapter.convert(action7, R)
    assert result[6] == 1.0, f"Expected +1.0, got {result[6]}"
    assert np.all(np.abs(result) <= 1.0), f"Output exceeds limits: {result}"
    print(f"PASS: -1.015625 -> gripper={result[6]}")


def test_gripper_binarize_positive_overflow():
    """Positive overflow should also binarize to -1 (close)."""
    adapter = FasterWAMA3ActionAdapterV3()
    action7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.015625])
    R = make_identity_rotation()

    result = adapter.convert(action7, R)
    assert result[6] == -1.0, f"Expected -1.0, got {result[6]}"
    print(f"PASS: +1.015625 -> gripper={result[6]}")


def test_gripper_normal_values():
    """Normal gripper values should binarize correctly."""
    adapter = FasterWAMA3ActionAdapterV3()
    R = make_identity_rotation()

    for val, expected in [
        (1.0, -1.0),
        (0.5, -1.0),
        (0.01, -1.0),
        (0.0, 0.0),
        (-0.01, 1.0),
        (-0.5, 1.0),
        (-1.0, 1.0),
    ]:
        action7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, val])
        result = adapter.convert(action7, R)
        assert result[6] == expected, f"gripper {val} -> expected {expected}, got {result[6]}"
        print(f"PASS: gripper {val:+.2f} -> {result[6]:+.1f}")


def test_spatial_channels_unchanged():
    """Spatial/rotation channels should NOT be affected by binarization."""
    adapter = FasterWAMA3ActionAdapterV3()
    R = make_identity_rotation()

    action7 = np.array([0.1, -0.2, 0.3, 0.05, -0.03, 0.02, -1.015625])
    result = adapter.convert(action7, R)

    assert np.isclose(result[0], 0.2), f"x: expected 0.2, got {result[0]}"
    assert np.isclose(result[1], -0.4), f"y: expected -0.4, got {result[1]}"
    assert np.isclose(result[2], 0.6), f"z: expected 0.6, got {result[2]}"
    print(f"PASS: spatial channels unchanged by gripper fix")


def test_full_14d_output_in_range():
    """Full 14D output must be within [-1, 1]."""
    adapter = FasterWAMA3ActionAdapterV3()
    R = make_identity_rotation()

    action7 = np.array([0.4, -0.3, 0.2, 0.1, -0.05, 0.03, -1.015625])
    result = adapter.convert(action7, R, right_gripper_opening=0.5)

    assert np.all(np.abs(result) <= 1.0 + 1e-6), f"Output exceeds limits: {result}"
    assert result[13] == 0.0, f"Right gripper: expected 0.0, got {result[13]}"
    print(f"PASS: full 14D output in range, max_abs={np.max(np.abs(result)):.4f}")
    print(f"  output: {result}")


if __name__ == "__main__":
    print("=" * 60)
    print("Offline adapter validation: gripper binarization fix")
    print("=" * 60)

    test_gripper_binarize_problematic_value()
    print()
    test_gripper_binarize_positive_overflow()
    print()
    test_gripper_normal_values()
    print()
    test_spatial_channels_unchanged()
    print()
    test_full_14d_output_in_range()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
