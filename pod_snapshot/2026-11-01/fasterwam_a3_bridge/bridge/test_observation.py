import numpy as np

from observation_adapter import (
    quat_wxyz_to_axis_angle,
    extract_left_eef_state,
)

identity = quat_wxyz_to_axis_angle([1, 0, 0, 0])
assert np.allclose(identity, [0, 0, 0], atol=1e-6)

rotation_z_90 = quat_wxyz_to_axis_angle([
    np.cos(np.pi / 4),
    0,
    0,
    np.sin(np.pi / 4),
])

assert np.allclose(
    rotation_z_90,
    [0, 0, np.pi / 2],
    atol=1e-5,
)

observation = {
    "observation.eef_pose": np.array([
        0.40, 0.15, 0.25,
        1.0, 0.0, 0.0, 0.0,
        0.50, -0.15, 0.30,
        1.0, 0.0, 0.0, 0.0,
    ])
}

state = extract_left_eef_state(observation)

assert state.shape == (6,)
assert np.allclose(state[:3], [0.40, 0.15, 0.25])
assert np.allclose(state[3:], [0, 0, 0])

print("Identity quaternion:", identity)
print("90-degree Z rotation:", rotation_z_90)
print("Left EEF state:", state)
print("All observation adapter tests passed.")
