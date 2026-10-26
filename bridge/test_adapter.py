import numpy as np
from action_adapter import FasterWAMA3ActionAdapter

adapter = FasterWAMA3ActionAdapter()

tests = [
    ("zero motion, open", [0, 0, 0, 0, 0, 0, 1]),
    ("move x 5mm", [0.005, 0, 0, 0, 0, 0, 1]),
    ("close gripper", [0, 0, 0, 0, 0, 0, 0]),
    ("rotate z 0.02rad", [0, 0, 0, 0, 0, 0.02, 1]),
]

for name, action in tests:
    converted = adapter.convert(action)
    print(name)
    print(np.round(converted, 4))
    assert converted.shape == (14,)
    assert np.all(np.abs(converted) <= 1.0)

print("All adapter tests passed.")
