import numpy as np

from action_adapter_v2 import FasterWAMA3ActionAdapterV2

adapter = FasterWAMA3ActionAdapterV2()

cases = [
    ("zero motion", [0, 0, 0, 0, 0, 0, 1]),
    ("x translation", [0.1, 0, 0, 0, 0, 0, 1]),
    ("z rotation", [0, 0, 0, 0, 0, 0.02, 1]),
    ("close gripper", [0, 0, 0, 0, 0, 0, 0]),
]

for name, action in cases:
    result = adapter.convert(action)
    assert result.shape == (14,)
    assert np.all(np.isfinite(result))
    print(name, np.round(result, 4))

assert np.isclose(adapter.convert(cases[1][1])[0], 0.2)
assert np.isclose(adapter.convert(cases[2][1])[5], 0.1)
assert np.isclose(adapter.convert(cases[3][1])[6], -1)

try:
    adapter.convert([0.75, 0, 0, 0, 0, 0, 1])
except ValueError:
    print("Oversized action correctly rejected.")
else:
    raise AssertionError("Oversized action was not rejected")

print("All V2 adapter tests passed.")
