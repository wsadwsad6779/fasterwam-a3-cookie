import json
from pathlib import Path

import numpy as np

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from observation_adapter import extract_left_eef_state
from gripper_proprio_adapter import a3_opening_to_panda_fingers

STATS_PATH = Path(
    "../FasterWAM/checkpoints/fasterwam_release/libero/dataset_stats.json"
)

with STATS_PATH.open() as f:
    stats = json.load(f)

minimum = np.asarray(
    stats["state"]["default"]["global_min"], dtype=np.float64
)
maximum = np.asarray(
    stats["state"]["default"]["global_max"], dtype=np.float64
)

env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    eef6 = np.asarray(extract_left_eef_state(obs), dtype=np.float64)
    opening = float(obs["observation.state"][7])
    fingers2 = a3_opening_to_panda_fingers(opening)

    proprio8 = np.concatenate([eef6, fingers2])

    assert proprio8.shape == (8,)
    assert np.all(np.isfinite(proprio8))

    names = [
        "x", "y", "z",
        "rot_x", "rot_y", "rot_z",
        "finger1", "finger2",
    ]

    print("=== A3 -> Faster-WAM proprio diagnostic ===")
    print("A3 gripper opening:", opening)
    print()
    print(f"{'Dimension':<10} {'A3 value':>12} {'Train min':>12} "
          f"{'Train max':>12} {'Status':>12}")

    for name, value, lo, hi in zip(
        names, proprio8, minimum, maximum
    ):
        status = "IN RANGE" if lo <= value <= hi else "OUT OF RANGE"
        print(
            f"{name:<10} {value:>12.6f} {lo:>12.6f} "
            f"{hi:>12.6f} {status:>12}"
        )

    print("\nComplete proprio8:", proprio8)

finally:
    env.close()
