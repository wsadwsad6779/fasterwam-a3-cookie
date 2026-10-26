import json
from pathlib import Path

import numpy as np
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from observation_adapter import extract_left_eef_state
from gripper_proprio_adapter import a3_opening_to_panda_fingers

STATS = Path(
    "../FasterWAM/checkpoints/fasterwam_release/libero/dataset_stats.json"
)

with STATS.open() as f:
    stats = json.load(f)

# Read Faster-WAM LIBERO global state min/max.
state_stats = stats["state"]["default"]
minimum = np.asarray(state_stats["global_min"], dtype=float)
maximum = np.asarray(state_stats["global_max"], dtype=float)

env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=False,
)

try:
    obs, _ = env.reset(seed=0)

    eef = np.asarray(extract_left_eef_state(obs), dtype=float)
    opening = float(obs["observation.state"][7])
    fingers = a3_opening_to_panda_fingers(opening)

    state = np.concatenate([eef, fingers])
    assert state.shape == (8,)

    # Diagnostic only: map training min/max to [-1, 1].
    normalized = 2 * (state - minimum) / (maximum - minimum) - 1

    names = [
        "x", "y", "z",
        "rot_x", "rot_y", "rot_z",
        "finger1", "finger2",
    ]

    print("=== A3 -> Faster-WAM proprio range diagnostic ===")
    print(f"{'name':<9} {'raw':>11} {'normalized':>12} {'in_range':>10}")

    for name, value, norm, lo, hi in zip(
        names, state, normalized, minimum, maximum
    ):
        inside = lo <= value <= hi
        print(f"{name:<9} {value:>11.6f} {norm:>12.6f} {str(inside):>10}")

    print("All dimensions within training min/max:",
          bool(np.all((state >= minimum) & (state <= maximum))))

finally:
    env.close()
