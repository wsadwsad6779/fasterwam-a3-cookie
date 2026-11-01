"""Interface + safety layer only. No task logic, no simulator access.

In the 2x2 ablation this is the low-intervention condition: the policy keeps all
six pose dimensions and the gripper command, and the layer only clamps what
would damage hardware or the bin. It reads `obs` alone -- no `env`, no
`cookie_positions`, no `data`. Forces come from `obs["observation.force"]`
(`contracts.FORCE`), whose layout is fixed by `A3DualArmEnv._observation`:

    0:3 L wrist force, 3:6 L wrist torque, 6:9 R wrist force, 9:12 R torque,
    12 L finger inner touch, 13 L finger outer touch, 14/15 R side, 16/17
    summed finger actuator forces.

Bin geometry is the calibration-time constant from `cookie_batch.yaml`
(`target_bin_world_position_m=[0.095, 0.100, 0.753]`, inner half `[0.057, 0.029]`,
wall `0.026`), not a per-episode simulator read.

Thresholds are placeholders until `calibrate.py --expert-steps` reports the
peak pad force the scripted expert itself produces; override them on the
command line rather than editing this file.
"""
from dataclasses import dataclass

import numpy as np

LEFT_GRIP = 6
L_INNER = 12
L_OUTER = 13
FORCE_KEY = "observation.force"


@dataclass
class SafetyLimits:
    # Tier 1: instant, no questions asked.
    hard_force_n: float = 40.0          # finger actuator forcerange
    max_step_translation: float = 1.0   # normalised A3 units per step
    max_step_rotation: float = 1.0
    # Tier 2: sustained, needs several consecutive steps.
    soft_force_n: float = 25.0
    soft_force_steps: int = 3
    soft_retreat: float = 0.06
    # Fixed scene geometry: half size [0.063, 0.035] minus a 6 mm wall.
    bin_xy: tuple = (0.095, 0.100)
    bin_base_z: float = 0.753
    bin_inner_half: tuple = (0.057, 0.029)
    bin_wall_height: float = 0.026
    bin_floor_margin: float = 0.010


def new_state() -> dict:
    return {"soft": 0, "hard": 0, "binfloor": 0, "clipped": 0, "estop": False}


def pad_force(obs) -> float:
    force = np.asarray(obs[FORCE_KEY], dtype=np.float64)
    if force.size < L_OUTER + 1:
        raise ValueError(f"unexpected force layout: {force.shape}")
    return float(max(force[L_INNER], force[L_OUTER]))


def apply(a3: np.ndarray, obs: dict, st: dict, lim: SafetyLimits) -> np.ndarray:
    a3 = np.asarray(a3, dtype=np.float64).copy()

    before = a3.copy()
    np.clip(a3[0:3], -lim.max_step_translation, lim.max_step_translation, out=a3[0:3])
    np.clip(a3[3:6], -lim.max_step_rotation, lim.max_step_rotation, out=a3[3:6])
    if not np.array_equal(before, a3):
        st["clipped"] += 1

    tcp = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
    inside = (abs(tcp[0] - lim.bin_xy[0]) < lim.bin_inner_half[0] + 0.004
              and abs(tcp[1] - lim.bin_xy[1]) < lim.bin_inner_half[1] + 0.004)
    rim_z = lim.bin_base_z + lim.bin_wall_height

    force = pad_force(obs)

    if (not np.all(np.isfinite(a3))) or force > lim.hard_force_n:
        st["hard"] += 1
        st["estop"] = True
        a3[0:6] = 0.0
        a3[LEFT_GRIP] = 1.0
        return a3

    if force > lim.soft_force_n:
        st["soft"] += 1
        if st["soft"] >= lim.soft_force_steps:
            a3[2] = max(float(a3[2]), lim.soft_retreat)
            a3[LEFT_GRIP] = max(float(a3[LEFT_GRIP]), 0.0)
    else:
        st["soft"] = 0

    if inside and tcp[2] < rim_z + lim.bin_floor_margin:
        st["binfloor"] += 1
        a3[2] = max(float(a3[2]), 0.06 if tcp[2] < rim_z - 0.004 else 0.02)

    return a3
