"""Per-step safety wrapper for the A3 cookie-transfer rollout.

The policy only emits the LIBERO gripper sign (+1 close / -1 open) while the
A3 env treats that channel as an *absolute* opening target, so the raw rollout
alternates between fully open and fully shut: an empty air close, then a hard
clamp that wedges into the cookie rows (4+ N on the pads) and is pushed back
open. Commanding "measured - 0.02" does not fix it either: the left jaw is a
kp=400 position servo, so a 1.7 mm error cannot beat the wedge.

The wrapper owns the left gripper and mirrors the expert's batch grasp:
  1. pre-close to PREOPEN=0.30 (five 6.37 mm cookies) while staying high,
  2. descend at 0.30 so the pads enter the two boundary gaps,
  3. ramp the absolute target down at 3-6 mm/step until the jaws stall with
     real pad force, never below MIN_TARGET (0.24) so nothing is crushed,
  4. hold that target through lift and transport,
  5. open only where the profile allows, slowly (2 mm of jaw per step),
  6. re-arm over the source so the next batch gets a gripper again.

Profiles:
  baseline  untouched policy
  force     pre-close + soft ramp + contact latch + hold; release when the
            policy asks to open
  guard     force + soft z floor at the source cookies + release only while
            the batch is over the target bin
  full      guard + guided placement: steer to the bin, descend inside the
            footprint, release only below the rim
"""
import os

import numpy as np

LEFT_GRIP = 6

PREOPEN = float(os.environ.get("A3_PREOPEN", "0.30"))
OPEN_INCR = float(os.environ.get("A3_OPEN_INCR", "0.02"))
CLOSE_RATE = float(os.environ.get("A3_CLOSE_RATE", "0.006"))
CONTACT_RATE = float(os.environ.get("A3_CONTACT_RATE", "0.003"))
MIN_TARGET = float(os.environ.get("A3_MIN_TARGET", "0.24"))
CONTACT_FORCE = float(os.environ.get("A3_CONTACT_FORCE", "0.30"))
HOLD_FORCE = float(os.environ.get("A3_HOLD_FORCE", "0.80"))
STALL_TOL = float(os.environ.get("A3_STALL_TOL", "0.012"))
Z_PAD = float(os.environ.get("A3_Z_PAD", "0.010"))
SRC_RADIUS = float(os.environ.get("A3_SRC_RADIUS", "0.14"))
BIN_NEAR = float(os.environ.get("A3_BIN_NEAR", "0.20"))
HOLD_MIN = int(os.environ.get("A3_HOLD_MIN", "10"))
HOLD_TIMEOUT = int(os.environ.get("A3_HOLD_TIMEOUT", "260"))
RELEASE_TIMEOUT = int(os.environ.get("A3_RELEASE_TIMEOUT", "160"))
STEER_AFTER = int(os.environ.get("A3_STEER_AFTER", "60"))
DONE_DWELL = int(os.environ.get("A3_DONE_DWELL", "12"))
MAX_CYCLES = int(os.environ.get("A3_MAX_CYCLES", "8"))
RELEASE_OPEN = 0.49
CARRY_Z = 0.035
RELEASE_DEPTH = float(os.environ.get("A3_RELEASE_DEPTH", "0.018"))

PROFILES = {
    "baseline": dict(force=False, z_floor=False, gate=False, place=False),
    "force": dict(force=True, z_floor=False, gate=False, place=False),
    "guard": dict(force=True, z_floor=True, gate=True, place=False),
    "full": dict(force=True, z_floor=True, gate=True, place=True),
}

PHASES = ("open", "preclose", "descend", "close", "hold", "release", "done", "raw")


def new_state():
    return {
        "phase": "open",
        "steps": 0,
        "tgt": 1.0,
        "contact": 0,
        "close_steps": 0,
        "descend_steps": 0,
        "hover": 0,
        "hold_steps": 0,
        "zfloor": 0,
        "steer": 0,
        "suppress": 0,
        "inner": 0.0,
        "outer": 0.0,
        "bin_dist": 0.0,
        "released": False,
        "dropped": False,
        "dwell": 0,
        "cycles": 0,
        "placed": 0,
    }


def nearest_cookie(env, tcp):
    cps = env.cookie_positions
    d = np.linalg.norm(cps - np.asarray(tcp)[None, :], axis=1)
    idx = int(np.argmin(d))
    return float(d[idx]), cps[idx]


def apply(a3, obs, env, st, prof):
    tcp = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
    opening = float(obs["observation.state"][7])
    st["steps"] += 1

    if not prof["force"]:
        st["phase"] = "raw"
        return a3

    inner = float(env._sensor("L_finger_inner_touch_sensor")[0])
    outer = float(env._sensor("L_finger_outer_touch_sensor")[0])
    st["inner"], st["outer"] = inner, outer
    near, cpos = nearest_cookie(env, tcp)
    near_xy = float(np.linalg.norm(tcp[:2] - cpos[:2]))
    wants_open = a3[LEFT_GRIP] >= 0.0

    bin_pos = np.asarray(env.data.xpos[env._target_bin_body], dtype=np.float64)
    half = np.asarray(env.TARGET_INNER_HALF_SIZE, dtype=np.float64)
    rim_z = float(bin_pos[2]) + float(env.TARGET_WALL_HEIGHT)
    bin_dist = float(np.linalg.norm(tcp[:2] - bin_pos[:2]))
    st["bin_dist"] = bin_dist
    inside = bool(
        abs(tcp[0] - bin_pos[0]) < half[0] - 0.003
        and abs(tcp[1] - bin_pos[1]) < half[1] - 0.003
    )
    low = bool(tcp[2] <= rim_z + RELEASE_DEPTH)

    def cmd(target):
        a3[LEFT_GRIP] = 2.0 * float(target) - 1.0

    def let_go():
        st["dropped"] = not (inside and low)
        if inside and low:
            st["placed"] += 1
        st["phase"] = "release"
        st["released"] = True
        st["dwell"] = 0

    phase = st["phase"]

    # z_floor only while approaching a source cookie: once something is held,
    # the nearest cookie rises with the tool and the floor would chase it.
    if prof["z_floor"] and phase in ("open", "preclose", "descend", "close"):
        src = np.asarray(env.SOURCE_CENTER, dtype=np.float64)[:2]
        if abs(cpos[0] - src[0]) < SRC_RADIUS and abs(cpos[1] - src[1]) < SRC_RADIUS:
            floor_z = float(cpos[2]) + Z_PAD
            if near_xy < 0.05 and tcp[2] < floor_z:
                st["zfloor"] += 1
                a3[2] = max(float(a3[2]), 0.45 if tcp[2] < floor_z - 0.004 else 0.12)

    if phase == "open":
        cmd(min(1.0, st["tgt"]))
        near_src = near_xy < 0.13 and tcp[2] > float(cpos[2]) + 0.06
        if not wants_open or (st["steps"] >= 12 and near_src):
            st["phase"] = "preclose"
            st["tgt"] = PREOPEN

    elif phase == "preclose":
        st["tgt"] = PREOPEN
        cmd(PREOPEN)
        a3[2] = max(float(a3[2]), 0.0)
        if opening <= PREOPEN + 0.03 or st["steps"] > 90:
            st["phase"] = "descend"

    elif phase == "descend":
        st["descend_steps"] += 1
        cmd(PREOPEN)
        at_floor = (
            prof["z_floor"]
            and near_xy < 0.05
            and tcp[2] <= float(cpos[2]) + Z_PAD + 0.002
        )
        low_enough = tcp[2] <= float(cpos[2]) + 0.015
        if near_xy < 0.05 and tcp[2] > float(cpos[2]) + 0.025:
            st["hover"] += 1
            if st["hover"] > 12:
                a3[2] = min(float(a3[2]), -0.15)
        else:
            st["hover"] = 0
        a3[2] = max(float(a3[2]), -0.20)
        at_pile = near_xy < 0.04 and tcp[2] <= float(cpos[2]) + 0.05
        if (
            (not wants_open and near_xy < 0.04 and low_enough)
            or at_floor
            or (st["descend_steps"] > 140 and at_pile)
        ):
            st["phase"] = "close"
            st["tgt"] = PREOPEN
            st["close_steps"] = 0
            st["contact"] = 0
        elif st["descend_steps"] > 140:
            # Never squeeze thin air: drop back to the approach and try again
            # if the policy brings the tool down to the pile.
            st["phase"] = "open"
            st["tgt"] = 1.0
            st["descend_steps"] = 0

    elif phase == "close":
        st["close_steps"] += 1
        a3[2] = max(float(a3[2]), -0.08)
        stalled = opening > st["tgt"] + STALL_TOL and min(inner, outer) > CONTACT_FORCE
        st["contact"] = st["contact"] + 1 if stalled else 0
        rate = CONTACT_RATE if st["contact"] > 0 else CLOSE_RATE
        st["tgt"] = max(MIN_TARGET, st["tgt"] - rate)
        cmd(st["tgt"])
        if st["contact"] >= 2 or st["tgt"] <= MIN_TARGET + 1e-9 or st["close_steps"] > 90:
            st["phase"] = "hold"
            st["hold_steps"] = 0

    elif phase == "hold":
        if min(inner, outer) < HOLD_FORCE and st["tgt"] > MIN_TARGET:
            st["tgt"] = max(MIN_TARGET, st["tgt"] - CONTACT_RATE)
        cmd(st["tgt"])
        st["hold_steps"] += 1
        if prof["place"]:
            if inside and low and st["hold_steps"] >= HOLD_MIN:
                let_go()
            elif (
                st["hold_steps"] >= HOLD_MIN
                and (bin_dist < BIN_NEAR or st["hold_steps"] >= STEER_AFTER)
            ):
                # Place assist: keep the batch shut, clear the walls, aim at
                # the bin, then descend; release happens from geometry above.
                st["steer"] += 1
                ex = float(bin_pos[0] - tcp[0])
                ey = float(bin_pos[1] - tcp[1])
                a3[0] = float(np.clip(0.8 * ex / 0.025, -0.35, 0.35))
                a3[1] = float(np.clip(0.8 * ey / 0.025, -0.35, 0.35))
                # Inside the footprint: descend until the release depth is
                # reached. Outside it: never go below the walls, lift if the
                # policy already did.
                if inside:
                    a3[2] = min(float(a3[2]), -0.20)
                elif tcp[2] < rim_z + CARRY_Z:
                    a3[2] = max(float(a3[2]), 0.25)
            if st["phase"] == "hold" and st["hold_steps"] >= HOLD_TIMEOUT:
                let_go()
        else:
            if prof["gate"]:
                # Guard: the policy owns transport, the wrapper only refuses
                # to let go until the batch is over the bin and low enough.
                fire = (
                    st["hold_steps"] >= HOLD_MIN and inside and low
                ) or st["hold_steps"] >= HOLD_TIMEOUT
            else:
                fire = (wants_open and st["hold_steps"] >= HOLD_MIN) or \
                    st["hold_steps"] >= RELEASE_TIMEOUT
            if fire:
                let_go()

    elif phase == "release":
        st["tgt"] = min(RELEASE_OPEN, st["tgt"] + OPEN_INCR)
        cmd(st["tgt"])
        if st["tgt"] >= RELEASE_OPEN - 1e-9:
            st["phase"] = "done"

    elif phase == "done":
        cmd(RELEASE_OPEN)
        st["dwell"] += 1
        # Second and later batches: the policy expects a usable gripper when
        # it comes back over the source, not a permanently open one.
        near_src_high = near_xy < 0.13 and tcp[2] > float(cpos[2]) + 0.06
        if (
            st["dwell"] >= DONE_DWELL
            and st["cycles"] < MAX_CYCLES
            and not inside
            and near_src_high
            and not wants_open
        ):
            st["cycles"] += 1
            st["phase"] = "preclose"
            st["tgt"] = PREOPEN
            st["contact"] = 0
            st["close_steps"] = 0
            st["descend_steps"] = 0
            st["hover"] = 0
            st["hold_steps"] = 0
            st["released"] = False
            st["dropped"] = False
            st["dwell"] = 0

    return a3
