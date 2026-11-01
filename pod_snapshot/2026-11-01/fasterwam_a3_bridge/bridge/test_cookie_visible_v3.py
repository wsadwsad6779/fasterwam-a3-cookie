from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v3 import FasterWAMA3ActionAdapterV3

np.set_printoptions(precision=7, suppress=True)

OUT = Path("outputs/cookie_visible_v3.mp4")
OUT.parent.mkdir(parents=True, exist_ok=True)

adapter = FasterWAMA3ActionAdapterV3()

env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=True,
)

writer = None
frames = 0


def rotation_from_wxyz(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def record(obs):
    global writer, frames

    front = obs["observation.images.front"]
    wrist = obs["observation.images.left_wrist"]

    frame = np.concatenate([front, wrist], axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if writer is None:
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(OUT),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Failed to open video writer")

    writer.write(frame)
    frames += 1


def run_action(obs, xyz, rot_world, gripper_command):
    pose = np.asarray(obs["observation.eef_pose"], dtype=float)
    R = rotation_from_wxyz(pose[3:7]).as_matrix()

    # Inputs are normalized LIBERO OSC_POSE commands.
    action7 = np.array([
        *xyz,
        *rot_world,
        gripper_command,
    ], dtype=float)

    action14 = adapter.convert(
        action7,
        left_eef_rotation=R,
        right_gripper_opening=1.0,
    )

    obs, reward, terminated, truncated, info = env.step(action14)
    record(obs)

    if terminated or truncated:
        raise RuntimeError(
            f"Episode ended: terminated={terminated}, "
            f"truncated={truncated}, info={info}"
        )

    if info.get("safety_reason"):
        raise RuntimeError(f"Safety stop: {info['safety_reason']}")

    return obs


try:
    obs, _ = env.reset(seed=0)
    initial_pose = np.asarray(
        obs["observation.eef_pose"], dtype=float
    ).copy()

    record(obs)

    # Phase 1: small +X translation and world-Z rotation.
    for _ in range(80):
        obs = run_action(
            obs,
            xyz=[0.005, 0, 0],
            rot_world=[0, 0, 0.00436],
            gripper_command=-1.0,  # LIBERO open
        )

    # Phase 2: close the gripper without moving the arm.
    for _ in range(35):
        obs = run_action(
            obs,
            xyz=[0, 0, 0],
            rot_world=[0, 0, 0],
            gripper_command=1.0,  # LIBERO close
        )

    # Phase 3: reopen the gripper.
    for _ in range(35):
        obs = run_action(
            obs,
            xyz=[0, 0, 0],
            rot_world=[0, 0, 0],
            gripper_command=-1.0,
        )

    # Phase 4: settle.
    for _ in range(20):
        obs = run_action(
            obs,
            xyz=[0, 0, 0],
            rot_world=[0, 0, 0],
            gripper_command=-1.0,
        )

    final_pose = np.asarray(
        obs["observation.eef_pose"], dtype=float
    )

    R_initial = rotation_from_wxyz(initial_pose[3:7])
    R_final = rotation_from_wxyz(final_pose[3:7])

    world_rotation = (
        R_final * R_initial.inv()
    ).as_rotvec()

    print("=== Combined V3 validation ===")
    print(
        "Left XYZ displacement (mm):",
        (final_pose[:3] - initial_pose[:3]) * 1000,
    )
    print("World rotation vector (rad):", world_rotation)
    print(
        "Right XYZ displacement (mm):",
        (final_pose[7:10] - initial_pose[7:10]) * 1000,
    )
    print("Frames:", frames)
    print("Video:", OUT.resolve())

finally:
    if writer is not None:
        writer.release()
    env.close()
