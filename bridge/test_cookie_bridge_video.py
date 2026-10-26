from pathlib import Path

import cv2
import numpy as np

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2


OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

VIDEO = OUT / "cookie_bridge_smoke.mp4"
FPS = 10


def get_image(obs, key):
    image = np.asarray(obs[key])

    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def make_frame(obs, initial_xyz, title):
    front = get_image(obs, "observation.images.front")
    wrist = get_image(obs, "observation.images.left_wrist")

    h, w = front.shape[:2]
    wrist = cv2.resize(wrist, (w, h))

    canvas = np.concatenate([front, wrist], axis=1)

    xyz = np.asarray(obs["observation.eef_pose"][:3], dtype=float)
    delta_mm = (xyz - initial_xyz) * 1000

    cv2.putText(
        canvas, "FRONT CAMERA", (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
    )
    cv2.putText(
        canvas, "LEFT WRIST", (w + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
    )
    cv2.putText(
        canvas, title, (12, h - 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
    )
    cv2.putText(
        canvas,
        f"Left EEF delta XYZ (mm): {delta_mm[0]:+.3f}, "
        f"{delta_mm[1]:+.3f}, {delta_mm[2]:+.3f}",
        (12, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1,
    )

    return canvas


def main():
    env = A3CookieTransferEnv(
        action_mode="cartesian_delta",
        render_cameras=True,
    )

    writer = None

    try:
        obs, info = env.reset(seed=0)

        initial_pose = np.asarray(
            obs["observation.eef_pose"], dtype=float
        ).copy()
        initial_state = np.asarray(
            obs["observation.state"], dtype=float
        ).copy()

        adapter = FasterWAMA3ActionAdapterV2()

        # Candidate LIBERO-style action:
        # +0.05 normalized X -> A3 target +2.5 mm.
        action7 = np.array([
            0.05, 0, 0,
            0, 0, 0,
            float(initial_state[7]),
        ])

        action14 = adapter.convert(
            action7,
            right_gripper_opening=float(initial_state[15]),
        )

        hold = np.zeros(14, dtype=float)
        hold[6] = action14[6]
        hold[13] = action14[13]

        first = make_frame(
            obs, initial_pose[:3], "Initial state"
        )

        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(VIDEO),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError("Cannot open MP4 video writer")

        # Initial pose: half a second.
        for _ in range(5):
            writer.write(first)

        print("Scene: cookie transfer")
        print("Commanded left X displacement: +2.5 mm")
        print("Converted A3 action:", action14)

        # Send one displacement, then hold joint targets.
        for step in range(41):
            command = action14 if step == 0 else hold

            obs, reward, terminated, truncated, info = env.step(
                command
            )

            if step % 2 == 0:
                frame = make_frame(
                    obs,
                    initial_pose[:3],
                    f"Simulation step {step + 1}",
                )
                writer.write(frame)

            if info.get("safety_reason"):
                print("SAFETY:", info["safety_reason"])
                break

            if terminated or truncated:
                print("Episode finished:", terminated, truncated)
                break

        final_pose = np.asarray(
            obs["observation.eef_pose"], dtype=float
        )

        actual_mm = (
            final_pose[:3] - initial_pose[:3]
        ) * 1000

        error_mm = actual_mm - np.array([2.5, 0, 0])

        print("\n=== Cookie scene motion validation ===")
        print("Actual left XYZ displacement (mm):", actual_mm)
        print("XYZ displacement error (mm):", error_mm)
        print("3D error norm (mm):", np.linalg.norm(error_mm))
        print(
            "Right XYZ displacement (mm):",
            (final_pose[7:10] - initial_pose[7:10]) * 1000,
        )
        print("Safety reason:", info.get("safety_reason"))

    finally:
        if writer is not None:
            writer.release()
        env.close()

    print("Video saved:", VIDEO.resolve())


if __name__ == "__main__":
    main()
