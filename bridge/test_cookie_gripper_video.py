from pathlib import Path
import cv2
import numpy as np

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v2 import FasterWAMA3ActionAdapterV2

out = Path("outputs")
out.mkdir(exist_ok=True)

env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=True,
)
adapter = FasterWAMA3ActionAdapterV2()
writer = None

try:
    obs, _ = env.reset(seed=0)
    state = np.asarray(obs["observation.state"], dtype=float)
    right_opening = float(state[15])

    stages = [
        ("OPEN", 1.0, 15),
        ("CLOSE", 0.0, 30),
        ("OPEN AGAIN", 1.0, 30),
    ]

    for title, opening, steps in stages:
        action = adapter.convert(
            [0, 0, 0, 0, 0, 0, opening],
            right_gripper_opening=right_opening,
        )

        for i in range(steps):
            obs, _, terminated, truncated, info = env.step(action)

            if i % 2 == 0:
                front = np.asarray(
                    obs["observation.images.front"], dtype=np.uint8
                )
                wrist = np.asarray(
                    obs["observation.images.left_wrist"], dtype=np.uint8
                )

                h, w = front.shape[:2]
                wrist = cv2.resize(wrist, (w, h))
                frame = np.concatenate([front, wrist], axis=1)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                current_opening = float(
                    obs["observation.state"][7]
                )

                cv2.putText(
                    frame,
                    f"{title} | opening={current_opening:.3f}",
                    (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2,
                )

                if writer is None:
                    writer = cv2.VideoWriter(
                        str(out / "cookie_gripper_test.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open MP4 writer")

                writer.write(frame)

            if terminated or truncated or info.get("safety_reason"):
                raise RuntimeError(
                    f"Simulation stopped: {info.get('safety_reason')}"
                )

        print(
            f"{title}: target={opening:.2f}, "
            f"actual={obs['observation.state'][7]:.4f}"
        )

finally:
    if writer is not None:
        writer.release()
    env.close()

print("Saved: outputs/cookie_gripper_test.mp4")
