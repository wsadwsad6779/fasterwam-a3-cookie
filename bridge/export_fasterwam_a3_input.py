from pathlib import Path
import numpy as np
from PIL import Image

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from vision_adapter_official import prepare_fasterwam_images_official

OUT = Path("outputs/fasterwam_a3_input")
OUT.mkdir(parents=True, exist_ok=True)

env = A3CookieTransferEnv(
    action_mode="cartesian_delta",
    render_cameras=True,
)

try:
    obs, _ = env.reset(seed=0)

    rgb, image_tensor = prepare_fasterwam_images_official(obs)

    eef_pose = np.asarray(
        obs["observation.eef_pose"], dtype=np.float32
    ).copy()

    robot_state = np.asarray(
        obs["observation.state"], dtype=np.float32
    ).copy()

    instruction = (
        "Pick up the cookie with the left robot arm "
        "and transfer it to the right robot arm."
    )

    np.savez_compressed(
        OUT / "observation.npz",
        image=image_tensor,
        rgb=rgb,
        eef_pose=eef_pose,
        robot_state=robot_state,
        instruction=np.array(instruction),
    )

    Image.fromarray(rgb).save(OUT / "observation.png")

    print("Saved:", OUT / "observation.npz")
    print("Image tensor:", image_tensor.shape, image_tensor.dtype)
    print("EEF pose:", eef_pose.shape)
    print("Robot state:", robot_state.shape)
    print("Instruction:", instruction)

finally:
    env.close()
