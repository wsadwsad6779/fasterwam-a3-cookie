from pathlib import Path

import cv2
import numpy as np


def prepare_fasterwam_images(obs, output_path=None):
    keys = (
        "observation.images.front",
        "observation.images.left_wrist",
    )

    images = []

    for key in keys:
        rgb = np.asarray(obs[key])

        if rgb.shape != (256, 256, 3):
            raise ValueError(f"{key}: unexpected shape {rgb.shape}")

        if rgb.dtype != np.uint8:
            raise ValueError(f"{key}: expected uint8 RGB")

        # First-pass adapter: resize to model input resolution.
        rgb224 = cv2.resize(
            rgb,
            (224, 224),
            interpolation=cv2.INTER_AREA,
        )

        images.append(rgb224)

    merged = np.concatenate(images, axis=1)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(path),
            cv2.cvtColor(merged, cv2.COLOR_RGB2BGR),
        )

    # uint8 RGB [224,448,3] -> float32 [1,3,224,448]
    model_input = (
        merged.astype(np.float32) * (2.0 / 255.0) - 1.0
    )
    model_input = np.transpose(model_input, (2, 0, 1))[None]

    return model_input, merged
