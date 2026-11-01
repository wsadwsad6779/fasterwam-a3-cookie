import numpy as np
from PIL import Image


def center_crop_resize(image, width=224, height=224):
    """Match Faster-WAM eval_libero_single.py exactly."""
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got {image.shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got {image.dtype}")

    pil_image = Image.fromarray(image)

    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)

    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)),
        resample=Image.BILINEAR,
    )

    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)

    cropped = resized.crop(
        (left, top, left + width, top + height)
    )

    return np.asarray(cropped, dtype=np.uint8)


def prepare_fasterwam_images_official(obs):
    """
    A3 front + left_wrist -> Faster-WAM image input.

    Returns:
        rgb: uint8 [224, 448, 3]
        x: float32 [1, 3, 224, 448], range [-1, 1]
    """
    front = center_crop_resize(
        obs["observation.images.front"]
    )

    wrist = center_crop_resize(
        obs["observation.images.left_wrist"]
    )

    rgb = np.concatenate([front, wrist], axis=1)

    x = (
        rgb.transpose(2, 0, 1)[None]
        .astype(np.float32)
        * (2.0 / 255.0)
        - 1.0
    )

    if x.shape != (1, 3, 224, 448):
        raise ValueError(f"Unexpected model input shape: {x.shape}")

    return rgb, x
