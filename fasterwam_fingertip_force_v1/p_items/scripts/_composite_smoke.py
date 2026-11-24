import os
import sys

import imageio
import numpy as np

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')

import t7_eval as E

frames = []
hist = []
rng = np.random.default_rng(0)
for t in range(60):
    obs = {
        'observation.images.left_wrist': rng.integers(50, 255, (256, 256, 3), dtype=np.uint8),
        'observation.images.front': rng.integers(50, 255, (256, 256, 3), dtype=np.uint8),
    }
    hist.append((1.4 + 0.4 * np.sin(t / 6.0), 0.9 + 0.25 * np.cos(t / 8.0)))
    frames.append(E.composite_frame(obs, hist, t, 7100,
                                    {'cookies_in_target': 3, 'cookies_in_source': 77,
                                     'safety_reason': None}, 0.49, 0.15))
imageio.mimsave('/workspace/fwv1/logs/composite_smoke.mp4', frames, fps=20,
                codec='libx264', quality=8)
imageio.imwrite('/workspace/fwv1/logs/composite_smoke_frame.png', frames[45])
print('shape', frames[0].shape, 'bytes', os.path.getsize('/workspace/fwv1/logs/composite_smoke.mp4'))