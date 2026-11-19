"""Dump a few recorded camera frames as PNG for visual inspection of the contact object.

Reads only the recorded observation archive; writes PNGs next to this script.
"""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
FWD = ROOT / 'forensics_full_wide_out'
OUT = ROOT / 'gpu_mechanism_v6_20261118' / 'frames'
OUT.mkdir(exist_ok=True)
OBS = np.load(FWD / 'seed17_observations.npz', allow_pickle=False)
frames = [int(x) for x in __import__('os').environ.get('FRAMES', '60,129,143,160,200').split(',')]
for t in frames:
    for cam in ('front', 'left_wrist'):
        a = np.asarray(OBS['observation.images.%s' % cam][t])
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        p = OUT / ('f%03d_%s.png' % (t, cam))
        Image.fromarray(a).save(p)
        print(p, a.shape, a.dtype)
