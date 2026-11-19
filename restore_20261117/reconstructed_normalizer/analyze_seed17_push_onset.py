"""When does the recorded seed17 trajectory start pushing into an immovable object?

CPU only, reads the recorded observation and chunk arrays, no model, no simulator.

For every executed step t it prints the measured force (6 calibrated channels), the world
translation that was commanded (adapter rows are world deltas, 1 A3 unit = 0.025 m) and the
world displacement the wrist actually made.  The first frame where a large command produces a
small displacement is where the trajectory stopped being free motion and became a jam.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FWD = ROOT / 'forensics_full_wide_out'
CAL = json.load(open(ROOT.parent / 'step_040000' / 'force_calib_v1.json'))
CHANS = [int(c) for c in CAL['channels']]
NAMES = list(CAL['channel_names'])

OBS = np.load(FWD / 'seed17_observations.npz', allow_pickle=False)
CHUNKS = np.load(FWD / 'seed17_original_chunks.npz', allow_pickle=False)

print('obs keys:', {k: OBS[k].shape for k in OBS.files if not k.startswith('observation.images')})
print('chunk keys (first 6):', sorted(CHUNKS.files)[:6], '... total', len(CHUNKS.files))
pose = np.asarray(OBS['observation.eef_pose'], dtype=np.float64)
force = np.asarray(OBS['observation.force'], dtype=np.float64)
n = len(pose)
print('frames', n, 'force channels used', CHANS, NAMES)

nf = force[:, CHANS]
fmax = np.abs(nf).max(1)
print('\nfirst frame with |f|>5N :', int(np.argmax(fmax > 5.0)) if (fmax > 5.0).any() else None)
print('first frame with |f|>20N:', int(np.argmax(fmax > 20.0)) if (fmax > 20.0).any() else None)
print('first frame with |f|>100N:', int(np.argmax(fmax > 100.0)) if (fmax > 100.0).any() else None)

# executed command per step: chunk k = frames 10k..10k+9, rows 0..9
cmd = np.zeros((n, 3))
grip = np.zeros(n)
for k in range(n // 10):
    ad = np.asarray(CHUNKS['chunk_%03d_adapter_out' % k], dtype=np.float64).reshape(32, 14)
    m = min(10, n - 10 * k)
    cmd[10 * k:10 * k + m] = ad[:m, :3] * 0.025
    grip[10 * k:10 * k + m] = ad[:m, 6]

disp = np.zeros((n, 3))
disp[:-1] = pose[1:, :3] - pose[:-1, :3]
csz = np.linalg.norm(cmd, axis=1)
dsz = np.linalg.norm(disp, axis=1)
track = np.where(csz > 1e-6, dsz / np.maximum(csz, 1e-9), np.nan)

print('\n%-5s %-9s %-30s %-30s %-8s %-6s %s' % (
    'frame', 'maxF(N)', 'cmd step (mm) xyz', 'disp step (mm) xyz', 'track', 'grip', 'per-ch F'))
for t in range(n):
    if t % 5 and not (fmax[t] > 5.0 and t < 150) and not (t > 140 and t % 10):
        continue
    print('%-5d %-9.2f [%+7.2f %+7.2f %+7.2f] [%+7.2f %+7.2f %+7.2f] %-8s %-6.0f %s' % (
        t, fmax[t],
        cmd[t, 0] * 1000, cmd[t, 1] * 1000, cmd[t, 2] * 1000,
        disp[t, 0] * 1000, disp[t, 1] * 1000, disp[t, 2] * 1000,
        ('%.2f' % track[t]) if np.isfinite(track[t]) else '-',
        grip[t],
        ' '.join('%6.2f' % v for v in nf[t])))

print('\ncumulative command vs displacement over 10-step chunks:')
print('%-6s %-24s %-24s %-8s %s' % ('chunk', 'cmd sum (mm) xyz', 'disp sum (mm) xyz', 'ratio', 'maxF'))
for k in range(n // 10):
    sl = slice(10 * k, min(10 * k + 10, n))
    c = cmd[sl].sum(0) * 1000
    d = disp[sl].sum(0) * 1000
    print('%-6d [%+8.2f %+8.2f %+8.2f] [%+8.2f %+8.2f %+8.2f] %-8.2f %.2f' % (
        k, c[0], c[1], c[2], d[0], d[1], d[2],
        np.linalg.norm(d) / max(np.linalg.norm(c), 1e-9), fmax[sl].max()))

print('\nworld position at replan boundaries (m):')
for k in range(n // 10):
    p = pose[10 * k, :3]
    print('  f%-4d [%+.4f %+.4f %+.4f]' % (10 * k, p[0], p[1], p[2]))
