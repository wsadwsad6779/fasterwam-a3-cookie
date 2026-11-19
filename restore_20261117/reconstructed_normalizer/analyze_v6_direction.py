"""World-frame reading of the v6 plans: which way does the amplified plan push?

Translation dims 0..2 of the LIBERO action are world-frame deltas (adapter: out[:3] = a[:3]*2,
A3 unit = 0.025 m), so no rotation is needed for them.  This script prints, per replan frame,
the commanded world translation for the recorded plan (A0) and for the support-clipped plan, and
the world displacement the robot actually made over the following ten steps, read from the
recorded wrist pose.  A large command with a near-zero displacement is a hard contact.

CPU only, no model, no simulator.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpu_mechanism_v6_20261118'
STATS = json.load(open(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'))
LO = np.asarray(STATS['action']['default']['global_min'], dtype=np.float64)
HI = np.asarray(STATS['action']['default']['global_max'], dtype=np.float64)

FRAMES = [60, 70, 140, 150, 160, 200, 210]
PAIRS = [(60, 70), (140, 150), (150, 160), (200, 210)]


def load(name):
    z = np.load(OUT / (name + '.npz'))
    return z


def denorm(raw32):
    x = np.asarray(raw32, dtype=np.float64)
    return (x + 1.0) / 2.0 * (HI - LO) + LO


def a3_trans(raw32):
    """First-ten world translation commands in A3 units (1 unit = 0.025 m), clipped to +-1."""
    a = denorm(raw32)[:10]
    return np.clip(a[:, :3] * 2.0, -1.0, 1.0)


def gripper(raw32):
    a = denorm(raw32)[:10, 6]
    pm1 = np.where(a >= 0.0, 1.0, -1.0)
    return -pm1


def pose(tag):
    z = load(tag)
    return np.asarray(z['observation.eef_pose'], dtype=np.float64).reshape(-1)[:7]


print('translation is world-frame; 1 A3 unit = 0.025 m; +z up')
print('%-5s %-12s %-26s %-26s %-22s %s' % (
    'frame', 'plan', 'mean cmd first10 (m/step) xyz', 'sum first10 (m) xyz',
    'cmd z sign', 'gripper first10'))
for f in FRAMES:
    for kind in ('A0', 'P2clip_all'):
        tag = 'replan_s17_f%d_%s' % (f, kind)
        if not (OUT / (tag + '.npz')).exists():
            continue
        z = load(tag)
        cmd = a3_trans(z['raw32'])
        g = gripper(z['raw32'])
        mean_m = cmd.mean(0) * 0.025
        sum_m = (cmd * 0.025).sum(0)
        print('%-5d %-12s %-26s %-26s %-22s %s' % (
            f, kind,
            '[%+.4f %+.4f %+.4f]' % tuple(mean_m),
            '[%+.4f %+.4f %+.4f]' % tuple(sum_m),
            ''.join('+' if v > 1e-9 else ('-' if v < -1e-9 else '0') for v in sum_m),
            ''.join('C' if v < 0 else 'O' for v in g)))

print()
print('recorded execution over the same ten steps (adapter_out of the recorded chunk):')
print('%-12s %-26s %-26s %s' % ('pair', 'sum cmd (m) xyz', 'measured disp (m) xyz', 'cmd|disp ratio'))
for a, b in PAIRS:
    tag = 'replan_s17_f%d_A0' % a
    if not ((OUT / (tag + '.npz')).exists() and (OUT / ('replan_s17_f%d_A0' % b + '.npz')).exists()):
        continue
    z = load(tag)
    ad = np.asarray(z['recorded_adapter_out'], dtype=np.float64).reshape(32, 14)[:10, :3]
    cmd_m = (ad * 0.025).sum(0)
    d = pose('replan_s17_f%d_A0' % b)[:3] - pose(tag)[:3]
    ratio = np.linalg.norm(d) / max(np.linalg.norm(cmd_m), 1e-9)
    print('%-12s %-26s %-26s %.3f' % ('f%d->f%d' % (a, b),
                                     '[%+.4f %+.4f %+.4f]' % tuple(cmd_m),
                                     '[%+.4f %+.4f %+.4f]' % tuple(d), ratio))

print()
print('per-row world z command at the jam frames (m/step), A0 vs clipped:')
for f in (140, 150, 160, 200, 210):
    a = a3_trans(load('replan_s17_f%d_A0' % f)['raw32'])[:, 2] * 0.025
    b = a3_trans(load('replan_s17_f%d_P2clip_all' % f)['raw32'])[:, 2] * 0.025
    print('  f%-4d A0 %s' % (f, ' '.join('%+.4f' % v for v in a)))
    print('        CL %s' % (' '.join('%+.4f' % v for v in b)))
