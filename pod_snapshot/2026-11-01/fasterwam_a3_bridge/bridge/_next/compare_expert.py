import numpy as np
from pathlib import Path

BIN = np.array([0.095, 0.100, 0.753])
SRC = np.array([0.175, 0.315])


def stats(tag, eef, opening):
    eef = np.asarray(eef, float)[:, :3]
    opening = np.asarray(opening, float)
    closed = opening < 0.6
    d_bin = np.linalg.norm(eef - BIN, axis=1)
    d_src = np.linalg.norm(eef[:, :2] - SRC[:2], axis=1)
    lifted = (eef[:, 2] > 0.86) & closed
    zc = eef[closed, 2].max() if closed.any() else float('nan')
    print(f'{tag}: n={len(opening)} open[min/mean/end]='
          f'{opening.min():.3f}/{opening.mean():.3f}/{opening[-1]:.3f} '
          f'closed%={100*closed.mean():.1f} zmax={eef[:,2].max():.3f} '
          f'z@closed_max={zc:.3f} lift_closed%={100*lifted.mean():.1f} '
          f'd_bin_min={d_bin.min():.3f} d_src_min={d_src.min():.3f}')


for seed in (11, 21, 22):
    ex = Path(f'/workspace/expert_seed{seed}.npz')
    if ex.exists():
        d = np.load(ex)
        stats(f'expert/{seed}', d['eef'], d['opening'])
    m = Path(f'/workspace/fasterwam_a3_bridge/bridge/outputs/E0_C0/lora_C0_{seed}.npz')
    if m.exists():
        d = np.load(m)
        stats(f'c0/{seed}', d['eef_xyz'], d['opening'])
