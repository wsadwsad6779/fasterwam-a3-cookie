"""Offline replay of denormalisation + A3 adapter on the downloaded v5 predictions.

Purpose: the v5 interventions changed the raw action amplitude by one to two orders of
magnitude. What matters for control is not the amplitude but the *commanded direction*, so this
maps each raw row through the reconstructed action normaliser and the frozen adapter
(`bridge/action_adapter_v3.py`: x*2, clip +-1, gripper = -sign(pm1)) and compares signs. It is
arithmetic on saved tensors: nothing is executed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT / 'forensics_download'
STA = ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'
OUT = ROOT / 'SEED17_V5_ADAPTER_REPLAY_CPU_20261118.json'

act = json.loads(STA.read_text())['action']['default']
lo = np.asarray(act['global_min'], np.float64)
hi = np.asarray(act['global_max'], np.float64)
mean = np.asarray(act['global_mean'], np.float64)
std = np.asarray(act['global_std'], np.float64)

# Identify which denormalisation reproduces the recorded (model_out, denorm) pairs.
c = np.load(P / 'seed17_original_chunks.npz', allow_pickle=False)
raw = np.vstack([c[f'chunk_{k:03d}_model_out'][:10] for k in range(60)]).astype(np.float64)
den = np.vstack([c[f'chunk_{k:03d}_denorm'].reshape(-1, 7)[:10] for k in range(60)]).astype(np.float64)
cand = {
    'minmax_literal': lambda x: (x + 1.0) / 2.0 * (hi - lo) + lo,
    'minmax_scale': lambda x: x * ((hi - lo) / 2.0) + (hi + lo) / 2.0,
    'meanstd': lambda x: x * std + mean,
}
fit = {k: float(np.abs(f(raw) - den).max()) for k, f in cand.items()}
best = min(fit, key=fit.get)
assert fit[best] < 1e-5, fit
denorm = cand[best]

def to_cmd(raw7):
    a = denorm(raw7)
    out = np.zeros(14); out[:3] = a[:3] * 2.0
    out[3:6] = a[3:6] * 0.5 / 0.1            # world->local rotation is not applied offline
    out[6] = -np.sign(a[6])
    return np.clip(out, -1.0, 1.0)

pairs = [
    ('s17_f159', 'mechanismv5_holdout_s17_f159_baseline', 'mechanismv5_holdout_s17_f159_all6clip'),
    ('s17_f143', 'mechanismv5_robust_s17_f143_N202_baseline', 'mechanismv5_robust_s17_f143_N202_all6clip'),
    ('s11_f98', 'mechanismv5_holdout_s11_f98_baseline', 'mechanismv5_holdout_s11_f98_all6clip'),
    ('s17_f70', 'mechanismv5_onset_s17_f70_fixednoise_baseline', 'mechanismv5_onset_s17_f70_fixednoise_all6clip'),
    ('s17_f64', 'mechanismv5_onset_s17_f64_fixednoise_baseline', 'mechanismv5_onset_s17_f64_fixednoise_all6clip'),
    ('s17_f60', 'mechanismv5_onset_s17_f60_fixednoise_baseline', 'mechanismv5_onset_s17_f60_fixednoise_all6clip'),
]

rows = []
for tag, b, k in pairs:
    zb, zk = np.load(P / (b + '.npz')), np.load(P / (k + '.npz'))
    rb, rk = zb['raw32'].astype(np.float64), zk['raw32'].astype(np.float64)
    cb = np.stack([to_cmd(x) for x in rb])
    ck = np.stack([to_cmd(x) for x in rk])
    sb, sk = np.sign(cb[:10, :3]), np.sign(ck[:10, :3])
    r = {'tag': tag, 'baseline_absmax_raw': float(np.abs(rb).max()),
         'clip_absmax_raw': float(np.abs(rk).max()),
         'baseline_trans_saturated_rows': int((np.abs(cb[:10, :3]) > 0.99).any(1).sum()),
         'clip_trans_saturated_rows': int((np.abs(ck[:10, :3]) > 0.99).any(1).sum()),
         'trans_sign_agree': [int((sb[:, j] == sk[:, j]).sum()) for j in range(3)],
         'grip_cmd_baseline': cb[:10, 6].tolist(), 'grip_cmd_clip': ck[:10, 6].tolist(),
         'grip_cmd_agree': int((cb[:10, 6] == ck[:10, 6]).sum()),
         'cmd_baseline': np.round(cb[:10], 3).tolist(), 'cmd_clip': np.round(ck[:10], 3).tolist()}
    rows.append(r)

report = {
    'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
    'original_stats_sha256': 'UNKNOWN',
    'scope': 'CPU replay of the frozen adapter on downloaded v5 raw rows. No execution, no model call.',
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), STA, P / 'seed17_original_chunks.npz',
                                *sorted(P.glob('mechanismv5_*.npz'))]},
    'denorm_identification': {'max_abs_error_vs_recorded_denorm': fit, 'selected': best},
    'rotation_note': 'The adapter rotates the world rotation vector by R^T before scaling; no '
                     'recorded EEF rotation is replayed here, so rotation dims 3..5 are only '
                     'amplitude-mapped and their signs are not compared.',
    'pairs': rows,
    'limits': ['Offline arithmetic; a sign flip here says the two predicted action plans differ, '
               'not that either would succeed.',
               'The adapter gripper rule is -sign(pm1); raw->pm1 uses the model binarisation.']
}
OUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print('denorm fit', {k: round(v, 6) for k, v in fit.items()})
for r in rows:
    print(r['tag'], 'raw base %.1f clip %.1f | sat rows %d->%d | xyz sign agree %s | grip agree %d/10' %
          (r['baseline_absmax_raw'], r['clip_absmax_raw'],
           r['baseline_trans_saturated_rows'], r['clip_trans_saturated_rows'],
           r['trans_sign_agree'], r['grip_cmd_agree']))
    print('   base cmd rows0-2', r['cmd_baseline'][:3], ' clip', r['cmd_clip'][:3])
print('OUTPUT', OUT)
