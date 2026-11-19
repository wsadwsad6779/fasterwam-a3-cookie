"""Observational coupling between measured pad force and recorded raw action amplitude.

Command i is computed from observation frame i (observation f follows command i=f-1), and its
raw model row is chunk i//10, row i%10 in `seed17_original_chunks.npz`. Correlating the two is
descriptive: it cannot separate force from the state/video that produced the contact.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT / 'forensics_download'
CAL = ROOT.parent / 'step_040000/force_calib_v1.json'
OUT = ROOT / 'SEED17_FORCE_ACTION_COUPLING_CPU_20261118.json'

cal = json.loads(CAL.read_text())
ch = cal['channels']
s = np.load(P / 'seed17_steps.npz', allow_pickle=False)
c = np.load(P / 'seed17_original_chunks.npz', allow_pickle=False)
o = np.load(P / 'cpu_seed17_state_force.npz', allow_pickle=False)

force = o['observation.force'][:600, ch].astype(np.float64)
pad = np.maximum(force[:, 3], force[:, 4])
raw = np.vstack([c[f'chunk_{k:03d}_model_out'][:10] for k in range(60)])
adapter = s['handed_14d'].astype(np.float64)
assert raw.shape == (600, 7) and adapter.shape[0] == 600
absmax = np.abs(raw).max(1)
trans_absmax = np.abs(adapter[:, :3]).max(1)

i = np.arange(600)
frames = [0, 5, 10, 20, 30, 40, 50, 59, 60, 63, 64, 70, 79, 80, 100, 120, 129,
          133, 134, 139, 143, 159, 179, 199, 207, 249, 299, 310, 311, 312, 350, 382, 391, 392]

def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

lags = {}
for L in range(-6, 7):
    if L >= 0:
        a, b = pad[:600 - L], absmax[L:]
    else:
        a, b = pad[-L:], absmax[:600 + L]
    lags[str(L)] = corr(a, b)

pre_contact = i <= 60
jam = (i >= 134) & (i <= 311)
traj = [
    {'i': int(k), 'pad_max_N': float(pad[k]), 'wrist_fz_N': float(force[k, 2]),
     'raw_absmax': float(absmax[k]), 'adapter_trans_absmax': float(trans_absmax[k]),
     'gripper_raw': float(raw[k, 6]), 'gripper_cmd': float(adapter[k, 6])}
    for k in frames]

report = {
    'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
    'original_stats_sha256': 'UNKNOWN',
    'scope': 'Read-only CPU arithmetic on the recorded episode and recorded chunk outputs. No model call, no execution.',
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), CAL, P / 'seed17_steps.npz',
                                P / 'seed17_original_chunks.npz', P / 'cpu_seed17_state_force.npz']},
    'values': {'pre_contact_i0_60': {'pad_max_med': float(np.median(pad[pre_contact])),
                                     'pad_max_max': float(pad[pre_contact].max()),
                                     'raw_absmax_med': float(np.median(absmax[pre_contact])),
                                     'raw_absmax_max': float(absmax[pre_contact].max())},
               'jam_i134_311': {'pad_max_med': float(np.median(pad[jam])),
                                'pad_max_max': float(pad[jam].max()),
                                'raw_absmax_med': float(np.median(absmax[jam])),
                                'raw_absmax_max': float(absmax[jam].max()),
                                'adapter_trans_saturated_fraction': float((trans_absmax[jam] > 0.99).mean())},
               'whole_episode': {'raw_absmax_med': float(np.median(absmax)),
                                 'raw_absmax_max': float(absmax.max()),
                                 'adapter_trans_saturated_fraction': float((trans_absmax > 0.99).mean())}},
    'correlation_same_frame': corr(pad, absmax),
    'correlation_pad_vs_absmax_by_lag': lags,
    'lag_convention': 'lag L>0 compares pad[i] with absmax[i+L]: force leads the action row by L steps.',
    'trajectory': traj,
    'limits': ['Observational only: contact, state and images co-vary.',
               'Raw model rows are the recorded executed-chunk outputs; rows 10..31 were never executed.',
               'Adapter saturation is a command saturation, not a measured motion.'],
}
OUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print('pre-contact i0..60  pad max<=%.2fN  raw absmax med %.2f max %.2f' %
      (report['values']['pre_contact_i0_60']['pad_max_max'],
       report['values']['pre_contact_i0_60']['raw_absmax_med'],
       report['values']['pre_contact_i0_60']['raw_absmax_max']))
print('jam i134..311       pad max med %.1fN max %.1fN  raw absmax med %.2f max %.1f  adapter sat %.1f%%' %
      (report['values']['jam_i134_311']['pad_max_med'], report['values']['jam_i134_311']['pad_max_max'],
       report['values']['jam_i134_311']['raw_absmax_med'], report['values']['jam_i134_311']['raw_absmax_max'],
       100 * report['values']['jam_i134_311']['adapter_trans_saturated_fraction']))
print('same-frame corr(pad, raw absmax) =', None if report['correlation_same_frame'] is None else round(report['correlation_same_frame'], 3))
print('lag corr:', {k: (None if v is None else round(v, 3)) for k, v in lags.items()})
for t in traj:
    print('  i%-3d pad%8.2f fz%9.2f rawmax%8.2f transsat%5.2f grip_raw%9.2f cmd%5.1f' %
          (t['i'], t['pad_max_N'], t['wrist_fz_N'], t['raw_absmax'], t['adapter_trans_absmax'],
           t['gripper_raw'], t['gripper_cmd']))
print('OUTPUT', OUT)
