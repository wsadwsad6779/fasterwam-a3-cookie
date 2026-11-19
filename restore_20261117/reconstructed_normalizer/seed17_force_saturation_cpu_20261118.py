"""Read-only CPU measurement: do the two force inputs saturate during the seed17 jam?

Path A (P1, proprio14 dims 8..13): the v1g normaliser is a min/max normaliser whose force
bounds were built as (-sigma, +sigma) (`build_v1_dataset.py:143-150`), so the mapping reduces
to residual/sigma and the model clamps at +-5. Path B (P2, force_history): the model's own
`ForceHistoryEncoder.normalize` is (raw - mean)/std, no clamp. Both are computed here from the
corpus and from the recorded episode; nothing is executed and no model is loaded.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT / 'forensics_download'
CAL = ROOT.parent / 'step_040000/force_calib_v1.json'
STA = ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'
OUT = ROOT / 'SEED17_FORCE_SATURATION_CPU_20261118.json'

cal = json.loads(CAL.read_text())
sta = json.loads(STA.read_text())['state']['default']
ch = cal['channels']
names = ['L_wrist_fx', 'L_wrist_fy', 'L_wrist_fz', 'L_pad_inner', 'L_pad_outer', 'L_gripper_act']
mean_raw = np.asarray(cal['mean'], np.float64)
std = np.asarray(cal['std'], np.float64)
tare = np.asarray(cal['zero_load_ref_median'], np.float64)
smin = np.asarray(sta['global_min'][8:14], np.float64)
smax = np.asarray(sta['global_max'][8:14], np.float64)
assert np.array_equal(smin, -smax)
std32 = smax                                                    # the v1g divisor: 2/(max-min)
assert np.allclose(std32, std.astype(np.float32).astype(np.float64), atol=0, rtol=1e-7)
assert np.allclose(np.asarray(sta['global_std'][8:14]), std, atol=0, rtol=1e-7)

f = np.load(P / 'training_source_a3_fw_force_arrays.npz', allow_pickle=False)
o = np.load(P / 'cpu_seed17_state_force.npz', allow_pickle=False)
train_raw = f['force6'].astype(np.float64)
# `force6_zeroed` in this archive is per-episode tared (implied tare constant inside each
# episode to 1e-8 but differing across episodes), so it is NOT the v1g global-tared array.
train_zeroed = train_raw - tare
assert np.allclose(train_zeroed.std(0, ddof=0), std, atol=1e-6, rtol=0)
assert np.allclose((train_raw - f['force6_zeroed']).mean(0), tare, atol=0.03, rtol=0)
dep_raw = o['observation.force'][:, ch].astype(np.float64)
dep_resid = (dep_raw.astype(np.float32) - tare).astype(np.float32).astype(np.float64)

# P1: what the v1g normaliser hands the model (clamped at +-5)
z1_train = train_zeroed / std32
z1_dep = dep_resid / std32
z1_dep_clip = np.clip(z1_dep, -5.0, 5.0)
# P2: what the force conditioner hands its linear layer (no clamp)
z2_train = (train_raw - mean_raw) / std
z2_dep = (dep_raw - mean_raw) / std

def band(a):
    return {'min': a.min(0), 'max': a.max(0), 'q01': np.percentile(a, 1, 0),
            'q99': np.percentile(a, 99, 0), 'absmax': np.abs(a).max(0)}

def over(a, thr):
    m = np.abs(a) > thr
    return {'frames_any': int(m.any(1).sum()), 'fraction_any': float(m.any(1).mean()),
            'per_channel_frames': m.sum(0), 'per_channel_fraction': m.mean(0),
            'first_frame': {names[i]: (int(np.flatnonzero(m[:, i])[0]) if m[:, i].any() else None)
                            for i in range(6)}}

train1, train2 = band(z1_train), band(z2_train)
dep1, dep2 = band(z1_dep), band(z2_dep)
train1_over5, train2_over1 = over(z1_train, 5.0), over(z2_train, 1.0)
dep1_over5 = over(z1_dep, 5.0)

def clip_loss(a):
    """L2 of what clipping removes, relative to the pre-clip vector."""
    c = np.clip(a, -5.0, 5.0)
    n = np.linalg.norm(a, axis=1)
    d = np.linalg.norm(a - c, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(n > 0, d / n, 0.0)
    return c, n, d, r

zc, n_pre, n_removed, rel = clip_loss(z1_dep)
jam = slice(134, 312)
contact_entries = [64, 72, 74, 129, 130, 134, 382, 386, 387, 391, 392]
keyframes = [0, 10, 63, 100, 128, 129, 134, 138, 142, 143, 159, 207, 260, 311, 312, 382, 391, 392, 600]

# How much of the jam window is a constant clipped vector?
sat_any = (np.abs(z1_dep) > 5.0).any(1)
sat_pads = (np.abs(z1_dep[:, 3:5]) > 5.0).all(1)
runs, run = [], None
for i, s in enumerate(sat_pads):
    if s and run is None:
        run = i
    elif not s and run is not None:
        runs.append([run, i - 1]); run = None
if run is not None:
    runs.append([run, len(sat_pads) - 1])

report = {
    'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
    'original_stats_sha256': 'UNKNOWN',
    'scope': 'Local CPU NumPy only. Reads the calibration, the reconstructed v1g stats, the '
             '42681-row force corpus and the recorded seed17 episode. No model, simulator, GPU '
             'or network. No counterfactual execution.',
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [Path(__file__), CAL, STA,
                                P / 'training_source_a3_fw_force_arrays.npz',
                                P / 'cpu_seed17_state_force.npz']},
    'semantics': {
        'P1_proprio_force_columns': 'dims 8..13 = raw6 - zero_load_ref_median, then the v1g '
                                    'min/max normaliser whose force bounds are (-sigma,+sigma), '
                                    'i.e. residual/sigma32, clamped to [-5,+5].',
        'P2_force_history': 'raw6 -> ForceHistoryEncoder.normalize = (raw - mean_raw)/std * 1.0, '
                            'no clamp. mean_raw/std come from the same calibration file.',
        'units': 'z1 and z2 are standardised units. 5.0 means five calibration sigmas.',
        'training_rows': int(train_raw.shape[0]),
    },
    'training_corpus': {'P1_normalised': train1, 'P2_normalised': train2,
                        'P1_frames_any_over_5': train1_over5,
                        'P2_frames_any_over_1': train2_over1},
    'deployment': {
        'P1_normalised_raw': dep1,
        'P1_frames_any_over_5': dep1_over5,
        'P2_normalised': dep2,
        'saturated_frames_total': int(sat_any.sum()),
        'saturated_fraction_total': float(sat_any.mean()),
        'saturated_frames_jam_134_311': int(sat_any[jam].sum()),
        'saturated_fraction_jam_134_311': float(sat_any[jam].mean()),
        'clip_removed_ratio_jam_median': float(np.median(rel[jam])),
        'clip_removed_ratio_jam_max': float(np.max(rel[jam])),
        'clip_removed_ratio_all_median': float(np.median(rel)),
        'preclip_norm_jam_median': float(np.median(n_pre[jam])),
        'preclip_norm_jam_max': float(np.max(n_pre[jam])),
        'pads_both_saturated_runs': runs,
        'pads_both_saturated_longest': max((b - a + 1) for a, b in runs) if runs else 0,
        'ratio_of_peak_to_training_absmax_P2': (np.abs(z2_dep).max(0) / np.abs(z2_train).max(0)).tolist(),
    },
    'contact_entry_frames': [
        {'frame': i, 'P1_z1': z1_dep[i], 'P1_clipped': z1_dep_clip[i],
         'P1_removed_ratio': float(rel[i]), 'P2_z2': z2_dep[i],
         'P1_channels_saturated': [names[j] for j in range(6) if abs(z1_dep[i, j]) > 5.0]}
        for i in contact_entries],
    'keyframes': [
        {'frame': i, 'P1_z1': z1_dep[i], 'P1_clipped': z1_dep_clip[i],
         'P1_removed_ratio': float(rel[i]), 'P2_z2': z2_dep[i]}
        for i in keyframes],
    'limits': [
        'A saturated input is a measured interface fact, not proof that it caused any action.',
        'The P1 clamp is applied after normalisation in the model normaliser; the exact call site '
        'is action_norm_identity40k.py:82 and has not been re-derived for this file.',
        'The P2 ring buffer feeds raw newtons; the model normalises internally, so the 500-sigma '
        'value reaches the encoder unless the model itself guards it (it does not).',
        'No intervention was run here; whether the model response changes is still open.',
    ],
}

def jsonable(a):
    if isinstance(a, np.ndarray):
        return [None if not np.isfinite(x) else float(x) for x in a.tolist()]
    return a

OUT.write_text(json.dumps(report, indent=2, default=jsonable, allow_nan=False), encoding='utf-8')

print('== training corpus, P1 (residual/sigma32) ==')
for i, n in enumerate(names):
    print('  %-16s min%9.3f max%9.3f | max%9.3f | frames>5: %d' %
          (n, train1['min'][i], train1['max'][i], train1['absmax'][i], train1_over5['per_channel_frames'][i]))
print('== training corpus, P2 ((raw-mean)/std) ==')
for i, n in enumerate(names):
    print('  %-16s min%9.3f max%9.3f | frames>1: %d' %
          (n, train2['min'][i], train2['max'][i], train2_over1['per_channel_frames'][i]))
print('== deployment, P1 ==')
for i, n in enumerate(names):
    f0 = dep1_over5['first_frame'][n]
    print('  %-16s min%10.2f max%10.2f | frames>5: %d (%.1f%%) first=%s' %
          (n, dep1['min'][i], dep1['max'][i], dep1_over5['per_channel_frames'][i],
           100 * dep1_over5['per_channel_fraction'][i], f0))
print('== deployment, P2 ==')
for i, n in enumerate(names):
    print('  %-16s min%10.2f max%10.2f | peak/train-max %.1fx' %
          (n, dep2['min'][i], dep2['max'][i], report['deployment']['ratio_of_peak_to_training_absmax_P2'][i]))
print('== saturation structure ==')
print('  frames with >=1 P1 channel over 5: %d/%d ; jam f134..311: %d/%d (%.1f%%)' %
      (sat_any.sum(), len(sat_any), sat_any[jam].sum(), jam.stop - jam.start, 100 * sat_any[jam].mean()))
print('  both pads saturated runs (longest first):', sorted(runs, key=lambda r: r[0] - r[1])[:6])
print('  jam clip-removed L2 fraction: median %.4f max %.4f ; pre-clip norm median %.1f max %.1f' %
      (np.median(rel[jam]), rel[jam].max(), np.median(n_pre[jam]), n_pre[jam].max()))
print('== contact entries ==')
for e in report['contact_entry_frames'][:12]:
    print('  f%-4d P1=%s P2_pads=(%.2f,%.2f) removed=%.4f sat=%s' %
          (e['frame'], np.round(e['P1_z1'], 2), e['P2_z2'][3], e['P2_z2'][4],
           e['P1_removed_ratio'], e['P1_channels_saturated']))
print('OUTPUT', OUT)
