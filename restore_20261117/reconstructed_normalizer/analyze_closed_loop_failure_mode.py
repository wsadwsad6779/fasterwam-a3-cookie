"""Closed-loop forensic dissection: per-channel OOD z-scores, and why the P2-clamped arm
still fails (grasp / lift / placement, commanded vs executed).

CPU only, reads closed_loop_p2clamp_20261118_v2. No model, no simulator.
"""
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'closed_loop_p2clamp_20261118_v2'
CAL = json.load(io.open(str(ROOT.parent / 'step_040000' / 'force_calib_v1.json'), encoding='utf-8'))
CHANS = [int(c) for c in CAL['channels']]
NAMES = list(CAL['channel_names'])
MEAN = np.asarray(CAL['mean'], dtype=np.float64)
STD = np.asarray(CAL['std'], dtype=np.float64)
LO = np.asarray(CAL['per_channel_min'], dtype=np.float64)
HI = np.asarray(CAL['per_channel_max'], dtype=np.float64)
MARK = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
        'original_stats_sha256': 'UNKNOWN'}


def load(tag):
    z = np.load(str(OUT / (tag + '.npz')), allow_pickle=False)
    return {
        'f18': np.asarray(z['step_force18'], dtype=np.float64),
        'eef': np.asarray(z['step_eef_pose'], dtype=np.float64),
        'op': np.asarray(z['step_opening'], dtype=np.float64),
        'ck': np.asarray(z['step_cookies'], dtype=np.int64),
        'su': np.asarray(z['step_success'], dtype=np.int64),
    }


def chunk_arrays(z, n):
    cmd = np.zeros((n, 3))
    grip = np.zeros(n)
    rot = np.zeros((n, 3))
    for k in range((n + 9) // 10):
        key = 'chunk_%03d_adapter_out' % k
        if key not in z.files:
            break
        ad = np.asarray(z[key], dtype=np.float64).reshape(-1, 14)
        m = min(10, n - 10 * k)
        cmd[10 * k:10 * k + m] = ad[:m, :3] * 0.025
        rot[10 * k:10 * k + m] = ad[:m, 3:6]
        grip[10 * k:10 * k + m] = ad[:m, 6]
    return cmd, grip, rot


def runs(mask):
    best = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        best.append((i, j - 1, j - i))
        i = j
    return best


b0, cl = load('B0'), load('CL')
n = min(len(b0['eef']), len(cl['eef']))
cmd_b0, grip_b0, rot_b0 = chunk_arrays(np.load(str(OUT / 'B0.npz')), n)
cmd_cl, grip_cl, rot_cl = chunk_arrays(np.load(str(OUT / 'CL.npz')), n)

res = {**MARK, 'scope': 'closed-loop simulator, single seed 17, B0 vs P2-clamp'}
res['per_channel_ood'] = {}
for i, (ch, name) in enumerate(zip(CHANS, NAMES)):
    row = {'channel': int(ch), 'name': name,
           'corpus_min': float(LO[i]), 'corpus_max': float(HI[i]),
           'calib_mean': float(MEAN[i]), 'calib_std': float(STD[i]),
           'corpus_min_z': float((LO[i] - MEAN[i]) / STD[i]),
           'corpus_max_z': float((HI[i] - MEAN[i]) / STD[i])}
    for tag, a in (('B0', b0), ('CL', cl)):
        v = a['f18'][:, ch]
        tmin, tmax = float(v.min()), float(v.max())
        row[tag] = {
            'observed_min': tmin, 'observed_max': tmax,
            'z_min': (tmin - MEAN[i]) / STD[i], 'z_max': (tmax - MEAN[i]) / STD[i],
            'frames_outside_corpus': int(((v < LO[i]) | (v > HI[i])).sum()),
            'frames_outside_3sigma': int((np.abs((v - MEAN[i]) / STD[i]) > 3).sum()),
        }
    res['per_channel_ood'][name] = row

# free-space baseline: frames 0..60 (both arms identical there)
free = np.abs(b0['f18'][:61][:, CHANS])
res['free_space_baseline'] = {
    'frames': '0..60', 'per_channel_mean': free.mean(0).tolist(),
    'per_channel_absmax': free.max(0).tolist(),
    'note': 'fy carries a static ~-10.7 N tare; the whole corpus band for fy is '
            '[-14.27,-9.45] N, i.e. the channel has almost no dynamic range in training',
}

# commanded vs executed, per chunk
rows = []
for k in range(n // 10):
    sl = slice(10 * k, 10 * k + 10)
    e = {}
    for tag, cmd, disp in (('B0', cmd_b0, b0), ('CL', cmd_cl, cl)):
        d = np.zeros_like(cmd)
        d[:-1] = disp['eef'][1:, :3] - disp['eef'][:-1, :3]
        e[tag] = {
            'cmd_sum_mm': (cmd[sl].sum(0) * 1000).tolist(),
            'disp_sum_mm': (d[sl].sum(0) * 1000).tolist(),
            'cmd_up_mm': float(cmd[sl, 2].clip(min=0).sum() * 1000),
        }
    rows.append({'chunk': k, 'frame': 10 * k,
                 'B0': e['B0'], 'CL': e['CL'],
                 'B0_z': float(b0['eef'][10 * k, 2]), 'CL_z': float(cl['eef'][10 * k, 2]),
                 'B0_op': float(b0['op'][10 * k]), 'CL_op': float(cl['op'][10 * k])})
res['chunk_cmd_vs_disp'] = rows

# grasp segments: opening below threshold
for tag, a in (('B0', b0), ('CL', cl)):
    held = a['op'] < 0.30
    segs = runs(held)
    segs = [s for s in segs if s[2] >= 5]
    res['%s_grasp_segments_opening_lt_0.30' % tag] = [
        {'start': int(s), 'end': int(e), 'len': int(l),
         'op_min': float(a['op'][s:e + 1].min()),
         'z_start': float(a['eef'][s, 2]), 'z_end': float(a['eef'][e, 2]),
         'z_max_in_seg': float(a['eef'][s:e + 1, 2].max()),
         'xy_start': a['eef'][s, :2].tolist(), 'xy_end': a['eef'][e, :2].tolist()}
        for s, e, l in segs]

res['lift_stats'] = {}
for tag, a in (('B0', b0), ('CL', cl)):
    z = a['eef'][:, 2]
    res['lift_stats'][tag] = {
        'z_start': float(z[0]), 'z_end': float(z[-1]),
        'z_min': float(z.min()), 'z_max': float(z.max()),
        'frame_zmax': int(np.argmax(z)),
        'z_above_1.0_frames': int((z > 1.0).sum()),
        'z_above_1.3_frames': int((z > 1.3).sum()),
    }

# commanded lift near the end, both arms
res['commanded_lift_last200'] = {}
for tag, cmd in (('B0', cmd_b0), ('CL', cmd_cl)):
    tail = cmd[-200:]
    res['commanded_lift_last200'][tag] = {
        'sum_cmd_mm': (tail.sum(0) * 1000).tolist(),
        'sum_up_mm': float(tail[:, 2].clip(min=0).sum() * 1000),
        'sum_down_mm': float(tail[:, 2].clip(max=0).sum() * 1000),
    }

# CL: what does the model command while holding? (frames 200..290 first long hold)
sl = slice(200, 290)
res['CL_first_hold_200_290'] = {
    'cmd_sum_mm': (cmd_cl[sl].sum(0) * 1000).tolist(),
    'disp_sum_mm': ((cl['eef'][201:291, :3] - cl['eef'][200:290, :3]).sum(0) * 1000).tolist(),
    'grip_cmd_mean': float(grip_cl[sl].mean()),
    'op_range': [float(cl['op'][sl].min()), float(cl['op'][sl].max())],
}

# force event localisation in CL: the f80-120 ram
sl = slice(78, 122)
res['CL_ram_f78_122'] = {
    'fmax': float(np.abs(cl['f18'][sl][:, CHANS]).max()),
    'per_channel_absmax': np.abs(cl['f18'][sl][:, CHANS]).max(0).tolist(),
    'eef_xyz_start': cl['eef'][78, :3].tolist(),
    'eef_xyz_end': cl['eef'][121, :3].tolist(),
    'cmd_sum_mm': (cmd_cl[sl].sum(0) * 1000).tolist(),
    'disp_sum_mm': ((cl['eef'][79:123, :3] - cl['eef'][78:122, :3]).sum(0) * 1000).tolist(),
    'op_range': [float(cl['op'][sl].min()), float(cl['op'][sl].max())],
    'grip_cmd_mean': float(grip_cl[sl].mean()),
}
sl = slice(80, 125)
res['B0_same_window_f80_125'] = {
    'fmax': float(np.abs(b0['f18'][sl][:, CHANS]).max()),
    'eef_xyz_start': b0['eef'][80, :3].tolist(),
    'eef_xyz_end': b0['eef'][124, :3].tolist(),
}

io.open(str(OUT / 'analysis_failure_mode.json'), 'w', encoding='utf-8').write(
    json.dumps(res, indent=2, default=str, ensure_ascii=False))

# human-readable
L = []
L.append('per-channel corpus support vs observed (z = (f-mean)/std)')
L.append('%-14s %9s %9s %8s %8s | %9s %9s %8s %8s | %9s %9s %8s %8s' % (
    'channel', 'lo', 'hi', 'z_lo', 'z_hi',
    'B0min', 'B0max', 'B0zmin', 'B0zmax', 'CLmin', 'CLmax', 'CLzmin', 'CLzmax'))
for name in res['per_channel_ood']:
    r = res['per_channel_ood'][name]
    L.append('%-14s %9.3f %9.3f %8.2f %8.2f | %9.2f %9.2f %8.1f %8.1f | %9.2f %9.2f %8.1f %8.1f' % (
        name, r['corpus_min'], r['corpus_max'], r['corpus_min_z'], r['corpus_max_z'],
        r['B0']['observed_min'], r['B0']['observed_max'], r['B0']['z_min'], r['B0']['z_max'],
        r['CL']['observed_min'], r['CL']['observed_max'], r['CL']['z_min'], r['CL']['z_max']))
L.append('')
L.append('frames outside corpus support / |z|>3')
for name in res['per_channel_ood']:
    r = res['per_channel_ood'][name]
    L.append('%-14s B0 %4d/%-4d   CL %4d/%-4d' % (
        name, r['B0']['frames_outside_corpus'], r['B0']['frames_outside_3sigma'],
        r['CL']['frames_outside_corpus'], r['CL']['frames_outside_3sigma']))
L.append('')
L.append('lift stats')
for tag in ('B0', 'CL'):
    L.append('%-3s %s' % (tag, json.dumps(res['lift_stats'][tag])))
L.append('')
L.append('grasp segments (opening < 0.30, >=5 frames)')
for tag in ('B0', 'CL'):
    L.append('%-3s n=%d' % (tag, len(res['%s_grasp_segments_opening_lt_0.30' % tag])))
    for s in res['%s_grasp_segments_opening_lt_0.30' % tag][:8]:
        L.append('    f%-4d..%-4d len%-3d opmin %.3f z %.3f->%.3f (zmax %.3f) xy %s -> %s' % (
            s['start'], s['end'], s['len'], s['op_min'], s['z_start'], s['z_end'],
            s['z_max_in_seg'], np.round(s['xy_start'], 3).tolist(), np.round(s['xy_end'], 3).tolist()))
L.append('')
L.append('CL ram f78..121 %s' % json.dumps(res['CL_ram_f78_122']))
L.append('B0 f80..124   %s' % json.dumps(res['B0_same_window_f80_125']))
L.append('')
L.append('CL first hold 200..290 %s' % json.dumps(res['CL_first_hold_200_290']))
L.append('commanded lift last200 %s' % json.dumps(res['commanded_lift_last200']))
io.open(str(OUT / 'analysis_failure_mode.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))