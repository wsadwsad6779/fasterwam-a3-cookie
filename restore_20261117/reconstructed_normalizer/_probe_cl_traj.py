import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'closed_loop_p2clamp_20261118_v2'
CAL = json.load(open(str(ROOT.parent / 'step_040000' / 'force_calib_v1.json')))
CHANS = [int(c) for c in CAL['channels']]
NAMES = list(CAL['channel_names'])


def load(tag):
    z = np.load(str(OUT / (tag + '.npz')), allow_pickle=False)
    return {
        'f18': np.asarray(z['step_force18'], dtype=np.float64),
        'eef': np.asarray(z['step_eef_pose'], dtype=np.float64),
        'op': np.asarray(z['step_opening'], dtype=np.float64),
        'ck': np.asarray(z['step_cookies'], dtype=np.int64),
        'su': np.asarray(z['step_success'], dtype=np.int64),
        'meta': json.loads((OUT / (tag + '.json')).read_text()),
    }


b0 = load('B0')
cl = load('CL')
n = min(len(b0['eef']), len(cl['eef']))
fmax_b0 = np.abs(b0['f18'][:, CHANS]).max(1)
fmax_cl = np.abs(cl['f18'][:, CHANS]).max(1)

lines = []
lines.append('eef width %d' % b0['eef'].shape[1])
lines.append('rows B0=%d CL=%d' % (len(b0['eef']), len(cl['eef'])))
lines.append('channels %s' % (CHANS,))
lines.append('names %s' % (NAMES,))

for tag, a in (('B0', b0), ('CL', cl)):
    d = np.diff(a['ck'])
    ev = np.flatnonzero(d != 0)
    lines.append('%s cookies_final=%d transitions=%d idx=%s after=%s' % (
        tag, int(a['ck'][-1]), len(ev), ev[:20].tolist(),
        [int(a['ck'][i + 1]) for i in ev[:20]]))
    lines.append('%s success_final=%d success_any=%d' % (
        tag, int(a['su'][-1]), int(a['su'].max())))
    lines.append('%s opening pct=%s min=%.4f max=%.4f' % (
        tag, np.round(np.percentile(a['op'], [1, 5, 50, 95, 99]), 4).tolist(),
        float(a['op'].min()), float(a['op'].max())))
    m = [float(a['op'][10 * k:10 * k + 10].mean()) for k in range(len(a['op']) // 10)]
    lines.append('%s opening_chunk10=%s' % (tag, np.round(m, 4).tolist()))

lines.append('')
for tag, a in (('B0', b0), ('CL', cl)):
    f = a['f18'][:, CHANS]
    dom = np.abs(f).argmax(1)
    lines.append('%s dominant_hist=%s' % (tag, np.bincount(dom, minlength=6).tolist()))
    flips = int((np.sign(f[1:, 0]) != np.sign(f[:-1, 0])).sum())
    lines.append('%s ch0_sign_flips=%d' % (tag, flips))

lines.append('')
lines.append('%-4s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s' % (
    'f', 'CLfmax', 'B0fmax', 'CLfx', 'CLfy', 'CLfz', 'CLop', 'B0op',
    'CLez', 'B0ez', 'CLck', 'CLsu'))


def fmt(t):
    return '%-4d %9.2f %9.2f %9.2f %9.2f %9.2f %9.4f %9.4f %9.4f %9.4f %9d %9d' % (
        t, fmax_cl[t], fmax_b0[t],
        cl['f18'][t, CHANS[0]], cl['f18'][t, CHANS[1]], cl['f18'][t, CHANS[2]],
        cl['op'][t], b0['op'][t],
        cl['eef'][t, 2], b0['eef'][t, 2],
        int(cl['ck'][t]), int(cl['su'][t]))


rows = list(range(0, 211, 5)) + list(range(220, n, 20)) + [n - 1]
for t in rows:
    lines.append(fmt(t))

body = '\n'.join(lines)
print(body)
(OUT / 'traj_probe.txt').write_text(body + '\n')
