import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT/'forensics_download'
o = np.load(P/'cpu_seed17_state_force.npz', allow_pickle=False)
s = np.load(P/'seed17_steps.npz', allow_pickle=False)
cal = json.loads((ROOT.parent/'step_040000/force_calib_v1.json').read_text())
force = o['observation.force'][:, cal['channels']].astype(float)
opening = o['observation.state'][:,7].astype(float)
pose = o['observation.eef_pose'][:,:3].astype(float)
command = s['handed_14d'][:,6]
pad = np.maximum(force[:,3], force[:,4])
contact = pad > 0.5
n = len(contact)

runs = []
i = 0
while i < n:
    if contact[i]:
        j = i
        while j+1 < n and contact[j+1]:
            j += 1
        runs.append([i, j])
        i = j+1
    else:
        i += 1

def end_state(f_end):
    after = opening[f_end+1] if f_end+1 < len(opening) else opening[f_end]
    return 'open_after' if after > 0.9 else ('closed_after' if after < 0.2 else 'mid_after')

rows = []
for f0, f1 in runs:
    seg = slice(f0, f1+1)
    rows.append({'f_start': f0, 'f_end': f1, 'len': f1-f0+1,
        'pad_peak_N': float(pad[seg].max()),
        'pad_inner_peak_N': float(force[seg,3].max()), 'pad_outer_peak_N': float(force[seg,4].max()),
        'opening_start': float(opening[f0]), 'opening_end': float(opening[f1]),
        'opening_min': float(opening[seg].min()), 'opening_max': float(opening[seg].max()),
        'z_start_m': float(pose[f0,2]), 'z_end_m': float(pose[f1,2]),
        'z_min_m': float(pose[seg,2].min()), 'z_max_m': float(pose[seg,2].max()),
        'after': end_state(f1),
        'command_start': float(command[f0]) if f0 < len(command) else None,
        'command_end': float(command[f1]) if f1 < len(command) else None})

# Excursions beyond source-calibrated pad ranges (from force_calib per_channel_max).
cal_max = cal['per_channel_max']
overs = {'inner_above_calib_max_frames': int(np.sum(force[:,3] > cal_max[3])),
         'outer_above_calib_max_frames': int(np.sum(force[:,4] > cal_max[4])),
         'inner_calib_max_N': float(cal_max[3]), 'outer_calib_max_N': float(cal_max[4])}

paths = [Path(__file__), P/'cpu_seed17_state_force.npz', P/'seed17_steps.npz', ROOT.parent/'step_040000/force_calib_v1.json']
report = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256': 'UNKNOWN',
 'scope': 'CPU descriptive contact-run segmentation of stored seed17 left-arm episode. Contact is pad force >0.5N; no contact-pair identity, no simulator.',
 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 'retraction': 'An earlier segmentation used closed<0.25/open>0.65 hysteresis; contact actually occurs at median opening0.45, so the earlier zero-force closed runs did NOT show an absent grasp. That inference is retracted. Thresholds cut through the grasp range.',
 'contact_frames_total': int(contact.sum()),
 'contact_runs': rows,
 'range_overshoot': overs,
 'limits': ['Pad channels are four fingertip proxies; >0.5N is an arbitrary descriptive threshold.', 'Contact with anything (table, box, cookie, self) is indistinguishable here.', 'No object identity, no box occupancy, no per-finger joints.', 'z is EEF site height, not object height.']}
(ROOT/'SEED17_CONTACT_RUNS_CPU_20261118.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'contact_frames_total': int(contact.sum()), 'contact_runs': rows, 'range_overshoot': overs}, indent=2))
